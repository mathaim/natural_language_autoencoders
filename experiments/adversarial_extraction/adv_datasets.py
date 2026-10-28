"""Adversarial dataset loaders — unified Sample interface.

Each loader returns list[Sample]. The `text` field is the model-facing
prompt (what gets tokenized and run through the base model). Metadata
fields are kept for downstream analysis but are NOT part of the prompt.

Datasets
────────
wmdp       cais/wmdp  (bio / chem / cyber configs)
harmbench  raw CSV from centerforaisafety/HarmBench GitHub
advbench   raw CSV from llm-attacks/llm-attacks GitHub
truthfulqa truthfulqa/truthful_qa  (generation config)
"""

from __future__ import annotations

import csv
import io
from dataclasses import dataclass, field
from typing import Any

import requests


@dataclass
class Sample:
    dataset: str          # "wmdp", "harmbench", "advbench", "truthfulqa"
    sample_id: str        # unique within the dataset
    text: str             # prompt fed to the model (tokenized as-is)
    meta: dict = field(default_factory=dict)   # dataset-specific extras


# ─── WMDP ────────────────────────────────────────────────────────────────────
# cais/wmdp  —  configs: wmdp-bio, wmdp-chem, wmdp-cyber  —  split: test
#
# Format choices into the prompt so the model sees a realistic MCQ. The
# correct answer index is stored in meta but NOT appended to the prompt
# (we want to capture what the model would predict, not cue it).

WMDP_SUBSETS = ("wmdp-bio", "wmdp-chem", "wmdp-cyber")
_WMDP_OPTION_LETTERS = ("A", "B", "C", "D")


def _wmdp_format(question: str, choices: list[str]) -> str:
    opts = "\n".join(
        f"{letter}. {choice}"
        for letter, choice in zip(_WMDP_OPTION_LETTERS, choices)
    )
    return f"{question}\n{opts}\nAnswer:"


def load_wmdp(
    subsets: tuple[str, ...] = WMDP_SUBSETS,
    limit: int | None = None,
) -> list[Sample]:
    from datasets import load_dataset

    samples: list[Sample] = []
    for subset in subsets:
        ds = load_dataset("cais/wmdp", subset, split="test")
        for i, row in enumerate(ds):
            if limit and len(samples) >= limit:
                break
            samples.append(Sample(
                dataset="wmdp",
                sample_id=f"{subset}_{i}",
                text=_wmdp_format(row["question"], row["choices"]),
                meta={
                    "subset": subset,
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer_idx": int(row["answer"]),
                    "answer_letter": _WMDP_OPTION_LETTERS[int(row["answer"])],
                },
            ))
        if limit and len(samples) >= limit:
            break
    return samples


# ─── HarmBench ───────────────────────────────────────────────────────────────
# Raw CSV from centerforaisafety/HarmBench (no HF auth required via raw URL).
# We use the `Behavior` column as the prompt text.

_HARMBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/centerforaisafety/HarmBench"
    "/main/data/behavior_datasets/harmbench_behaviors_text_all.csv"
)


def load_harmbench(
    url: str = _HARMBENCH_CSV_URL,
    limit: int | None = None,
    functional_categories: tuple[str, ...] | None = None,
    semantic_categories: tuple[str, ...] | None = None,
) -> list[Sample]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    samples: list[Sample] = []
    for row in reader:
        if functional_categories and row["FunctionalCategory"] not in functional_categories:
            continue
        if semantic_categories and row["SemanticCategory"] not in semantic_categories:
            continue
        text = row["Behavior"].strip()
        if row.get("ContextString", "").strip():
            # contextual rows: prepend the context passage
            text = row["ContextString"].strip() + "\n\n" + text
        samples.append(Sample(
            dataset="harmbench",
            sample_id=row["BehaviorID"],
            text=text,
            meta={
                "behavior": row["Behavior"],
                "functional_category": row["FunctionalCategory"],
                "semantic_category": row["SemanticCategory"],
                "context_string": row.get("ContextString", ""),
                "tags": row.get("Tags", ""),
            },
        ))
        if limit and len(samples) >= limit:
            break
    return samples


# ─── AdvBench ────────────────────────────────────────────────────────────────
# Raw CSV from llm-attacks/llm-attacks. `goal` is the harmful instruction.
# `target` (the compliance opener) is saved in meta, not appended to prompt.

_ADVBENCH_CSV_URL = (
    "https://raw.githubusercontent.com/llm-attacks/llm-attacks"
    "/main/data/advbench/harmful_behaviors.csv"
)


def load_advbench(
    url: str = _ADVBENCH_CSV_URL,
    limit: int | None = None,
) -> list[Sample]:
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    reader = csv.DictReader(io.StringIO(resp.text))
    samples: list[Sample] = []
    for i, row in enumerate(reader):
        samples.append(Sample(
            dataset="advbench",
            sample_id=f"advbench_{i}",
            text=row["goal"].strip(),
            meta={
                "goal": row["goal"],
                "target": row["target"],
            },
        ))
        if limit and len(samples) >= limit:
            break
    return samples


# ─── TruthfulQA ──────────────────────────────────────────────────────────────
# truthfulqa/truthful_qa  —  config: generation  —  split: validation
# We use the `question` column as the prompt.

def load_truthfulqa(
    limit: int | None = None,
    adversarial_only: bool = False,
) -> list[Sample]:
    from datasets import load_dataset

    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    samples: list[Sample] = []
    for i, row in enumerate(ds):
        if adversarial_only and row.get("type") != "Adversarial":
            continue
        samples.append(Sample(
            dataset="truthfulqa",
            sample_id=f"truthfulqa_{i}",
            text=row["question"].strip(),
            meta={
                "best_answer": row.get("best_answer", ""),
                "correct_answers": row.get("correct_answers", []),
                "incorrect_answers": row.get("incorrect_answers", []),
                "category": row.get("category", ""),
                "type": row.get("type", ""),
                "source": row.get("source", ""),
            },
        ))
        if limit and len(samples) >= limit:
            break
    return samples


# ─── Registry ────────────────────────────────────────────────────────────────

DATASET_NAMES = ("wmdp", "harmbench", "advbench", "truthfulqa")


def load_dataset_by_name(
    name: str,
    limit: int | None = None,
    **kwargs: Any,
) -> list[Sample]:
    if name == "wmdp":
        subsets = kwargs.pop("wmdp_subsets", WMDP_SUBSETS)
        return load_wmdp(subsets=subsets, limit=limit, **kwargs)
    if name == "harmbench":
        return load_harmbench(limit=limit, **kwargs)
    if name == "advbench":
        return load_advbench(limit=limit, **kwargs)
    if name == "truthfulqa":
        return load_truthfulqa(limit=limit, **kwargs)
    raise ValueError(f"unknown dataset {name!r}; choices: {DATASET_NAMES}")
