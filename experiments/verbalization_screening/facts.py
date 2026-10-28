"""Factual probe dataset for the verbalization-screening experiment.

The screening test asks: when we extract an activation from a position where a
model is recalling a fact, does the AV's verbalization actually mention that
fact? To answer it cleanly we need facts that are:

  - single-token answers  (so "swap Tokyo->Paris" is a one-token edit later)
  - high-confidence        (the base model reliably predicts the answer, so the
                            fact is genuinely present in the activation)
  - minimal-pair structured (all share one template, differing only in the
                            subject -> the eventual edit is isolated)

Country -> capital is the cleanest such relation. We ship a built-in list and
filter it to single-token capitals *at runtime* against the live tokenizer
(token counts differ across models), rather than hard-coding a tokenization
assumption. A CounterFact loader is included for scaling up later — CounterFact
(Meng et al., ROME) is exactly this minimal-pair / counterfactual-target format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# Template shared by every fact. Ends WITHOUT a trailing space so the model's
# next-token prediction is the answer with its leading-space tokenization
# (" Paris"). The shared template is what makes these minimal pairs.
CAPITAL_TEMPLATE = "The capital of {subject} is"


@dataclass(frozen=True)
class Fact:
    subject: str          # e.g. "France"        — the swappable entity
    answer: str           # e.g. "Paris"         — single-token target
    prompt: str           # e.g. "The capital of France is"
    relation: str = "capital"


# Country -> capital. Generously sized; the single-token filter trims it to
# whatever the live tokenizer encodes as one token. Capitals chosen to be
# common, unambiguous, and short (most are 1 token under Qwen/Llama BPE).
_CAPITALS: list[tuple[str, str]] = [
    ("France", "Paris"), ("Japan", "Tokyo"), ("England", "London"),
    ("Italy", "Rome"), ("Germany", "Berlin"), ("Spain", "Madrid"),
    ("Egypt", "Cairo"), ("Russia", "Moscow"), ("Greece", "Athens"),
    ("Austria", "Vienna"), ("Norway", "Oslo"), ("Ireland", "Dublin"),
    ("Portugal", "Lisbon"), ("Poland", "Warsaw"), ("Thailand", "Bangkok"),
    ("Peru", "Lima"), ("Cuba", "Havana"), ("Iran", "Tehran"),
    ("Iraq", "Baghdad"), ("Turkey", "Ankara"), ("Kenya", "Nairobi"),
    ("Hungary", "Budapest"), ("Sweden", "Stockholm"), ("Denmark", "Copenhagen"),
    ("Finland", "Helsinki"), ("Scotland", "Edinburgh"), ("Canada", "Ottawa"),
    ("Lebanon", "Beirut"), ("Jamaica", "Kingston"), ("Syria", "Damascus"),
    ("Afghanistan", "Kabul"), ("Ukraine", "Kyiv"), ("Chile", "Santiago"),
    ("Venezuela", "Caracas"), ("Morocco", "Rabat"), ("Algeria", "Algiers"),
    ("Vietnam", "Hanoi"), ("Indonesia", "Jakarta"), ("Pakistan", "Islamabad"),
    ("Nigeria", "Abuja"),
]


def builtin_capitals() -> list[Fact]:
    """All built-in country->capital facts (unfiltered)."""
    return [
        Fact(subject=c, answer=a, prompt=CAPITAL_TEMPLATE.format(subject=c))
        for c, a in _CAPITALS
    ]


def is_single_token_answer(tokenizer: Any, answer: str) -> bool:
    """True if `answer` encodes to exactly one token in answer-position form.

    Answer position is mid-sentence after a space, so the natural encoding is
    " Paris" (leading space). We check that form — it's what the base model
    actually predicts and what a later token-swap edit would manipulate.
    """
    ids = tokenizer.encode(" " + answer, add_special_tokens=False)
    return len(ids) == 1


def filter_single_token(facts: list[Fact], tokenizer: Any) -> tuple[list[Fact], list[Fact]]:
    """Split facts into (kept single-token, dropped multi-token)."""
    kept, dropped = [], []
    for f in facts:
        (kept if is_single_token_answer(tokenizer, f.answer) else dropped).append(f)
    return kept, dropped


def load_counterfact(
    path: str | Path, template_relation: str | None = None, limit: int | None = None
) -> list[Fact]:
    """Load CounterFact (ROME) JSON into Fact records.

    CounterFact entries look like:
        {"requested_rewrite": {"prompt": "The capital of {} is",
                               "subject": "France",
                               "target_true": {"str": "Paris"}, ...}, ...}

    `template_relation`: if set, keep only entries whose prompt contains this
        substring (e.g. "capital of") so you screen one relation at a time.
    """
    records = json.loads(Path(path).read_text())
    facts: list[Fact] = []
    for rec in records:
        rw = rec["requested_rewrite"]
        prompt_tmpl, subject = rw["prompt"], rw["subject"]
        answer = rw["target_true"]["str"]
        if template_relation and template_relation not in prompt_tmpl:
            continue
        facts.append(Fact(
            subject=subject,
            answer=answer,
            prompt=prompt_tmpl.format(subject),
            relation=prompt_tmpl,
        ))
        if limit and len(facts) >= limit:
            break
    return facts
