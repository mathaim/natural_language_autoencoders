"""SAEBench sparse probing, adapted to NLAs.

SAEBench asks: can a concept (profession, topic, sentiment, language) be
classified from only K SAE latents? An NLA has no latent dictionary — its
bottleneck is text — so the "latents" here are the WORDS of the verbalization.
For every (dataset, class, layer) one-vs-rest task this reports:

  llm_test_accuracy            probe on the raw activation          (ceiling)
  llm_top_{k}_test_accuracy    probe on the top-k activation dims   (SAEBench baseline)
  nla_top_{k}_test_accuracy    probe on the top-k verbalization words   ← the metric
  nla_test_accuracy            probe on all verbalization words     (text ceiling)
  recon_test_accuracy          probe trained + tested on AR reconstructions
  recon_transfer_test_accuracy raw-activation probe applied to AR reconstructions

Top-k selection is SAEBench's rule: largest |mean(positive) - mean(negative)|
on the training split. Word features are binary presence ("the latent fired").

Two subcommands; the GPU work in between is the existing all-layer pipeline:

  prepare   Sample class-balanced train/test examples from the SAEBench
            datasets → examples.jsonl. Needs network. No GPU.

  (run_multilayer.py extract   → activations.npy    [n, n_layers, d])
  (run_multilayer.py roundtrip → verbalizations.jsonl, reconstructions.npy)

  probe     Fit the probes above. CPU only. → sparse_probing.json / .csv

Differences from SAEBench, all deliberate:
  - One activation per example (last token of the 128-token context), not a
    mean over tokens: an NLA verbalizes a single activation.
  - Examples are filtered to >= --min-chars so that token sits past position
    50 (the NLA's training range). Europarl sentences are joined to reach it.
  - 100 train / 50 test per class by default (SAEBench: 2000 / 500), because
    every example costs one generation per layer.
  - All probes are L2 logistic regression fit to convergence. SAEBench's
    full-dimension probe early-stops on test accuracy; this does not.
  - codeparrot/github-code is not included (script-based loader).

Example:

  python experiments/saebench_metrics/sparse_probing.py prepare \\
      --out /scratch/$USER/saebench_nla/sp_v1

  python experiments/saebench_metrics/sparse_probing.py probe \\
      --examples /scratch/$USER/saebench_nla/sp_v1/examples.jsonl \\
      --acts     /scratch/$USER/saebench_nla/sp_v1/activations.npy \\
      --run      /scratch/$USER/saebench_nla/sp_v1/c_direct
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

# ─── datasets (IDs, columns and class sets as in SAEBench dataset_info.py) ───

_PROFESSIONS = [
    "accountant", "architect", "attorney", "chiropractor", "comedian",
    "composer", "dentist", "dietitian", "dj", "filmmaker", "interior_designer",
    "journalist", "model", "nurse", "painter", "paralegal", "pastor",
    "personal_trainer", "photographer", "physician", "poet", "professor",
    "psychologist", "rapper", "software_engineer", "surgeon", "teacher",
    "yoga_teacher",
]


def _bios(ids: list[int]) -> dict:
    return {"kind": "table", "hf": "LabHC/bias_in_bios", "text": "hard_text",
            "label": "profession", "balance": "gender",
            "classes": {str(i): _PROFESSIONS[i] for i in ids}}


DATASETS: dict[str, dict] = {
    "LabHC/bias_in_bios_class_set1": _bios([0, 1, 2, 6, 9]),
    "LabHC/bias_in_bios_class_set2": _bios([11, 13, 14, 18, 19]),
    "LabHC/bias_in_bios_class_set3": _bios([20, 21, 22, 25, 26]),
    "canrager/amazon_reviews_mcauley_1and5": {
        "kind": "table", "hf": "canrager/amazon_reviews_mcauley_1and5",
        "text": "text", "label": "category", "balance": "rating",
        "classes": {"1": "Toys_and_Games", "2": "Cell_Phones_and_Accessories",
                    "3": "Industrial_and_Scientific", "5": "Musical_Instruments",
                    "6": "Electronics"}},
    "canrager/amazon_reviews_mcauley_1and5_sentiment": {
        "kind": "table", "hf": "canrager/amazon_reviews_mcauley_1and5",
        "text": "text", "label": "rating", "balance": None,
        "classes": {"1.0": "negative", "5.0": "positive"}},
    "fancyzhx/ag_news": {
        "kind": "table", "hf": "fancyzhx/ag_news", "text": "text",
        "label": "label", "balance": None,
        "classes": {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}},
    "Helsinki-NLP/europarl": {
        "kind": "europarl", "hf": "Helsinki-NLP/europarl",
        # language → the bilingual config it is read from (SAEBench's choice)
        "pairs": {"en": "en-fr", "fr": "fr-it", "de": "de-en", "es": "es-fr",
                  "nl": "nl-pt"},
        "classes": {"en": "English", "fr": "French", "de": "German",
                    "es": "Spanish", "nl": "Dutch"}},
}
DEFAULT_DATASETS = [
    "LabHC/bias_in_bios_class_set1", "fancyzhx/ag_news",
    "canrager/amazon_reviews_mcauley_1and5_sentiment", "Helsinki-NLP/europarl",
]
_SHORT = {
    "LabHC/bias_in_bios_class_set1": "bios1",
    "LabHC/bias_in_bios_class_set2": "bios2",
    "LabHC/bias_in_bios_class_set3": "bios3",
    "canrager/amazon_reviews_mcauley_1and5": "amzcat",
    "canrager/amazon_reviews_mcauley_1and5_sentiment": "amzsent",
    "fancyzhx/ag_news": "agnews",
    "Helsinki-NLP/europarl": "europarl",
}


def _key(v) -> str:
    """Class key as SAEBench writes it: '2' for ints, '5.0' for ratings."""
    return str(v)


def _stable_seed(*parts) -> int:
    """Seed that does not depend on PYTHONHASHSEED or on task ordering."""
    h = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(h[:8], "little")


# ─── prepare ─────────────────────────────────────────────────────────────────

def _sample_table(name: str, spec: dict, split: str, per_class: int,
                  min_chars: int, seed: int) -> list[dict]:
    """per_class examples of each chosen class, split evenly across the
    balance column's groups (gender / rating) when the dataset has one."""
    from datasets import load_dataset

    ds = load_dataset(spec["hf"], split=split)
    labels = [_key(v) for v in ds[spec["label"]]]
    groups = ds[spec["balance"]] if spec["balance"] else [None] * len(labels)
    cells: dict[tuple, list[int]] = defaultdict(list)
    for i, (lab, grp) in enumerate(zip(labels, groups)):
        if lab in spec["classes"]:
            cells[(lab, grp)].append(i)

    rows = []
    for lab, lab_name in spec["classes"].items():
        grp_keys = sorted(g for (l, g) in cells if l == lab)
        assert grp_keys, f"{name}/{split}: class {lab} ({lab_name}) has no rows"
        quota = per_class // len(grp_keys)
        assert quota > 0, f"--*-per-class too small for {len(grp_keys)} groups"
        for grp in grp_keys:
            idx = cells[(lab, grp)]
            random.Random(_stable_seed(seed, name, split, lab, grp)).shuffle(idx)
            got = 0
            for i in idx:
                text = ds[int(i)][spec["text"]]
                if len(text) < min_chars:
                    continue
                aux = {spec["balance"]: grp} if spec["balance"] else {}
                rows.append({"dataset": name, "split": split, "label": lab,
                             "label_name": lab_name, "text": text, "aux": aux})
                got += 1
                if got == quota:
                    break
            assert got == quota, (
                f"{name}/{split}: class {lab} ({lab_name}) group {grp} has only "
                f"{got} texts of >= {min_chars} chars; need {quota}")
    return rows


def _sample_europarl(name: str, spec: dict, n_train: int, n_test: int,
                     min_chars: int, seed: int) -> list[dict]:
    """Europarl has only a train split and one sentence per row. Consecutive
    sentences are joined until a chunk reaches min_chars, the first
    n_train + n_test chunks are taken (streaming, as SAEBench does), then
    shuffled and sliced into train / test."""
    from datasets import load_dataset

    rows = []
    for lang, lang_name in spec["classes"].items():
        stream = load_dataset(spec["hf"], spec["pairs"][lang], split="train",
                              streaming=True)
        chunks, buf = [], ""
        for row in stream:
            sent = (row["translation"][lang] or "").strip()
            if not sent:
                continue
            buf = f"{buf} {sent}" if buf else sent
            if len(buf) >= min_chars:
                chunks.append(buf)
                buf = ""
                if len(chunks) == n_train + n_test:
                    break
        assert len(chunks) == n_train + n_test, (
            f"{name}: only {len(chunks)} chunks for {lang}")
        random.Random(_stable_seed(seed, name, lang)).shuffle(chunks)
        for j, text in enumerate(chunks):
            rows.append({"dataset": name, "split": "train" if j < n_train else "test",
                         "label": lang, "label_name": lang_name, "text": text,
                         "aux": {}})
    return rows


def cmd_prepare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for name in args.datasets:
        spec = DATASETS[name]
        if spec["kind"] == "europarl":
            got = _sample_europarl(name, spec, args.train_per_class,
                                   args.test_per_class, args.min_chars, args.seed)
        else:
            got = []
            for split, n in (("train", args.train_per_class),
                             ("test", args.test_per_class)):
                got += _sample_table(name, spec, split, n, args.min_chars, args.seed)
        counts = Counter((r["split"], r["label_name"]) for r in got)
        print(f"[prepare] {name}: {len(got)} examples  "
              + "  ".join(f"{s}/{c}={n}" for (s, c), n in sorted(counts.items())))
        rows += got
    for i, r in enumerate(rows):
        r["id"] = f"{_SHORT[r['dataset']]}-{r['split']}-{i:06d}"
    path = out / "examples.jsonl"
    path.write_text("".join(
        json.dumps({"id": r["id"], **{k: v for k, v in r.items() if k != "id"}},
                   ensure_ascii=False) + "\n" for r in rows))
    print(f"[prepare] {len(rows)} examples → {path}")
    print(f"[prepare] next: run_multilayer.py extract --prompts {path} "
          f"--positions last --max-length 128 --out {out}")


# ─── probes ──────────────────────────────────────────────────────────────────

def fit_logreg(x: np.ndarray, y: np.ndarray, c: float = 1.0,
               max_iter: int = 100, tol: float = 1e-8):
    """L2 logistic regression, the objective sklearn's LogisticRegression(C=c)
    minimises (intercept unpenalised), solved to convergence with damped
    Newton steps. Returns (w [F], b) as float64."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, f = x.shape
    if f > n:
        # More features than examples: the optimum lies in the row space of
        # x, so solve the same problem in n dimensions and map back. Exact.
        _, _, vt = np.linalg.svd(x, full_matrices=False)
        wz, b0 = fit_logreg(x @ vt.T, y, c, max_iter, tol)
        return vt.T @ wz, b0

    xa = np.hstack([x, np.ones((n, 1))])
    reg = np.full(f + 1, 1.0 / c)
    reg[-1] = 0.0                                   # intercept is not penalised

    def loss(th: np.ndarray) -> float:
        z = xa @ th
        return float(np.sum(np.logaddexp(0.0, z) - y * z) + 0.5 * np.sum(reg * th * th))

    theta = np.zeros(f + 1)
    cur = loss(theta)
    for _ in range(max_iter):
        p = 1.0 / (1.0 + np.exp(-np.clip(xa @ theta, -500, 500)))
        grad = xa.T @ (p - y) + reg * theta
        if np.abs(grad).max() < tol:
            break
        hess = (xa * (p * (1.0 - p))[:, None]).T @ xa + np.diag(reg)
        hess[-1, -1] += 1e-10                       # separable data: keep it PD
        step = np.linalg.solve(hess, grad)
        t, slope = 1.0, float(grad @ step)
        while True:                                 # backtracking line search
            new = theta - t * step
            new_loss = loss(new)
            if new_loss <= cur - 1e-4 * t * slope or t < 1e-10:
                break
            t *= 0.5
        if cur - new_loss < 1e-12:
            theta = new
            break
        theta, cur = new, new_loss
    return theta[:-1], float(theta[-1])


def _accuracy(w: np.ndarray, b: float, x: np.ndarray, y: np.ndarray) -> float:
    return float((((x.astype(np.float64) @ w + b) > 0) == (y > 0.5)).mean())


def _standardizer(x_train: np.ndarray):
    mu = x_train.mean(0, keepdims=True)
    sd = x_train.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return lambda x: (x - mu) / sd


def top_k_by_mean_diff(x: np.ndarray, y: np.ndarray, k: int) -> np.ndarray:
    """SAEBench get_top_k_mean_diff_mask: indices of the k features with the
    largest |mean over positives - mean over negatives|. Ties → lowest index."""
    diff = np.abs(x[y == 1].mean(0) - x[y == 0].mean(0))
    return np.argsort(-diff, kind="stable")[:k]


def binary_task(labels: np.ndarray, cls: str, seed_parts: tuple) -> tuple[np.ndarray, np.ndarray]:
    """SAEBench prepare_probe_data: all examples of `cls` as positives, an
    equal number of negatives drawn evenly from the other classes.
    Returns (row indices into `labels`, 0/1 targets), shuffled."""
    rng = np.random.default_rng(_stable_seed(*seed_parts, cls))
    pos = np.flatnonzero(labels == cls)
    others = sorted(set(labels.tolist()) - {cls})
    per_other = math.ceil(len(pos) / len(others))
    neg = np.concatenate([
        rng.permutation(np.flatnonzero(labels == o))[:per_other] for o in others])
    neg = rng.permutation(neg)[: len(pos)]
    idx = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = rng.permutation(len(idx))
    return idx[order], y[order]


# Letters only (any script), with an inner apostrophe: "doctor's" is one word.
_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*")


def tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def word_matrix(docs: list[set[str]], vocab: dict[str, int]) -> np.ndarray:
    m = np.zeros((len(docs), len(vocab)), dtype=np.float32)
    for i, words in enumerate(docs):
        cols = [vocab[w] for w in words if w in vocab]
        m[i, cols] = 1.0
    return m


# ─── probe ───────────────────────────────────────────────────────────────────

def _load_run(run: Path) -> tuple[dict, dict[tuple[int, int], str], np.ndarray]:
    summary = json.loads((run / "summary.json").read_text())
    verbs: dict[tuple[int, int], str] = {}
    for line in (run / "verbalizations.jsonl").read_text().splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        verbs[(r["vec_idx"], r["layer"])] = r["verbalization"]
    recon = np.load(run / "reconstructions.npy", mmap_mode="r")
    assert recon.shape[:2] == (len(summary["rows"]), len(summary["layers"])), (
        f"reconstructions.npy {recon.shape} does not match summary.json "
        f"({len(summary['rows'])} rows x {len(summary['layers'])} layers)")
    return summary, verbs, recon


def cmd_probe(args: argparse.Namespace) -> None:
    run = Path(args.run)
    out = Path(args.out or run)
    out.mkdir(parents=True, exist_ok=True)

    examples = [json.loads(l) for l in Path(args.examples).read_text().splitlines()
                if l.strip()]
    acts = np.load(args.acts, mmap_mode="r")
    assert acts.ndim == 3 and acts.shape[0] == len(examples), (
        f"activations {acts.shape} vs {len(examples)} examples — extract must be "
        f"run with --positions last on the same examples.jsonl")
    samples = Path(args.acts).with_name("samples.jsonl")
    if samples.exists():
        ids = [json.loads(l)["sample_id"] for l in samples.read_text().splitlines()]
        assert ids == [e["id"] for e in examples], (
            f"{samples} is not in the same order as {args.examples}")

    summary, verbs, recon = _load_run(run)
    row_pos = {v: i for i, v in enumerate(summary["rows"])}
    layer_pos = {l: j for j, l in enumerate(summary["layers"])}
    layers = args.layers or summary["layers"]
    assert all(l in layer_pos for l in layers), (
        f"layers {layers} not all in this run ({summary['layers']})")
    # Only examples the run covers (roundtrip --n may have taken a prefix).
    covered = [i for i in range(len(examples)) if i in row_pos]
    print(f"[probe] {len(covered)}/{len(examples)} examples, layers {layers}, "
          f"adapter={summary.get('adapter')}")

    by_dataset: dict[str, list[int]] = defaultdict(list)
    for i in covered:
        by_dataset[examples[i]["dataset"]].append(i)

    per_task: list[dict] = []
    for dataset, ex_idx in by_dataset.items():
        ex_idx_arr = np.array(ex_idx)
        split = np.array([examples[i]["split"] for i in ex_idx])
        label = np.array([examples[i]["label"] for i in ex_idx])
        names = {examples[i]["label"]: examples[i]["label_name"] for i in ex_idx}
        tr, te = np.flatnonzero(split == "train"), np.flatnonzero(split == "test")
        classes = sorted(set(label[tr].tolist()) & set(label[te].tolist()))
        if len(classes) < 2:
            print(f"[probe] skipping {dataset}: fewer than 2 classes covered")
            continue
        # Same binary tasks at every layer, so layers are comparable.
        tasks = {c: (binary_task(label[tr], c, (args.seed, dataset, "train")),
                     binary_task(label[te], c, (args.seed, dataset, "test")))
                 for c in classes}

        for layer in layers:
            j = layer_pos[layer]
            h = np.asarray(acts[ex_idx_arr, layer], dtype=np.float32)
            r = np.asarray(recon[[row_pos[i] for i in ex_idx], j], dtype=np.float32)
            missing = [i for i in ex_idx if (i, layer) not in verbs]
            assert not missing, (
                f"{len(missing)} verbalizations missing at layer {layer} "
                f"(e.g. vec_idx {missing[0]}) — rerun roundtrip to finish")
            docs = [tokenize(verbs[(i, layer)]) for i in ex_idx]

            for c in classes:
                (tr_i, ytr), (te_i, yte) = tasks[c]
                tr_rows, te_rows = tr[tr_i], te[te_i]
                m: dict[str, float] = {}

                # Raw activation: full-dimension probe and SAEBench's top-k dims.
                z = _standardizer(h[tr_rows])
                w, b = fit_logreg(z(h[tr_rows]), ytr)
                m["llm_test_accuracy"] = _accuracy(w, b, z(h[te_rows]), yte)
                # The model's own probe read off the reconstruction.
                m["recon_transfer_test_accuracy"] = _accuracy(w, b, z(r[te_rows]), yte)
                for k in args.k:
                    sel = top_k_by_mean_diff(h[tr_rows], ytr, k)
                    zk = _standardizer(h[tr_rows][:, sel])
                    wk, bk = fit_logreg(zk(h[tr_rows][:, sel]), ytr)
                    m[f"llm_top_{k}_test_accuracy"] = _accuracy(
                        wk, bk, zk(h[te_rows][:, sel]), yte)

                # Reconstruction: how much of the concept survives the text.
                zr = _standardizer(r[tr_rows])
                wr, br = fit_logreg(zr(r[tr_rows]), ytr)
                m["recon_test_accuracy"] = _accuracy(wr, br, zr(r[te_rows]), yte)

                # Verbalization words. Vocabulary from this task's train docs.
                df = Counter(w_ for i in tr_rows for w_ in docs[i])
                vocab = {w_: n for n, w_ in enumerate(
                    sorted(w_ for w_, cnt in df.items() if cnt >= args.min_df))}
                top_words: list[list] = []
                if vocab:
                    inv = sorted(vocab, key=vocab.get)
                    xtr = word_matrix([docs[i] for i in tr_rows], vocab)
                    xte = word_matrix([docs[i] for i in te_rows], vocab)
                    ww, bw = fit_logreg(xtr, ytr)
                    m["nla_test_accuracy"] = _accuracy(ww, bw, xte, yte)
                    diff = xtr[ytr == 1].mean(0) - xtr[ytr == 0].mean(0)
                    for k in args.k:
                        sel = top_k_by_mean_diff(xtr, ytr, k)
                        wk, bk = fit_logreg(xtr[:, sel], ytr)
                        m[f"nla_top_{k}_test_accuracy"] = _accuracy(
                            wk, bk, xte[:, sel], yte)
                    top_words = [[inv[s], round(float(diff[s]), 3)]
                                 for s in top_k_by_mean_diff(xtr, ytr, max(args.k))]
                else:   # every verbalization empty: chance
                    m["nla_test_accuracy"] = 0.5
                    for k in args.k:
                        m[f"nla_top_{k}_test_accuracy"] = 0.5

                per_task.append({
                    "dataset": dataset, "class": c, "class_name": names[c],
                    "layer": layer, "n_train": len(ytr), "n_test": len(yte),
                    "vocab_size": len(vocab), "metrics": m,
                    # [word, P(word | class) - P(word | not class)] on train
                    "top_words": top_words,
                })
            print(f"  {_SHORT.get(dataset, dataset):9s} layer {layer:2d}  "
                  + "  ".join(f"{k_}={np.mean([t['metrics'][k_] for t in per_task[-len(classes):]]):.3f}"
                              for k_ in ("llm_test_accuracy", "nla_test_accuracy",
                                         f"nla_top_{args.k[0]}_test_accuracy")))

    assert per_task, "no tasks — check that the run covers train and test examples"
    metric_names = list(per_task[0]["metrics"])

    # Mean over classes within a dataset, then over datasets (SAEBench).
    def _mean(rows: list[dict]) -> dict[str, float]:
        return {k: float(np.mean([t["metrics"][k] for t in rows])) for k in metric_names}

    per_dataset_layer, per_layer = [], []
    for layer in layers:
        ds_means = []
        for dataset in by_dataset:
            rows = [t for t in per_task if t["layer"] == layer and t["dataset"] == dataset]
            if rows:
                ds_means.append(_mean(rows))
                per_dataset_layer.append({"dataset": dataset, "layer": layer,
                                          "n_classes": len(rows), **ds_means[-1]})
        s = summary["per_layer"].get(str(layer), {})
        per_layer.append({
            "layer": layer,
            **{k: float(np.mean([d[k] for d in ds_means])) for k in metric_names},
            "cos_roundtrip": s.get("cos_roundtrip"), "cjk_rate": s.get("cjk_rate"),
        })

    (out / "sparse_probing.json").write_text(json.dumps({
        "config": {"examples": str(args.examples), "acts": str(args.acts),
                   "run": str(run), "adapter": summary.get("adapter"),
                   "k": args.k, "min_df": args.min_df, "seed": args.seed,
                   "n_examples": len(covered), "datasets": list(by_dataset)},
        "per_layer": per_layer, "per_dataset_layer": per_dataset_layer,
        "per_task": per_task,
    }, indent=1, ensure_ascii=False))
    with (out / "sparse_probing_by_layer.csv").open("w", newline="") as fh:
        wr_ = csv.DictWriter(fh, fieldnames=list(per_layer[0]))
        wr_.writeheader()
        wr_.writerows(per_layer)

    show = (["llm_test_accuracy"] + [f"llm_top_{k}_test_accuracy" for k in args.k]
            + [f"nla_top_{k}_test_accuracy" for k in args.k]
            + ["nla_test_accuracy", "recon_test_accuracy",
               "recon_transfer_test_accuracy"])
    head = {"llm_test_accuracy": "llm", "nla_test_accuracy": "nla_all",
            "recon_test_accuracy": "recon", "recon_transfer_test_accuracy": "recon_xfer"}
    cols = [head.get(k, k.replace("_test_accuracy", "")) for k in show]
    print(f"\n[probe] mean test accuracy over {len(by_dataset)} datasets "
          f"(chance = 0.5)")
    print("  layer  " + "  ".join(f"{c:>10s}" for c in cols) + "    cos_rt")
    for row in per_layer:
        cos = row["cos_roundtrip"]
        print(f"  {row['layer']:5d}  " + "  ".join(f"{row[k]:10.3f}" for k in show)
              + (f"  {cos:8.3f}" if cos is not None else ""))
    print(f"[probe] → {out}/sparse_probing.json, sparse_probing_by_layer.csv")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare", help="sample class-balanced examples (needs network)")
    p.add_argument("--out", required=True, help="directory for examples.jsonl")
    p.add_argument("--datasets", nargs="+", choices=sorted(DATASETS),
                   default=DEFAULT_DATASETS)
    p.add_argument("--train-per-class", type=int, default=100)
    p.add_argument("--test-per-class", type=int, default=50)
    p.add_argument("--min-chars", type=int, default=300,
                   help="skip shorter texts so the last token is past position ~50")
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_prepare)

    q = sub.add_parser("probe", help="fit the probes (CPU)")
    q.add_argument("--examples", required=True, help="examples.jsonl from prepare")
    q.add_argument("--acts", required=True,
                   help="activations.npy from run_multilayer.py extract")
    q.add_argument("--run", required=True,
                   help="run_multilayer.py roundtrip output directory")
    q.add_argument("--out", default=None, help="default: --run")
    q.add_argument("--layers", type=int, nargs="+", default=None,
                   help="default: every layer in the run")
    q.add_argument("--k", type=int, nargs="+", default=[1, 2, 5])
    q.add_argument("--min-df", type=int, default=2,
                   help="a word must appear in this many training verbalizations")
    q.add_argument("--seed", type=int, default=42)
    q.set_defaults(func=cmd_probe)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
