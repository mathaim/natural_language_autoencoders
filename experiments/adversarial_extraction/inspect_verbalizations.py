"""Sanity-check a finished (or in-progress) verbalization run.

Answers, per dataset: how many positions, what fraction the AV refused, whether
injection silently failed (CJK smoke test from the repo docs), and what the
verbalizations actually look like — with the token in context so you can judge
whether the description matches the position.

  python inspect_verbalizations.py --run-dir /scratch/euh7ys/adversarial_runs/qwen7b-l20
  python inspect_verbalizations.py --run-dir ... --dataset wmdp --n-examples 5 --min-position 20
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

_CJK = range(0x4E00, 0x9FFF + 1)


def _has_cjk(s: str | None) -> bool:
    return bool(s) and any(ord(c) in _CJK for c in s)


# Broader than the is_refusal patterns: catches "soft" refusals where the AV
# describes the content as sensitive/dangerous without an explicit "I can't".
# Counts only — a second lens on whether the 0% refusal rate is real.
_SOFT = ("sensitive", "harmful", "dangerous", "unethical", "inappropriate",
         "illegal", "will not", "cannot provide", "not able to provide", "decline")


def _is_soft(s: str | None) -> bool:
    return bool(s) and any(k in s.lower() for k in _SOFT)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dataset", default=None, help="restrict examples to one dataset")
    ap.add_argument("--n-examples", type=int, default=3, help="examples per dataset")
    ap.add_argument("--min-position", type=int, default=0,
                    help="only sample examples from positions >= this")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--context", type=int, default=12,
                    help="tokens of context to show on each side of the position")
    args = ap.parse_args()

    run = Path(args.run_dir)
    samples = {s["sample_id"]: s for s in
               (json.loads(l) for l in (run / "samples.jsonl").read_text().splitlines())}
    recs = [json.loads(l) for l in (run / "verbalizations.jsonl").read_text().splitlines()
            if l.strip()]

    # ── Per-dataset stats ───────────────────────────────────────────────────
    by_ds: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_ds[r["dataset"]].append(r)

    expected = sum(s["n_tokens"] for s in samples.values())
    print(f"run: {run}")
    print(f"positions: {len(recs)} / {expected} expected "
          f"({len(recs) / expected:.0%})  samples: {len(samples)}\n")

    hdr = (f"  {'dataset':<12} {'positions':>9} {'verbalized':>10} {'refused':>8} "
           f"{'refusal%':>9} {'any-ref%':>8} {'CJK':>5} {'early-ref%':>10} {'late-ref%':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for ds in sorted(by_ds):
        rows = by_ds[ds]
        verb = [r for r in rows if r["verbalization"] is not None]
        ref = [r for r in verb if r["is_refusal"]]
        # any-ref = hard refusal OR soft wording, so it can only be >= refusal%
        soft = sum(r["is_refusal"] or _is_soft(r["verbalization"]) for r in verb)
        cjk = sum(_has_cjk(r["verbalization"]) for r in verb)
        early = [r for r in verb if r["min_position_warning"]]
        late = [r for r in verb if not r["min_position_warning"]]
        e_ref = sum(r["is_refusal"] for r in early) / len(early) if early else float("nan")
        l_ref = sum(r["is_refusal"] for r in late) / len(late) if late else float("nan")
        n = len(verb) or 1
        print(f"  {ds:<12} {len(rows):>9} {len(verb):>10} {len(ref):>8} "
              f"{len(ref) / n:>8.1%} {soft / n:>7.1%} {cjk:>5} "
              f"{e_ref:>9.1%} {l_ref:>8.1%}")

    # CJK smoke test. A broken injection path (AV sees the literal marker char)
    # makes nearly EVERY output CJK — so the signal is the rate, not the count.
    print()
    all_verb = [r for r in recs if r["verbalization"] is not None]
    cjk_recs = [r for r in all_verb if _has_cjk(r["verbalization"])]
    if not cjk_recs:
        print("  ✓ no CJK in any verbalization — injection path looks healthy")
    else:
        rate = len(cjk_recs) / len(all_verb)
        n_early = sum(r["min_position_warning"] for r in cjk_recs)
        n_samp = len({r["sample_id"] for r in cjk_recs})
        print(f"  CJK: {len(cjk_recs)}/{len(all_verb)} verbalizations ({rate:.2%}), "
              f"{n_early} at early positions, spread over {n_samp} samples")
        if rate > 0.05:
            print("  ✗ LIKELY BROKEN INJECTION — the AV is describing the marker char. "
                  "Check hs_offset and the sidecar neighbor IDs.")
        else:
            print("  ✓ within noise — a broken path would make nearly all outputs CJK, "
                  "not a fraction of a percent (see generate() docstring on noisy inputs)")

    # ── Examples with the token in context ──────────────────────────────────
    # Reconstruct each sample's token strings from its own records (no
    # tokenizer needed), then show the chosen position bracketed.
    tok_by_sample: dict[str, dict[int, str]] = defaultdict(dict)
    for r in recs:
        tok_by_sample[r["sample_id"]][r["position_idx"]] = r["token_str"]

    rng = random.Random(args.seed)
    for ds in sorted(by_ds):
        if args.dataset and ds != args.dataset:
            continue
        pool = [r for r in by_ds[ds]
                if r["verbalization"] is not None and r["position_idx"] >= args.min_position]
        if not pool:
            continue
        print(f"\n{'═' * 78}\n{ds}  ({len(pool)} eligible positions)\n{'═' * 78}")
        for r in rng.sample(pool, min(args.n_examples, len(pool))):
            toks = tok_by_sample[r["sample_id"]]
            p = r["position_idx"]
            n_tok = samples.get(r["sample_id"], {}).get("n_tokens", p + args.context + 1)
            lo, hi = max(0, p - args.context), min(p + args.context + 1, n_tok)
            ctx = "".join(
                f"[[{toks.get(i, '?')}]]" if i == p else toks.get(i, "?")
                for i in range(lo, hi)
            ).replace("\n", "⏎")
            flags = " ".join(f for f, on in (("REFUSAL", r["is_refusal"]),
                                             ("early", r["min_position_warning"]),
                                             ("CJK", _has_cjk(r["verbalization"]))) if on)
            print(f"\n  {r['sample_id']}  pos {p}  {flags}")
            print(f"  …{ctx}…")
            print(f"  → {r['verbalization'][:400]}")
            s = samples.get(r["sample_id"], {})
            if s.get("model_output_is_refusal"):
                print(f"  (base model refused this prompt)")


if __name__ == "__main__":
    main()
