"""Verbalization screening — does the AV verbalize the fact?

The whole NLA-controllability idea hinges on one empirical question: when an
activation encodes a fact (model is about to say "Paris"), does the AV's
verbalization actually contain that fact in editable form? This script measures
the hit-rate. It's the cheap go/no-go test before any editing machinery.

Two phases (decoupled so the HF base model and the SGLang AV server don't fight
over one GPU, and so you can re-verbalize without re-extracting):

  1. extract    base model -> activations + "does the model know it" check
                writes  activations.npy  +  records.jsonl
  2. verbalize  activations -> AV (via SGLang) -> answer/subject detection
                writes  results.jsonl  + prints the report

Run, in order:

  # phase 1 (needs the BASE model + a GPU; ~15GB for Qwen-7B)
  python run_screening.py extract \
      --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 --out runs/qwen_l20

  # start the AV server in a separate process (own GPU):
  python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \
      --port 30000 --disable-radix-cache --trust-remote-code

  # phase 2 (talks to SGLang; the AV checkpoint dir supplies the sidecar)
  python run_screening.py verbalize \
      --out runs/qwen_l20 --av-checkpoint /path/to/nla-qwen2.5-7b-L20-av \
      --sglang-url http://localhost:30000

Interpretation: a high answer-hit-rate (over model-knows facts) means the fact
is verbalized and editable -> proceed to the swap experiment. A high
subject-hit but low answer-hit means the AV describes the *input* (France) not
the *prediction* (Paris) -> the edit handle is the subject, not the answer.
Low both -> the method has a low ceiling, and you learned it cheaply.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# nla_inference.py lives at the repo root (two dirs up from this file).
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from facts import builtin_capitals, filter_single_token, load_counterfact  # noqa: E402


# ─── Phase 1: extract ────────────────────────────────────────────────────────

def cmd_extract(args: argparse.Namespace) -> None:
    import torch
    from extract import BaseModelExtractor

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    extractor = BaseModelExtractor(
        args.base_model, device=args.device,
        dtype=torch.bfloat16 if args.dtype == "bf16" else torch.float32,
    )

    if args.counterfact:
        facts = load_counterfact(args.counterfact, template_relation="capital of",
                                 limit=args.limit)
    else:
        facts = builtin_capitals()

    kept, dropped = filter_single_token(facts, extractor.tokenizer)
    if dropped:
        print(f"[extract] dropped {len(dropped)} multi-token-answer facts: "
              f"{', '.join(f.answer for f in dropped)}")
    if args.limit:
        kept = kept[: args.limit]
    print(f"[extract] {len(kept)} single-token facts, layer {args.layer} "
          f"(hidden_states[{args.layer + args.hs_offset}])")

    vectors: list[np.ndarray] = []
    records: list[dict] = []
    n_known = 0
    for f in kept:
        for ex in extractor.extract(f, layer=args.layer, hs_offset=args.hs_offset):
            records.append({
                "vec_index": len(vectors),
                "subject": ex.fact.subject,
                "answer": ex.fact.answer,
                "prompt": ex.fact.prompt,
                "relation": ex.fact.relation,
                "position": ex.position,
                "model_top1": ex.model_top1,
                "model_knows": ex.model_knows,
            })
            vectors.append(ex.vector.astype(np.float32))
        # count once per fact (model_knows is per-fact, same for both positions)
        n_known += int(records[-1]["model_knows"])

    np.save(out / "activations.npy", np.stack(vectors))
    (out / "records.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in records)
    )
    print(f"[extract] base model knew {n_known}/{len(kept)} facts "
          f"(top-1 == answer)")
    print(f"[extract] wrote {len(vectors)} vectors -> {out}/activations.npy")
    print(f"[extract] next: start SGLang, then run `verbalize --out {out}`")


# ─── Phase 2: verbalize ──────────────────────────────────────────────────────

def _detect(text: str, answer: str, subject: str) -> dict:
    """Substring detection of answer / subject in the verbalization (a screening
    signal, not a precise judge). Lowercased; whole-text search."""
    t = text.lower()
    return {
        "answer_hit": answer.lower() in t,
        "subject_hit": subject.lower() in t,
    }


def cmd_verbalize(args: argparse.Namespace) -> None:
    from nla_inference import NLAClient

    out = Path(args.out)
    vectors = np.load(out / "activations.npy")
    records = [json.loads(line) for line in
               (out / "records.jsonl").read_text().splitlines()]

    client = NLAClient(
        args.av_checkpoint,
        sglang_url=args.sglang_url,
        injection_scale_override=args.injection_scale,
    )

    results: list[dict] = []
    for r in records:
        if args.positions and r["position"] not in args.positions:
            continue
        v = vectors[r["vec_index"]]
        text = client.generate(
            v, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
        )
        det = _detect(text, r["answer"], r["subject"])
        results.append({**r, **det, "verbalization": text})
        flag = "✓" if det["answer_hit"] else ("~" if det["subject_hit"] else "·")
        print(f"  {flag} [{r['position']:>7}] {r['subject']:>12} -> "
              f"{r['answer']:<11} knows={r['model_knows']!s:<5} "
              f"ans={det['answer_hit']!s:<5} subj={det['subject_hit']!s}")

    (out / "results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results)
    )
    _report(results)
    print(f"\n[verbalize] full transcripts -> {out}/results.jsonl")


def _report(results: list[dict]) -> None:
    """Aggregate hit-rates per probe position, conditioned on model_knows."""
    positions = sorted({r["position"] for r in results})
    print("\n" + "═" * 72)
    print("VERBALIZATION SCREENING — does the AV verbalize the fact?")
    print("═" * 72)
    for pos in positions:
        rows = [r for r in results if r["position"] == pos]
        known = [r for r in rows if r["model_knows"]]
        denom = known or rows  # fall back to all if model knew none (shouldn't)
        n = len(denom)
        ans = sum(r["answer_hit"] for r in denom)
        subj = sum(r["subject_hit"] for r in denom)
        combo = Counter(
            ("answer+subject" if r["answer_hit"] and r["subject_hit"]
             else "answer-only" if r["answer_hit"]
             else "subject-only" if r["subject_hit"]
             else "neither")
            for r in denom
        )
        print(f"\n position = {pos!r}   "
              f"(over {n} facts the base model knows; {len(rows)} total)")
        print(f"   answer  verbalized: {ans}/{n}  ({ans / n:.0%})")
        print(f"   subject verbalized: {subj}/{n}  ({subj / n:.0%})")
        for k in ("answer+subject", "answer-only", "subject-only", "neither"):
            if combo[k]:
                print(f"     {k:<14}: {combo[k]}")
    print("\n" + "═" * 72)
    print("read: high answer-rate -> fact is editable, go to the swap test.")
    print("      subject-only     -> AV describes the input, not the prediction.")
    print("      neither          -> low ceiling; stop here, you saved the effort.")
    print("═" * 72)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract", help="phase 1: base model -> activations")
    e.add_argument("--base-model", required=True,
                   help="HF base model the AV was trained on, e.g. "
                        "Qwen/Qwen2.5-7B-Instruct")
    e.add_argument("--layer", type=int, required=True,
                   help="NLA extraction layer K (e.g. 20 for Qwen-L20)")
    e.add_argument("--hs-offset", type=int, default=1,
                   help="hidden_states index = layer + offset. Default 1 "
                        "(output of block K). Flip to 0 if outputs are garbage.")
    e.add_argument("--out", required=True, help="run directory")
    e.add_argument("--device", default="cuda")
    e.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    e.add_argument("--counterfact", default=None,
                   help="path to CounterFact json (default: built-in capitals)")
    e.add_argument("--limit", type=int, default=None)
    e.set_defaults(func=cmd_extract)

    v = sub.add_parser("verbalize", help="phase 2: activations -> AV -> detect")
    v.add_argument("--out", required=True, help="run directory from phase 1")
    v.add_argument("--av-checkpoint", required=True,
                   help="HF-format AV (actor) dir with nla_meta.yaml")
    v.add_argument("--sglang-url", default="http://localhost:30000")
    v.add_argument("--temperature", type=float, default=0.0,
                   help="0 = greedy (reproducible). Default 0 for screening.")
    v.add_argument("--max-new-tokens", type=int, default=200)
    v.add_argument("--injection-scale", type=float, default=None,
                   help="override sidecar injection_scale (OOD — rarely needed)")
    v.add_argument("--positions", nargs="*", default=None,
                   choices=["last", "subject"],
                   help="restrict to these probe positions (default: both)")
    v.set_defaults(func=cmd_verbalize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
