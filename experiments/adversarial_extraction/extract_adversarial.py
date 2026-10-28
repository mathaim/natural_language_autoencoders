"""Extract activations and verbalizations from adversarial datasets.

Runs the base model over prompts from WMDP / HarmBench / AdvBench /
TruthfulQA and saves per-token-position activations, verbalizations (AV),
and model outputs (greedy decode).

Two-phase design (decouple the base-model GPU from the SGLang AV server):

  Phase 1 — extract
    base model forward pass → activations [T, d_model] at every position,
    token IDs, model greedy output. Writes activations.npy + samples.jsonl.

  Phase 2 — verbalize
    NLAClient (via SGLang) → verbalization text per (sample, position).
    Reads from Phase 1 output; writes verbalizations.jsonl.

Usage
─────
  # Phase 1: extract (needs base model on GPU)
  python extract_adversarial.py extract \\
      --datasets wmdp harmbench advbench truthfulqa \\
      --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 \\
      --out runs/qwen_l20_adversarial \\
      --n-samples 200          # per dataset; omit for full

  # Start AV server (separate GPU):
  python -m sglang.launch_server \\
      --model-path kitft/nla-qwen2.5-7b-L20-av \\
      --port 30000 --disable-radix-cache --trust-remote-code

  # Phase 2: verbalize
  python extract_adversarial.py verbalize \\
      --out runs/qwen_l20_adversarial \\
      --av-checkpoint kitft/nla-qwen2.5-7b-L20-av \\
      --sglang-url http://localhost:30000

Output files
────────────
  activations.npy        float32 [N_total_tokens, d_model]
                         all token positions across all samples, stacked
  samples.jsonl          one record per sample:
                           sample_id, dataset, text, token_ids, n_tokens,
                           vec_start (index into activations.npy),
                           model_output (greedy decode), meta (dataset fields)
  verbalizations.jsonl   one record per (sample × token position):
                           sample_id, dataset, position_idx, token_id,
                           token_str, verbalization

Notes
─────
- Activations are raw (not L2-normalized) — normalization happens at
  injection time, consistent with the datagen pipeline invariant.
- model_output is a greedy decode of --decode-tokens tokens. It captures
  what the base model would say, independent of NLA.
- All positions are verbalized by default. Positions with index < 50 are
  flagged min_position_warning=True (the datagen left-context floor) so you
  can filter downstream; pass --skip-early-positions to null them instead.
- The injection marker smoke test: if verbalizations contain CJK characters
  (㊗, ㈎, etc.), injection failed. Run with --debug to print raw outputs.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

# Datagen uses a 50-token floor because fineweb MID-document samples with
# little left-context decode to noise. That doesn't apply to short adversarial
# prompts where position 8 of 10 has all the context there is — so we
# verbalize every position by default and just FLAG early ones.
_MIN_POSITION = 50

# Safety-refusal detection — the AV inherits RLHF safety training from its
# instruct-tuned base model and may refuse or sanitize verbalizations of
# harmful activations. We flag these explicitly rather than treating them as
# legitimate verbalizations. The refusal rate per dataset / position is itself
# a useful signal (the AV "knows" it's looking at harmful content).
_REFUSAL_PATTERNS = (
    "i can't", "i cannot", "i'm not able", "i won't", "i will not",
    "i'm unable to", "as an ai", "i don't feel comfortable",
    "i apologize", "i am not able", "i am unable",
    "i'm afraid i can't", "i must decline", "i need to decline",
    "cannot assist", "can't assist", "not something i can",
    "against my", "goes against",
)


def _is_refusal(text: str) -> bool:
    """Heuristic refusal detector for AV outputs.

    Not perfect — a genuine verbalization could mention "harmful" in passing —
    but good enough to flag the bulk of safety-triggered non-verbalizations.
    """
    t = text.lower()
    return any(p in t for p in _REFUSAL_PATTERNS)


# ─── Phase 1: extract ────────────────────────────────────────────────────────

def cmd_extract(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from adv_datasets import load_dataset_by_name, DATASET_NAMES

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading {args.base_model} …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()
    # Gemma-3 nests the text-model fields under config.text_config
    # (mirrors nla.arch_adapters.resolve_text_config without the import).
    text_cfg = getattr(model.config, "text_config", model.config)
    print(f"[extract] {text_cfg.num_hidden_layers} layers, "
          f"d_model={text_cfg.hidden_size}")

    datasets_to_run = args.datasets or list(DATASET_NAMES)

    all_vectors: list[np.ndarray] = []   # each: [d_model] float32
    samples_meta: list[dict] = []

    for dataset_name in datasets_to_run:
        print(f"\n[extract] loading {dataset_name} …")
        samples = load_dataset_by_name(dataset_name, limit=args.n_samples)
        print(f"[extract] {len(samples)} samples")

        for s_idx, sample in enumerate(samples):
            ids = tokenizer(
                sample.text,
                return_tensors="pt",
                add_special_tokens=True,
            )["input_ids"].to(args.device)

            T = ids.shape[1]

            with torch.inference_mode():
                fwd = model(ids, output_hidden_states=True, use_cache=False)
                hs = fwd.hidden_states[args.layer + args.hs_offset][0]  # [T, d_model]

                # greedy decode
                decode_ids = model.generate(
                    ids,
                    max_new_tokens=args.decode_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                )
            model_output = tokenizer.decode(
                decode_ids[0, T:], skip_special_tokens=True,
            )

            vec_start = len(all_vectors)
            for t in range(T):
                all_vectors.append(hs[t].float().cpu().numpy())

            samples_meta.append({
                "sample_id": sample.sample_id,
                "dataset": sample.dataset,
                "text": sample.text,
                "token_ids": ids[0].tolist(),
                "n_tokens": T,
                "vec_start": vec_start,
                "model_output": model_output,
                # Instruct-tuned base models refuse harmful prompts. Flag so
                # downstream code can distinguish refusal from genuine output.
                "model_output_is_refusal": _is_refusal(model_output),
                "meta": sample.meta,
            })

            if (s_idx + 1) % 25 == 0 or s_idx == len(samples) - 1:
                print(f"  {dataset_name}: {s_idx + 1}/{len(samples)} "
                      f"({len(all_vectors)} total position vectors)")

    vec_matrix = np.stack(all_vectors).astype(np.float32)
    np.save(out / "activations.npy", vec_matrix)
    (out / "samples.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples_meta)
    )

    n_by_dataset = {}
    for s in samples_meta:
        n_by_dataset[s["dataset"]] = n_by_dataset.get(s["dataset"], 0) + 1

    print(f"\n[extract] done")
    for ds, n in n_by_dataset.items():
        print(f"  {ds}: {n} samples")
    print(f"  total positions: {vec_matrix.shape[0]}  shape: {vec_matrix.shape}")
    print(f"[extract] activations → {out}/activations.npy")
    print(f"[extract] metadata   → {out}/samples.jsonl")
    print(f"[extract] next: start SGLang, then run `verbalize --out {out}`")


# ─── Phase 2: verbalize ──────────────────────────────────────────────────────

def cmd_verbalize(args: argparse.Namespace) -> None:
    import time
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from nla_inference import NLAClient

    out = Path(args.out)
    out_path = out / "verbalizations.jsonl"
    # mmap: 400MB+ for 7B and several GB for 70B — no need to hold it in RAM.
    vec_matrix = np.load(out / "activations.npy", mmap_mode="r")
    samples = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
    if args.datasets:
        samples = [s for s in samples if s["dataset"] in args.datasets]

    # ── Resume. Anything already in the output file is done. A SLURM timeout
    # can leave a truncated last line; drop it and rewrite the file clean.
    done: set[tuple[str, int]] = set()
    if out_path.exists():
        kept: list[str] = []
        for line in out_path.read_text().splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((r["sample_id"], r["position_idx"]))
            kept.append(line)
        out_path.write_text("".join(l + "\n" for l in kept))
        print(f"[verbalize] resuming — {len(done)} positions already in {out_path.name}")

    client = NLAClient(
        args.av_checkpoint,
        sglang_url=args.sglang_url,
        injection_scale_override=args.injection_scale,
    )

    # ── Work list. Token strings are decoded here on the main thread so the
    # tokenizer never runs inside the worker pool.
    work: list[tuple[dict, int, str]] = []
    for s in samples:
        tok_strs = [client.tokenizer.decode([t]) for t in s["token_ids"]]
        for t_idx in range(s["n_tokens"]):
            if (s["sample_id"], t_idx) not in done:
                work.append((s, t_idx, tok_strs[t_idx]))
    total = len(work)
    print(f"[verbalize] {total} positions to do across {len(samples)} samples, "
          f"{args.workers} concurrent requests")

    def one(s: dict, t_idx: int, tok_str: str) -> dict:
        early = t_idx < _MIN_POSITION
        if early and args.skip_early_positions:
            text = None
        else:
            vec = np.array(vec_matrix[s["vec_start"] + t_idx], dtype=np.float32)
            text = client.generate(
                vec, temperature=args.temperature, max_new_tokens=args.max_new_tokens,
            )
            if args.debug:
                _check_cjk(text, s["sample_id"], t_idx)
        return {
            "sample_id": s["sample_id"],
            "dataset": s["dataset"],
            "position_idx": t_idx,
            "token_id": s["token_ids"][t_idx],
            "token_str": tok_str,
            "min_position_warning": early,
            "verbalization": text,
            # True when the AV's safety training fired instead of describing
            # the activation. Exclude from NLA analyses; the rate itself is
            # informative (high on WMDP-chem = the AV recognises the content).
            "is_refusal": text is not None and _is_refusal(text),
        }

    # ── Run. Records are appended as they complete, so completion order is
    # nondeterministic — key on (sample_id, position_idx) downstream. Failed
    # requests are NOT written, so a resume retries them.
    max_errors = 25
    n_ok = n_err = 0
    t0 = time.time()
    with out_path.open("a") as fh, ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(one, s, t, ts) for s, t, ts in work]
        for fut in as_completed(futures):
            try:
                rec = fut.result()
            except Exception as e:  # noqa: BLE001 — log, skip, keep going
                n_err += 1
                if n_err <= 5:
                    print(f"[verbalize] ✗ request failed: {type(e).__name__}: {e}")
                if n_err >= max_errors:
                    print(f"[verbalize] {n_err} failures — server is probably down; "
                          f"stopping. Resubmit to resume from here.")
                    ex.shutdown(wait=False, cancel_futures=True)
                    break
                continue
            fh.write(json.dumps(rec) + "\n")
            n_ok += 1
            if n_ok % 100 == 0:
                fh.flush()   # bound the loss to <100 records on a hard kill
            if n_ok % 500 == 0 or n_ok + n_err == total:
                elapsed = time.time() - t0
                rate = n_ok / elapsed if elapsed else 0.0
                eta_min = (total - n_ok) / rate / 60 if rate else float("inf")
                failed = f", {n_err} failed" if n_err else ""
                print(f"  {n_ok}/{total} done  ({rate:.1f}/s, ETA {eta_min:.0f} min{failed})")
        fh.flush()

    records = [json.loads(l) for l in out_path.read_text().splitlines() if l.strip()]
    _summarize(records)
    print(f"\n[verbalize] verbalizations → {out_path}")


def _summarize(records: list[dict]) -> None:
    n_text = sum(1 for r in records if r["verbalization"] is not None)
    n_ref = sum(1 for r in records if r["is_refusal"])
    print(f"\n[verbalize] {len(records)} positions in file")
    print(f"  verbalized (genuine): {n_text - n_ref}")
    print(f"  refusals (AV safety training triggered): {n_ref}")
    print(f"  skipped (early/low-context, --skip-early-positions): {len(records) - n_text}")

    # Per-dataset refusal rate — high on one dataset means the AV recognises
    # that content type even through the activation bottleneck.
    by_ds: dict[str, list[int]] = {}
    for r in records:
        if r["verbalization"] is None:
            continue
        tot_ref = by_ds.setdefault(r["dataset"], [0, 0])
        tot_ref[0] += 1
        tot_ref[1] += int(r["is_refusal"])
    print("\n  refusal rate by dataset:")
    for ds, (tot, ref) in sorted(by_ds.items()):
        print(f"    {ds:<12}: {ref:>5}/{tot:<5}  ({ref / tot:.0%})")


_CJK_RANGE = range(0x4E00, 0x9FFF + 1)


def _check_cjk(text: str, sample_id: str, pos: int) -> None:
    cjk_chars = [c for c in text if ord(c) in _CJK_RANGE]
    if cjk_chars:
        print(f"  ⚠ CJK in verbalization ({sample_id} pos={pos}): "
              f"{cjk_chars[:5]} — injection may have failed (check hs_offset, "
              f"injection token, sidecar neighbor IDs)")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    from adv_datasets import DATASET_NAMES

    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ── extract ──────────────────────────────────────────────────────────────
    e = sub.add_parser(
        "extract",
        help="phase 1: base model → activations + token IDs + model output",
    )
    e.add_argument("--datasets", nargs="+", default=None,
                   choices=list(DATASET_NAMES),
                   help="which datasets to run (default: all four)")
    e.add_argument("--base-model", required=True,
                   help="e.g. Qwen/Qwen2.5-7B-Instruct")
    e.add_argument("--layer", type=int, required=True,
                   help="NLA extraction layer K (e.g. 20 for Qwen-L20)")
    e.add_argument("--hs-offset", type=int, default=1,
                   help="hidden_states[layer + offset]. Default 1. "
                        "Flip to 0 if verbalizations look like CJK garbage.")
    e.add_argument("--out", required=True, help="output directory")
    e.add_argument("--device", default="cuda")
    e.add_argument("--n-samples", type=int, default=None,
                   help="max samples per dataset (default: full dataset)")
    e.add_argument("--decode-tokens", type=int, default=64,
                   help="tokens to greedy-decode per sample for model_output")
    e.set_defaults(func=cmd_extract)

    # ── verbalize ─────────────────────────────────────────────────────────────
    v = sub.add_parser(
        "verbalize",
        help="phase 2: activations → AV → verbalization text per position",
    )
    v.add_argument("--out", required=True,
                   help="run directory from phase 1")
    v.add_argument("--av-checkpoint", required=True,
                   help="HF-format AV checkpoint dir (with nla_meta.yaml)")
    v.add_argument("--sglang-url", default="http://localhost:30000")
    v.add_argument("--datasets", nargs="+", default=None,
                   choices=list(DATASET_NAMES),
                   help="restrict to these datasets (default: all in samples.jsonl)")
    v.add_argument("--temperature", type=float, default=0.0,
                   help="AV sampling temperature (0 = greedy)")
    v.add_argument("--max-new-tokens", type=int, default=200)
    v.add_argument("--injection-scale", type=float, default=None,
                   help="override sidecar injection_scale (rarely needed)")
    v.add_argument("--workers", type=int, default=16,
                   help="concurrent AV requests in flight. SGLang batches them "
                        "server-side; sequential (1) is ~10x slower. Default 16.")
    v.add_argument("--skip-early-positions", action="store_true",
                   help=f"skip positions < {_MIN_POSITION} (set verbalization=null) "
                        "to mirror the datagen floor. Default: verbalize all "
                        "positions; early ones carry min_position_warning=True.")
    v.add_argument("--debug", action="store_true",
                   help="print CJK warning if injection looks wrong")
    v.set_defaults(func=cmd_verbalize)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
