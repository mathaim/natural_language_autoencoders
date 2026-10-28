"""ROME ablation sweep — patch EVERY input-token position, measure fact recall.

The leakage problem: earlier experiments patched only "subject" or "last"
position, while leaving all other prompt positions at their original
embeddings. The model could recover the fact from the unpatched positions.
Fix: for every condition we replace ALL prompt token embeddings, so there is
no unpatched residual stream for the model to fall back on.

Five conditions, in order from ablation-floor to aspirational:

  baseline     no injection — raw model output (gold reference)
  empty        AR("") at every position — does erasing all content suppress recall?
  control      AR(AV(h_l)) at every position — round-trip with no edit; how
               much information survives the text bottleneck?
  forget       AR(AV(h_l) + " Forget {gold}.") at every position — does the
               forget instruction actually transmit through the bottleneck?
  ground_truth AR("The answer to this prompt is {gold}.") at every position —
               upper-bound: if the reconstruction explicitly says the answer,
               does recall go up?

Measurement: after patching, does the base model's top-1 next-token at the
final prompt position equal the gold answer?  We report:
  - recall@1  per condition
  - Δ vs baseline (negative = forgetting, positive = leakage/amplification)

Usage
─────
  # Phase 1: extract activations for all prompt positions
  python sweep.py extract \
      --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 \
      --counterfact /data/counterfact.json --n-samples 200 \
      --out runs/qwen_l20

  # Phase 2: run the ablation sweep (needs SGLang AV server + AR checkpoint)
  python sweep.py sweep \
      --out runs/qwen_l20 \
      --av-checkpoint kitft/nla-qwen2.5-7b-L20-av \
      --ar-checkpoint kitft/nla-qwen2.5-7b-L20-ar \
      --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 \
      --sglang-url http://localhost:30000 \
      --conditions baseline empty control forget ground_truth

Notes
─────
- AR("") is what the critic produces from an empty prompt — it's not the zero
  vector; the critic's truncated LM still produces some output at the final
  token position. Use it as the ablation floor, not as "zero injection."
- The base model runs with inputs_embeds (not input_ids) for every patched
  condition, so the forward pass is identical in structure across conditions.
  "baseline" is the only exception: it uses input_ids directly, which is the
  true unpatched baseline.
- Gemma-3 embedding forward() multiplies by √d. resolve_embed_scale() corrects
  for this when we build the patched embeds; the AR output is in residual-stream
  space (already √d-scaled by the base model's training), so we do NOT apply
  the scale to AR vectors — only to the lookup table rows that we keep at their
  original token embeddings.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent / "verbalization_screening"))


# ─── Condition registry ──────────────────────────────────────────────────────

ALL_CONDITIONS = ("baseline", "empty", "control", "forget", "ground_truth")


# ─── Phase 1: extract all token positions ────────────────────────────────────

def cmd_extract(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from facts import load_counterfact, builtin_capitals, filter_single_token

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[extract] loading {args.base_model} …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()
    print(f"[extract] {model.config.num_hidden_layers} layers, "
          f"d_model={model.config.hidden_size}")

    if args.counterfact:
        facts = load_counterfact(
            args.counterfact, template_relation=args.template_relation,
            limit=args.n_samples,
        )
    else:
        facts = builtin_capitals()

    kept, dropped = filter_single_token(facts, tokenizer)
    if dropped:
        print(f"[extract] dropped {len(dropped)} multi-token facts: "
              f"{', '.join(f.answer for f in dropped[:8])}")
    if args.n_samples:
        kept = kept[: args.n_samples]
    print(f"[extract] {len(kept)} facts, extracting all positions at layer "
          f"{args.layer} (hidden_states[{args.layer + args.hs_offset}])")

    samples: list[dict] = []       # metadata per sample
    all_vectors: list[np.ndarray] = []  # shape [T_i, d_model] each

    with torch.inference_mode():
        for i, f in enumerate(kept):
            ids = tokenizer(
                f.prompt, return_tensors="pt", add_special_tokens=True,
            )["input_ids"].to(args.device)

            out_fwd = model(ids, output_hidden_states=True, use_cache=False)
            hs = out_fwd.hidden_states[args.layer + args.hs_offset][0]  # [T, d]

            # Base model next-token prediction (unpatched baseline)
            top1_id = int(out_fwd.logits[0, -1].argmax())
            top1_tok = tokenizer.decode([top1_id])
            knows = top1_tok.strip().lower() == f.answer.strip().lower()

            vec_start = len(all_vectors)
            T = ids.shape[1]
            # Store each token position as its own row in all_vectors so we
            # can load a single matrix and slice by [vec_start : vec_start + T].
            for t in range(T):
                all_vectors.append(hs[t].float().cpu().numpy())

            samples.append({
                "sample_idx": i,
                "subject": f.subject,
                "answer": f.answer,
                "prompt": f.prompt,
                "relation": f.relation,
                "n_tokens": T,
                "vec_start": vec_start,    # index into activations.npy
                "token_ids": ids[0].tolist(),
                "model_top1": top1_tok,
                "model_knows": knows,
            })
            if (i + 1) % 20 == 0:
                print(f"  {i + 1}/{len(kept)}  "
                      f"knows={sum(s['model_knows'] for s in samples)}")

    vec_matrix = np.stack(all_vectors).astype(np.float32)
    np.save(out / "activations.npy", vec_matrix)
    (out / "samples.jsonl").write_text(
        "".join(json.dumps(s) + "\n" for s in samples)
    )
    n_knows = sum(s["model_knows"] for s in samples)
    print(f"\n[extract] {len(kept)} facts; model knew {n_knows} ({n_knows/len(kept):.0%})")
    print(f"[extract] {vec_matrix.shape[0]} total token vectors → activations.npy")
    print(f"[extract] next: start SGLang AV server, then run `sweep --out {out}`")


# ─── Phase 2: ablation sweep ─────────────────────────────────────────────────

def _ar_reconstruct(ar_critic, text: str) -> np.ndarray:
    """Return AR's reconstruction of `text` as a float32 numpy vector."""
    return ar_critic.reconstruct(text).float().cpu().numpy()


def _build_patched_embeds(
    token_ids: list[int],
    replacement_vectors: np.ndarray,  # [T, d_model] float32
    base_embed: torch.nn.Embedding,
    embed_scale: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a [1, T, d_model] inputs_embeds tensor where every position is
    replaced by the corresponding replacement vector.

    Token embeddings from the lookup table are kept ONLY for the baseline
    condition (where replacement_vectors is None). Here every position is
    replaced, so we never call base_embed at all.
    """
    vecs = torch.from_numpy(replacement_vectors).to(device=device, dtype=dtype)
    return vecs.unsqueeze(0)    # [1, T, d_model]


def _build_original_embeds(
    token_ids: list[int],
    base_embed: torch.nn.Embedding,
    embed_scale: float,
    device: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build the unpatched [1, T, d_model] embeds from the token lookup table.

    Used for the baseline condition. Applies embed_scale to match
    architecture (Gemma-3 forward multiplies by √d in its Embedding class;
    since we bypassed that class here we apply the scale manually).
    """
    ids_t = torch.tensor(token_ids, dtype=torch.long, device=device)
    embs = base_embed(ids_t).to(dtype)   # [T, d_model]
    if embed_scale != 1.0:
        embs = embs * embed_scale
    return embs.unsqueeze(0)


def _top1_from_embeds(
    model,
    embeds: torch.Tensor,  # [1, T, d_model]
) -> int:
    with torch.inference_mode():
        out = model(inputs_embeds=embeds, use_cache=False)
    return int(out.logits[0, -1].argmax())


def cmd_sweep(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from nla_inference import NLAClient, NLACritic, resolve_embed_scale, load_embedding_only

    out = Path(args.out)
    vec_matrix = np.load(out / "activations.npy")
    samples = [json.loads(l) for l in (out / "samples.jsonl").read_text().splitlines()]
    conditions = args.conditions

    print(f"[sweep] {len(samples)} samples, conditions: {conditions}")
    print(f"[sweep] loading base model {args.base_model} …")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    # Embedding table for rebuilding embeddings
    embed_scale = resolve_embed_scale(args.base_model)
    base_embed = load_embedding_only(args.base_model, dtype=torch.bfloat16).to(args.device)
    print(f"[sweep] embed_scale={embed_scale:.4f}")

    # NLA pair
    av_client: NLAClient | None = None
    ar_critic: NLACritic | None = None
    if any(c != "baseline" for c in conditions):
        print(f"[sweep] loading AV …")
        av_client = NLAClient(
            args.av_checkpoint,
            sglang_url=args.sglang_url,
            injection_scale_override=args.injection_scale,
        )
    if any(c not in ("baseline",) for c in conditions):
        print(f"[sweep] loading AR critic …")
        ar_critic = NLACritic(args.ar_checkpoint, device=args.device)

    results: list[dict] = []

    for s in samples:
        T = s["n_tokens"]
        vecs = vec_matrix[s["vec_start"]: s["vec_start"] + T]   # [T, d_model]
        gold = s["answer"]
        gold_id = tokenizer.encode(" " + gold, add_special_tokens=False)[0]

        row: dict = {
            "sample_idx": s["sample_idx"],
            "subject": s["subject"],
            "answer": gold,
            "prompt": s["prompt"],
            "model_knows": s["model_knows"],
        }

        for cond in conditions:
            if cond == "baseline":
                # True unpatched forward — use input_ids, not embeds
                with torch.inference_mode():
                    ids_t = torch.tensor(
                        s["token_ids"], dtype=torch.long, device=args.device
                    ).unsqueeze(0)
                    top1_id = int(model(ids_t).logits[0, -1].argmax())
            else:
                # Build replacement vectors for ALL T positions
                replacement_rows: list[np.ndarray] = []
                verbalizations: list[str] = []

                for t_idx in range(T):
                    h = vecs[t_idx]  # [d_model]

                    if cond == "empty":
                        text = ""
                    elif cond == "ground_truth":
                        text = f"The answer to this prompt is {gold}."
                    elif cond in ("control", "forget"):
                        # verbalize this position's activation
                        text = av_client.generate(
                            h,
                            temperature=args.temperature,
                            max_new_tokens=args.max_new_tokens,
                        )
                        if cond == "forget":
                            text = text + f" Forget {gold}."
                    else:
                        raise ValueError(f"unknown condition: {cond!r}")

                    verbalizations.append(text)
                    recon = _ar_reconstruct(ar_critic, text)
                    replacement_rows.append(recon)

                replacement_matrix = np.stack(replacement_rows)  # [T, d_model]
                embeds = _build_patched_embeds(
                    s["token_ids"], replacement_matrix,
                    base_embed, embed_scale,
                    args.device, torch.bfloat16,
                )
                top1_id = _top1_from_embeds(model, embeds)
                row[f"{cond}_verbalizations"] = verbalizations

            top1_tok = tokenizer.decode([top1_id])
            recalls = top1_tok.strip().lower() == gold.strip().lower()
            row[f"{cond}_top1"] = top1_tok
            row[f"{cond}_recall"] = recalls

        results.append(row)
        _print_row(row, conditions, s["model_knows"])

    (out / "sweep_results.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in results)
    )
    _report(results, conditions)
    print(f"\n[sweep] full results → {out}/sweep_results.jsonl")


def _print_row(row: dict, conditions: list[str], knows: bool) -> None:
    flag = "K" if knows else "?"
    parts = [f"[{flag}] {row['subject']:>14} → {row['answer']:<10}"]
    for c in conditions:
        r = row.get(f"{c}_recall")
        if r is None:
            parts.append(f"{c}=?")
        else:
            parts.append(f"{c}={'✓' if r else '✗'}")
    print("  " + "  ".join(parts))


def _report(results: list[dict], conditions: list[str]) -> None:
    knows_rows = [r for r in results if r["model_knows"]]
    all_rows = results
    denom = knows_rows if knows_rows else all_rows

    baseline_rate = sum(r.get("baseline_recall", False) for r in denom) / len(denom)

    print("\n" + "═" * 72)
    print("ROME ABLATION SWEEP — recall@1 after patching ALL token positions")
    print(f"(over {len(denom)} facts the base model knows; {len(all_rows)} total)")
    print("═" * 72)
    print(f"\n  {'condition':<16}  {'recall@1':>8}  {'Δ vs baseline':>14}  {'interpretation'}")
    print(f"  {'-'*16}  {'-'*8}  {'-'*14}  {'-'*32}")

    for cond in conditions:
        n_recall = sum(r.get(f"{cond}_recall", False) for r in denom)
        rate = n_recall / len(denom)
        delta = rate - baseline_rate
        interp = _interpret(cond, rate, delta)
        delta_str = f"{delta:+.1%}"
        print(f"  {cond:<16}  {rate:>7.1%}  {delta_str:>14}  {interp}")

    print("\n" + "═" * 72)
    print("read:")
    print("  empty≈0            → erasing activations fully suppresses recall")
    print("  control≈baseline   → round-trip is faithful; text bottleneck preserves info")
    print("  forget<baseline    → forget instruction transmits (good!)")
    print("  forget≈baseline    → forget instruction does NOT transmit (text bottleneck lossy)")
    print("  ground_truth>baseline → AR can amplify explicit text cues")
    print("═" * 72)


def _interpret(cond: str, rate: float, delta: float) -> str:
    if cond == "baseline":
        return "gold reference"
    if cond == "empty":
        if rate < 0.10:
            return "✓ complete erasure (floor)"
        if rate < 0.30:
            return "partial erasure"
        return "✗ leakage survives AR(\"\") reconstruction"
    if cond == "control":
        if abs(delta) < 0.05:
            return "faithful round-trip"
        if delta < 0:
            return f"round-trip lossy ({delta:+.1%})"
        return f"round-trip amplifies ({delta:+.1%})"
    if cond == "forget":
        if delta < -0.15:
            return f"✓ forget instruction transmitted ({delta:+.1%})"
        if delta < -0.05:
            return f"weak transmission ({delta:+.1%})"
        return "✗ forget instruction not transmitted"
    if cond == "ground_truth":
        if delta > 0.05:
            return f"AR amplifies explicit label ({delta:+.1%})"
        return f"AR ignores/distorts explicit label ({delta:+.1%})"
    return ""


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    # ── extract ──
    e = sub.add_parser(
        "extract",
        help="phase 1: base model → activations at ALL token positions",
    )
    e.add_argument("--base-model", required=True,
                   help="e.g. Qwen/Qwen2.5-7B-Instruct")
    e.add_argument("--layer", type=int, required=True,
                   help="NLA extraction layer K")
    e.add_argument("--hs-offset", type=int, default=1,
                   help="hidden_states[layer + offset]. Default 1 (block output).")
    e.add_argument("--out", required=True, help="run directory")
    e.add_argument("--device", default="cuda")
    e.add_argument("--counterfact", default=None,
                   help="path to CounterFact JSON (ROME dataset)")
    e.add_argument("--template-relation", default=None,
                   help="filter CounterFact by this prompt substring, e.g. 'capital of'")
    e.add_argument("--n-samples", type=int, default=None,
                   help="max facts to process (default: all)")
    e.set_defaults(func=cmd_extract)

    # ── sweep ──
    s = sub.add_parser(
        "sweep",
        help="phase 2: run ablation conditions on extracted activations",
    )
    s.add_argument("--out", required=True,
                   help="run directory from phase 1 (contains activations.npy + samples.jsonl)")
    s.add_argument("--av-checkpoint", required=True,
                   help="HF-format AV checkpoint dir (with nla_meta.yaml)")
    s.add_argument("--ar-checkpoint", required=True,
                   help="HF-format AR checkpoint dir (with nla_meta.yaml)")
    s.add_argument("--base-model", required=True,
                   help="same base model used in extraction, for the forward pass")
    s.add_argument("--layer", type=int, required=True,
                   help="same layer used in extraction")
    s.add_argument("--sglang-url", default="http://localhost:30000")
    s.add_argument("--conditions", nargs="+", default=list(ALL_CONDITIONS),
                   choices=list(ALL_CONDITIONS),
                   help=f"which conditions to run (default: all = {ALL_CONDITIONS})")
    s.add_argument("--temperature", type=float, default=0.0,
                   help="AV sampling temperature (0=greedy)")
    s.add_argument("--max-new-tokens", type=int, default=200)
    s.add_argument("--injection-scale", type=float, default=None,
                   help="override sidecar injection_scale (rarely needed)")
    s.add_argument("--device", default="cuda")
    s.set_defaults(func=cmd_sweep)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
