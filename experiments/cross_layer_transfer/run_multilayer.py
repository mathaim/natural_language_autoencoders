"""All-layer NLA setup and smoke tests (Qwen2.5-7B, anchor layer 20).

Four subcommands:

  fetch      Download the final-step adapter checkpoints (5 files, ~2.1 GB)
             and optionally the 2k-document all-layer eval corpus. Run on a
             node with network access.

  check      CPU only, no AV/AR. Verifies the adapter port against real
             paired activations: `encode` must pull layer-l activations
             toward the layer-20 activation of the same token, and `decode`
             must do the reverse. Run this first after `fetch`.

  extract    One forward pass per prompt, raw residual stream saved at EVERY
             layer. Writes activations.npy [n_vectors, n_layers, d] float16
             and samples.jsonl.

  roundtrip  For every (vector, layer): encode → AV (SGLang) → text → AR →
             decode, then cosine against the original. Writes
             verbalizations.jsonl (resumable), reconstructions.npy and
             summary.json. This is the end-to-end GPU smoke test and the
             template for experiments that edit the text between AV and AR.

Layer convention everywhere: layer l = raw output of decoder block l,
captured with a forward hook (HF `hidden_states[-1]` is post-LayerNorm, so
it is not usable for the last layer).

Examples:

  python experiments/cross_layer_transfer/run_multilayer.py fetch \\
      --out /scratch/$USER/nla_ckpts/nla-adapter --with-eval-data

  python experiments/cross_layer_transfer/run_multilayer.py check \\
      --ckpt-root /scratch/$USER/nla_ckpts/nla-adapter

  python experiments/cross_layer_transfer/run_multilayer.py extract \\
      --base-model Qwen/Qwen2.5-7B-Instruct --prompts prompts.txt \\
      --out runs/demo --positions last

  # SGLang serving the AV checkpoint must be up (see slurm/submit_qwen7b.sh)
  python experiments/cross_layer_transfer/run_multilayer.py roundtrip \\
      --acts runs/demo/activations.npy --out runs/demo \\
      --adapter c_direct --ckpt-root /scratch/$USER/nla_ckpts/nla-adapter \\
      --av-checkpoint kitft/nla-qwen2.5-7b-L20-av \\
      --ar-checkpoint kitft/nla-qwen2.5-7b-L20-ar
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from adapters import (  # noqa: E402
    ADAPTER_REPO, ADAPTER_SUBDIR, DEFAULT_RUN, FINAL_CHECKPOINTS,
    load_adapters, resolve_checkpoint,
)

# Injection marker range (U+3200–33FF) plus common CJK ideographs. If
# injection silently fails the AV describes the marker char itself.
_CJK_RE = re.compile(r"[㈀-㏿一-鿿]")
_EVAL_FILES = ("data/activations.npy", "data/texts.json", "data/meta.json")


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a, b = a.astype(np.float32), b.astype(np.float32)
    return (a * b).sum(-1) / np.maximum(
        np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1), 1e-12)


def _load_acts(path: str) -> np.ndarray:
    acts = np.load(path, mmap_mode="r")
    assert acts.ndim == 3, (
        f"expected activations [n_vectors, n_layers, d], got shape {acts.shape}")
    return acts


# ─── fetch ───────────────────────────────────────────────────────────────────

def cmd_fetch(args: argparse.Namespace) -> None:
    from huggingface_hub import hf_hub_download

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [f"{ADAPTER_SUBDIR}/{FINAL_CHECKPOINTS[r]}" for r in args.runs]
    if args.with_eval_data:
        files += list(_EVAL_FILES)
    for f in files:
        p = hf_hub_download(repo_id=args.repo, filename=f, local_dir=str(out))
        print(f"[fetch] {Path(p).stat().st_size / 1e6:8.1f} MB  {p}")
    print(f"[fetch] done. Pass --ckpt-root {out} to the other subcommands.")


# ─── check ───────────────────────────────────────────────────────────────────

def cmd_check(args: argparse.Namespace) -> None:
    import torch

    data = args.data
    if data is None:
        local = Path(args.ckpt_root or ".") / _EVAL_FILES[0]
        if local.is_file():
            data = str(local)
        else:
            from huggingface_hub import hf_hub_download
            data = hf_hub_download(repo_id=args.repo, filename=_EVAL_FILES[0])
    acts = _load_acts(data)
    n = min(args.n, acts.shape[0])
    x = np.asarray(acts[:n], dtype=np.float32)          # [n, L, d]
    n_layers = x.shape[1]

    failed = []
    for run in args.adapters:
        path = resolve_checkpoint(run, args.ckpt_root, repo=args.repo)
        adapters, prov = load_adapters(path)
        assert adapters.n_layers == n_layers and adapters.d == x.shape[2], (
            f"{path.name}: adapter is {adapters.n_layers}x{adapters.d}, "
            f"activations are {n_layers}x{x.shape[2]}")
        k = adapters.anchor
        anchor = x[:, k]
        print(f"\n[check] {path.name}  method={adapters.method} anchor={k} "
              f"stage={prov['stage']} step={prov['step']}  ({n} documents)")
        print("  layer   cos(h_l,h_K)  cos(enc(h_l),h_K)   cos(h_K,h_l)  cos(dec(h_K),h_l)")
        gains = []
        for layer in range(n_layers):
            lay = torch.full((n,), layer, dtype=torch.long)
            with torch.inference_mode():
                enc = adapters.encode(torch.from_numpy(x[:, layer]), lay).numpy()
                dec = adapters.decode(torch.from_numpy(anchor), lay).numpy()
            raw = float(_cos(x[:, layer], anchor).mean())
            c_in = float(_cos(enc, anchor).mean())
            c_out = float(_cos(dec, x[:, layer]).mean())
            print(f"  {layer:5d}   {raw:12.3f}  {c_in:17.3f}   {raw:12.3f}  {c_out:17.3f}")
            if layer == k:
                assert np.array_equal(enc, x[:, layer]) and np.array_equal(dec, anchor), (
                    "anchor layer must bypass the adapters exactly")
            else:
                gains += [c_in - raw, c_out - raw]
        mean_gain = float(np.mean(gains))
        ok = mean_gain > 0.05
        print(f"  mean cosine gain over no-adapter: {mean_gain:+.3f}  "
              f"{'OK' if ok else 'FAIL — port or checkpoint is wrong'}")
        if not ok:
            failed.append(path.name)
    if failed:
        sys.exit(f"[check] FAILED: {failed}")
    print("\n[check] all adapters OK")


# ─── extract ─────────────────────────────────────────────────────────────────

def _read_prompts(path: str) -> list[dict]:
    """.jsonl with a "text" field (optional "id"), or plain text, one per line."""
    rows = []
    for i, line in enumerate(Path(path).read_text().splitlines()):
        if not line.strip():
            continue
        if path.endswith(".jsonl"):
            r = json.loads(line)
            rows.append({"sample_id": str(r.get("id", i)), "text": r["text"]})
        else:
            rows.append({"sample_id": str(i), "text": line})
    return rows


def cmd_extract(args: argparse.Namespace) -> None:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from multilayer import extract_all_layers

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prompts = _read_prompts(args.prompts)
    print(f"[extract] {len(prompts)} prompts; loading {args.base_model} …")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(args.device).eval()

    chunks: list[np.ndarray] = []
    samples: list[dict] = []
    n_vec = 0
    for i, p in enumerate(prompts):
        ids = tokenizer(p["text"], return_tensors="pt", add_special_tokens=True,
                        truncation=True, max_length=args.max_length)["input_ids"]
        hs = extract_all_layers(model, ids.to(args.device))      # [L, T, d]
        T = hs.shape[1]
        positions = [T - 1] if args.positions == "last" else list(range(T))
        # [n_pos, L, d]; raw vectors, float16 as in the adapter training data.
        chunks.append(hs[:, positions].permute(1, 0, 2).numpy().astype(np.float16))
        samples.append({**p, "token_ids": ids[0].tolist(), "n_tokens": T,
                        "positions": positions, "vec_start": n_vec})
        n_vec += len(positions)
        if (i + 1) % 25 == 0 or i == len(prompts) - 1:
            print(f"  {i + 1}/{len(prompts)}  ({n_vec} vectors)")

    acts = np.concatenate(chunks, axis=0)
    np.save(out / "activations.npy", acts)
    (out / "samples.jsonl").write_text("".join(json.dumps(s) + "\n" for s in samples))
    print(f"[extract] activations {acts.shape} {acts.dtype} → {out}/activations.npy")
    print(f"[extract] samples → {out}/samples.jsonl  (vector row for a sample's "
          f"position p = vec_start + positions.index(p))")


# ─── roundtrip ───────────────────────────────────────────────────────────────

def cmd_roundtrip(args: argparse.Namespace) -> None:
    from multilayer import MultiLayerNLA

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    verb_path = out / "verbalizations.jsonl"
    acts = _load_acts(args.acts)
    n_vec, n_layers, d = acts.shape
    rows = list(range(n_vec))[: args.n or None]
    layers = args.layers or list(range(n_layers))
    assert all(0 <= l < n_layers for l in layers), f"layer out of range 0..{n_layers - 1}"

    mnla = MultiLayerNLA.from_checkpoints(
        args.adapter, args.av_checkpoint, args.ar_checkpoint,
        ckpt_root=args.ckpt_root, sglang_url=args.sglang_url,
        critic_device=args.device,
    )

    # ── Resume. A SLURM timeout can leave a truncated last line; drop it.
    done: dict[tuple[int, int], str] = {}
    if verb_path.exists():
        kept = []
        for line in verb_path.read_text().splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            done[(r["vec_idx"], r["layer"])] = r["verbalization"]
            kept.append(line)
        verb_path.write_text("".join(l + "\n" for l in kept))
        print(f"[roundtrip] resuming — {len(done)} already in {verb_path.name}")

    # ── AV pass. Rows from every layer go out together; SGLang's batcher
    # packs them. Failed requests are not written, so a rerun retries them.
    work = [(v, l) for v in rows for l in layers if (v, l) not in done]
    print(f"[roundtrip] {len(work)} (vector, layer) pairs to verbalize, "
          f"{args.workers} concurrent requests")
    t0 = time.time()
    n_ok = n_err = 0
    with verb_path.open("a") as fh:
        for start in range(0, len(work), args.chunk):
            chunk = work[start:start + args.chunk]
            x = np.stack([np.asarray(acts[v, l], dtype=np.float32) for v, l in chunk])
            lay = [l for _, l in chunk]
            for i, res in mnla.verbalize_iter(
                    x, lay, workers=args.workers, temperature=args.temperature,
                    max_new_tokens=args.max_new_tokens):
                if isinstance(res, Exception):
                    n_err += 1
                    if n_err <= 5:
                        print(f"[roundtrip] ✗ request failed: {type(res).__name__}: {res}")
                    if n_err >= 25:
                        sys.exit("[roundtrip] 25 failures — server is probably "
                                 "down. Rerun to resume from here.")
                    continue
                v, l = chunk[i]
                done[(v, l)] = res
                fh.write(json.dumps({"vec_idx": v, "layer": l, "verbalization": res}) + "\n")
                n_ok += 1
                if n_ok % 100 == 0:
                    fh.flush()
                if n_ok % 500 == 0:
                    rate = n_ok / (time.time() - t0)
                    print(f"  {n_ok}/{len(work)}  ({rate:.1f}/s, "
                          f"ETA {(len(work) - n_ok) / rate / 60:.0f} min)")
    if n_err:
        sys.exit(f"[roundtrip] {n_err} requests failed — rerun to retry them "
                 f"before the AR pass.")

    # ── AR pass. One reconstruction per text, then the output adapter and a
    # rescale to the original activation's norm.
    print("[roundtrip] reconstructing …")
    recon = np.zeros((len(rows), len(layers), d), dtype=np.float16)
    summary: dict[str, dict] = {}
    for j, l in enumerate(layers):
        texts = [done[(v, l)] for v in rows]
        gold = np.stack([np.asarray(acts[v, l], dtype=np.float32) for v in rows])
        anchor_vecs = mnla.reconstruct_anchor(texts)
        dec = mnla.decode(anchor_vecs, l, target_norms=np.linalg.norm(gold, axis=-1)).numpy()
        recon[:, j] = dec.astype(np.float16)
        summary[str(l)] = {
            "n": len(rows),
            # AV→AR→output adapter vs the original layer-l activation.
            "cos_roundtrip": float(_cos(dec, gold).mean()),
            # Same AR output with NO output adapter: what the adapter adds.
            "cos_ar_no_output_adapter": float(_cos(anchor_vecs.numpy(), gold).mean()),
            # Loudest smoke test for a silently failed injection.
            "cjk_rate": float(np.mean([bool(_CJK_RE.search(t)) for t in texts])),
        }
        s = summary[str(l)]
        print(f"  layer {l:2d}: cos {s['cos_roundtrip']:.3f}  "
              f"(no output adapter {s['cos_ar_no_output_adapter']:.3f})  "
              f"cjk {s['cjk_rate']:.2f}")

    np.save(out / "reconstructions.npy", recon)
    (out / "summary.json").write_text(json.dumps({
        "adapter": args.adapter, "av_checkpoint": str(args.av_checkpoint),
        "ar_checkpoint": str(args.ar_checkpoint), "acts": str(args.acts),
        "rows": rows, "layers": layers, "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens, "per_layer": summary,
    }, indent=2))
    worst = max(s["cjk_rate"] for s in summary.values())
    if worst > 0.2:
        print(f"[roundtrip] ⚠ CJK in up to {worst:.0%} of explanations at some layer — "
              f"injection may be failing. See docs/inference.md § Debugging.")
    print(f"[roundtrip] → {out}/summary.json, reconstructions.npy {recon.shape}")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    runs = sorted(FINAL_CHECKPOINTS)

    f = sub.add_parser("fetch", help="download final-step adapter checkpoints")
    f.add_argument("--out", required=True, help="local dir (becomes --ckpt-root)")
    f.add_argument("--runs", nargs="+", choices=runs, default=runs)
    f.add_argument("--with-eval-data", action="store_true",
                   help="also fetch the 2k-doc all-layer eval corpus (~404 MB)")
    f.add_argument("--repo", default=ADAPTER_REPO)
    f.set_defaults(func=cmd_fetch)

    c = sub.add_parser("check", help="CPU-only check of the adapter port")
    c.add_argument("--adapters", nargs="+", default=runs,
                   help="run names or .pt paths (default: all five final runs)")
    c.add_argument("--ckpt-root", default=None)
    c.add_argument("--data", default=None,
                   help="activations.npy [N, n_layers, d] (default: the eval corpus)")
    c.add_argument("--n", type=int, default=200, help="documents to use")
    c.add_argument("--repo", default=ADAPTER_REPO)
    c.set_defaults(func=cmd_check)

    e = sub.add_parser("extract", help="all-layer activations for a prompt file")
    e.add_argument("--base-model", required=True)
    e.add_argument("--prompts", required=True,
                   help=".txt (one prompt per line) or .jsonl with a 'text' field")
    e.add_argument("--out", required=True)
    e.add_argument("--positions", choices=["last", "all"], default="last")
    e.add_argument("--max-length", type=int, default=512)
    e.add_argument("--device", default="cuda")
    e.set_defaults(func=cmd_extract)

    r = sub.add_parser("roundtrip", help="AV → AR round trip at every layer")
    r.add_argument("--acts", required=True, help="activations.npy [N, n_layers, d]")
    r.add_argument("--out", required=True)
    r.add_argument("--adapter", default=DEFAULT_RUN,
                   help=f"run name, .pt path, or 'none' for zero-shot (default: {DEFAULT_RUN})")
    r.add_argument("--ckpt-root", default=None)
    r.add_argument("--av-checkpoint", required=True, help="anchor-layer AV: dir or HF id")
    r.add_argument("--ar-checkpoint", required=True, help="anchor-layer AR: dir or HF id")
    r.add_argument("--sglang-url", default="http://localhost:30000")
    r.add_argument("--layers", type=int, nargs="+", default=None, help="default: all")
    r.add_argument("--n", type=int, default=0, help="first N vectors (0 = all)")
    r.add_argument("--workers", type=int, default=16)
    r.add_argument("--chunk", type=int, default=2048,
                   help="(vector, layer) pairs encoded per batch")
    r.add_argument("--temperature", type=float, default=0.0)
    r.add_argument("--max-new-tokens", type=int, default=200)
    r.add_argument("--device", default="cuda", help="device for the AR critic")
    r.set_defaults(func=cmd_roundtrip)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
