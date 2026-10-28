"""Run the full adversarial extraction pipeline across all NLA model families.

For each model:
  1. Extract activations (base model forward pass)
  2. Launch an SGLang AV server
  3. Verbalize all positions
  4. Kill the SGLang server
  5. Move to the next model

Results land in:
  {out_root}/{model_name}/activations.npy
  {out_root}/{model_name}/samples.jsonl
  {out_root}/{model_name}/verbalizations.jsonl

Prerequisites
─────────────
  - CUDA GPUs with enough VRAM for each base model
    (7B: ~16GB, 12B: ~24GB, 27B: ~56GB, 70B: ~140GB or multi-GPU)
  - HF_TOKEN set in environment for Gemma models (gated repos)
  - SGLang installed: uv pip install "sglang[all]>=0.5.6"
  - Port 30000 free (or pass --sglang-port)

Usage
─────
  # All four models, all four datasets, 200 samples per dataset
  python run_all.py \\
      --out-root runs/adversarial \\
      --n-samples 200

  # Specific models only
  python run_all.py \\
      --models qwen7b-l20 gemma12b-l32 \\
      --out-root runs/adversarial \\
      --n-samples 200

  # Specific datasets only
  python run_all.py \\
      --datasets wmdp harmbench \\
      --out-root runs/adversarial \\
      --n-samples 200

  # Skip extraction (reuse existing activations.npy) — only verbalize
  python run_all.py --out-root runs/adversarial --skip-extract

  # Skip verbalization — only extract (useful to queue extraction first,
  # then verbalize later when the AV server is available)
  python run_all.py --out-root runs/adversarial --skip-verbalize

SGLang flags
────────────
  --sglang-port          default 30000
  --sglang-tp            tensor_parallel_size for multi-GPU (default 1;
                         use 4 or 8 for the 70B model)
  --sglang-mem-fraction  mem_fraction_static (default 0.85)
  --sglang-extra-args    any extra flags passed verbatim to sglang.launch_server
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_DIR = Path(__file__).parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(EXPERIMENT_DIR))

from nla_model_configs import NLA_MODELS, NLAModelConfig, resolve_checkpoints
from adv_datasets import DATASET_NAMES

EXTRACT_SCRIPT = EXPERIMENT_DIR / "extract_adversarial.py"

_SGLANG_READY_MARKER = "The server is fired up and ready to roll"
_SGLANG_STARTUP_TIMEOUT = 300   # seconds — large models take time to load
_SGLANG_POLL_INTERVAL = 5


# ─── SGLang server management ────────────────────────────────────────────────

def _find_python(sglang_python: str | None) -> str:
    """Resolve the Python binary to use for the SGLang server subprocess.

    If --sglang-python is given, use it directly (e.g. the python from a
    dedicated sglang-server conda env). Falls back to sys.executable so that
    single-env setups still work.
    """
    if sglang_python:
        return sglang_python
    return sys.executable


def _start_sglang(
    model_path: str,
    port: int,
    tp: int,
    mem_fraction: float,
    extra_args: list[str],
    log_path: Path,
    sglang_python: str | None = None,
) -> subprocess.Popen:
    python = _find_python(sglang_python)
    cmd = [
        python, "-m", "sglang.launch_server",
        "--model-path", model_path,
        "--port", str(port),
        "--disable-radix-cache",
        "--trust-remote-code",
        "--mem-fraction-static", str(mem_fraction),
    ]
    if tp > 1:
        cmd += ["--tensor-parallel-size", str(tp)]
    cmd += extra_args

    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w")
    print(f"[sglang] launching: {' '.join(cmd)}")
    print(f"[sglang] log → {log_path}")
    proc = subprocess.Popen(
        cmd, stdout=log_file, stderr=subprocess.STDOUT,
        env={**os.environ},
    )
    return proc


def _wait_for_sglang(log_path: Path, timeout: int, poll: int) -> bool:
    """Poll the SGLang log until the ready marker appears or timeout."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        time.sleep(poll)
        if log_path.exists():
            text = log_path.read_text(errors="ignore")
            if _SGLANG_READY_MARKER in text:
                return True
            if "Error" in text or "Traceback" in text:
                print(f"[sglang] ⚠  error detected in log — check {log_path}")
        print(f"[sglang] waiting for server (up to {int(deadline - time.time())}s left) …")
    return False


def _stop_sglang(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        print("[sglang] shutting down …")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()


# ─── Per-model pipeline ──────────────────────────────────────────────────────

def run_model(
    cfg: NLAModelConfig,
    out_root: Path,
    args: argparse.Namespace,
) -> None:
    out = out_root / cfg.name
    out.mkdir(parents=True, exist_ok=True)

    print(f"\n{'═' * 72}")
    print(f"MODEL: {cfg.name}  ({cfg.base_model}  layer {cfg.layer}/{cfg.n_layers})")
    print(f"{'═' * 72}")

    if cfg.requires_hf_token and not os.environ.get("HF_TOKEN"):
        print(f"[warn] {cfg.name} requires HF_TOKEN for gated Gemma repo — "
              f"set it in your environment before running.")

    # ── Phase 1: extract ────────────────────────────────────────────────────
    activations_path = out / "activations.npy"
    if args.skip_extract and activations_path.exists():
        print(f"[extract] skipping — {activations_path} already exists")
    else:
        extract_cmd = [
            sys.executable, str(EXTRACT_SCRIPT), "extract",
            "--base-model", cfg.base_model,
            "--layer", str(cfg.layer),
            "--out", str(out),
            "--device", args.device,
            "--decode-tokens", str(args.decode_tokens),
        ]
        if args.datasets:
            extract_cmd += ["--datasets"] + args.datasets
        if args.n_samples:
            extract_cmd += ["--n-samples", str(args.n_samples)]

        print(f"[extract] running …")
        result = subprocess.run(extract_cmd, check=True)
        if result.returncode != 0:
            print(f"[extract] FAILED for {cfg.name} — skipping verbalize")
            return

    # ── Phase 2: verbalize ──────────────────────────────────────────────────
    verbalizations_path = out / "verbalizations.jsonl"
    if args.skip_verbalize:
        print(f"[verbalize] skipping (--skip-verbalize)")
        return
    if verbalizations_path.exists() and not args.reverbalize:
        print(f"[verbalize] skipping — {verbalizations_path} already exists "
              f"(pass --reverbalize to overwrite)")
        return

    sglang_log = out / "sglang_av.log"
    sglang_proc = _start_sglang(
        model_path=cfg.av_checkpoint,
        port=args.sglang_port,
        tp=args.sglang_tp,
        mem_fraction=args.sglang_mem_fraction,
        extra_args=args.sglang_extra_args or [],
        log_path=sglang_log,
        sglang_python=getattr(args, "sglang_python", None),
    )

    try:
        ready = _wait_for_sglang(sglang_log, _SGLANG_STARTUP_TIMEOUT, _SGLANG_POLL_INTERVAL)
        if not ready:
            print(f"[sglang] TIMEOUT waiting for {cfg.av_checkpoint} — "
                  f"check {sglang_log}")
            return

        print(f"[sglang] server ready on port {args.sglang_port}")

        verbalize_cmd = [
            sys.executable, str(EXTRACT_SCRIPT), "verbalize",
            "--out", str(out),
            "--av-checkpoint", cfg.av_checkpoint,
            "--sglang-url", f"http://localhost:{args.sglang_port}",
            "--temperature", str(args.temperature),
            "--max-new-tokens", str(args.max_new_tokens),
        ]
        if args.datasets:
            verbalize_cmd += ["--datasets"] + args.datasets
        if args.verbalize_all:
            verbalize_cmd.append("--verbalize-all")
        if args.debug:
            verbalize_cmd.append("--debug")

        print(f"[verbalize] running …")
        subprocess.run(verbalize_cmd, check=True)

    finally:
        _stop_sglang(sglang_proc)


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Checkpoint root (local path instead of HF hub IDs)
    ap.add_argument("--ckpt-root", default=None,
                    help="local directory containing downloaded NLA checkpoints "
                         "(e.g. /scratch/euh7ys/nla_ckpts). Subdir names must "
                         "match the HF repo slug, e.g. qwen2.5-7b-L20-av/. "
                         "Falls back to HF hub download if a subdir is missing.")

    # Model selection
    ap.add_argument("--models", nargs="+", default=None,
                    choices=list(NLA_MODELS),
                    help="which NLA models to run (default: all four). "
                         f"choices: {list(NLA_MODELS)}")

    # Dataset selection
    ap.add_argument("--datasets", nargs="+", default=None,
                    choices=list(DATASET_NAMES),
                    help="which datasets to run (default: all four)")
    ap.add_argument("--n-samples", type=int, default=None,
                    help="max samples per dataset per model (default: full dataset)")

    # Output
    ap.add_argument("--out-root", required=True,
                    help="root directory; results go in {out-root}/{model-name}/")

    # Phase control
    ap.add_argument("--skip-extract", action="store_true",
                    help="skip extraction if activations.npy already exists")
    ap.add_argument("--skip-verbalize", action="store_true",
                    help="skip verbalization entirely (extract only)")
    ap.add_argument("--reverbalize", action="store_true",
                    help="re-run verbalization even if verbalizations.jsonl exists")

    # Extract options
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--decode-tokens", type=int, default=64,
                    help="tokens to greedy-decode for model_output")

    # Verbalize options
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--verbalize-all", action="store_true",
                    help="verbalize positions < min_position too")
    ap.add_argument("--debug", action="store_true",
                    help="print CJK warning on injection failure")

    # SGLang options
    ap.add_argument("--sglang-python", default=None,
                    help="Python binary to use for the SGLang server subprocess. "
                         "Use this to point at a dedicated sglang conda env when "
                         "the main env has incompatible deps, e.g.: "
                         "/home/euh7ys/miniconda3/envs/sglang-server/bin/python")
    ap.add_argument("--sglang-port", type=int, default=30000)
    ap.add_argument("--sglang-tp", type=int, default=1,
                    help="tensor_parallel_size for SGLang (use 4–8 for 70B)")
    ap.add_argument("--sglang-mem-fraction", type=float, default=0.85)
    ap.add_argument("--sglang-extra-args", nargs="*", default=None,
                    help="extra args passed verbatim to sglang.launch_server")

    args = ap.parse_args()

    models_to_run = args.models or list(NLA_MODELS)
    out_root = Path(args.out_root)

    print(f"Running {len(models_to_run)} model(s): {models_to_run}")
    print(f"Datasets: {args.datasets or list(DATASET_NAMES)}")
    print(f"Output root: {out_root}")
    if args.ckpt_root:
        print(f"Checkpoint root: {args.ckpt_root}")

    for model_name in models_to_run:
        cfg = NLA_MODELS[model_name]
        if args.ckpt_root:
            cfg = resolve_checkpoints(cfg, args.ckpt_root)
        try:
            run_model(cfg, out_root, args)
        except subprocess.CalledProcessError as e:
            print(f"\n[run_all] ✗ {model_name} failed (exit {e.returncode}) — continuing")
        except KeyboardInterrupt:
            print(f"\n[run_all] interrupted — stopping")
            sys.exit(1)

    print(f"\n{'═' * 72}")
    print("ALL DONE")
    for model_name in models_to_run:
        out = out_root / model_name
        v = out / "verbalizations.jsonl"
        a = out / "activations.npy"
        status = "✓" if v.exists() else ("extract-only" if a.exists() else "✗ failed")
        print(f"  {model_name:<16}: {status}  →  {out}")
    print(f"{'═' * 72}")


if __name__ == "__main__":
    main()
