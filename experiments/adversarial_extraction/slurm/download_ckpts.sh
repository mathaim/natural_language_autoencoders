#!/bin/bash
#SBATCH --job-name=nla_download
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/euh7ys/logs/nla_download_%j.out
#SBATCH --error=/scratch/euh7ys/logs/nla_download_%j.err
#
# CPU-only: pre-download the 27B and 70B NLA checkpoints plus their gated base
# models into /scratch so the GPU jobs don't spend their allocation downloading.
# ~600 GB total. Requires HF_TOKEN and accepted licenses for Gemma-3 and
# Llama-3.3 on the HF Hub. Each download is independent — one gated failure
# doesn't stop the rest; the summary at the end says what landed.
#
# Directory names match what nla_model_configs.resolve_checkpoints() looks for.

set -uo pipefail
mkdir -p /scratch/euh7ys/logs

export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
source /home/euh7ys/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/euh7ys/envs/nla

export HF_HOME=/scratch/euh7ys/.cache/huggingface
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN not set — needed for gated Gemma/Llama repos}
CKPT_ROOT=/scratch/euh7ys/nla_ckpts

python - "$CKPT_ROOT" <<'EOF'
import sys, time
from pathlib import Path
from huggingface_hub import snapshot_download

ckpt_root = Path(sys.argv[1])

# (repo_id, local_dir or None for the default HF cache)
JOBS = [
    ("kitft/nla-gemma3-27b-L41-av",      ckpt_root / "gemma3-27b-L41-av"),
    ("kitft/nla-gemma3-27b-L41-ar",      ckpt_root / "gemma3-27b-L41-ar"),
    ("kitft/Llama-3.3-70B-NLA-L53-av",   ckpt_root / "Llama-3.3-70B-NLA-L53-av"),
    ("kitft/Llama-3.3-70B-NLA-L53-ar",   ckpt_root / "Llama-3.3-70B-NLA-L53-ar"),
    ("google/gemma-3-12b-it",            None),   # base models go to HF_HOME cache,
    ("google/gemma-3-27b-it",            None),   # which is where from_pretrained
    ("meta-llama/Llama-3.3-70B-Instruct", None),  # will look for them
]

results = []
for repo, local_dir in JOBS:
    t0 = time.time()
    print(f"\n══ {repo}  →  {local_dir or '(HF cache)'}", flush=True)
    try:
        path = snapshot_download(
            repo_id=repo,
            local_dir=str(local_dir) if local_dir else None,
            max_workers=8,
        )
        results.append((repo, "ok", f"{(time.time() - t0) / 60:.0f} min", path))
    except Exception as e:  # noqa: BLE001 — keep going, report at the end
        results.append((repo, "FAILED", type(e).__name__, str(e)[:200]))
        print(f"   ✗ {type(e).__name__}: {str(e)[:300]}", flush=True)

print("\n" + "═" * 72)
print("DOWNLOAD SUMMARY")
print("═" * 72)
for repo, status, info, detail in results:
    print(f"  {status:<7} {repo:<40} {info}")
    if status == "FAILED":
        print(f"          {detail}")
print("═" * 72)
EOF
