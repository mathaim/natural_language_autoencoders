#!/bin/bash
#SBATCH --job-name=nla_adv_llama70b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=a100_80gb
#SBATCH --mem=256G
#SBATCH --time=20:00:00
#SBATCH --output=/scratch/euh7ys/logs/nla_adv_llama70b_%j.out
#SBATCH --error=/scratch/euh7ys/logs/nla_adv_llama70b_%j.err
#
# Llama-3.3-70B bf16 is ~140 GB of weights — four 80 GB A100s on one node.
#   extraction : --device auto  → HF device_map=auto shards the base model
#   verbalize  : --sglang-tp 4  → 35 GB of AV weights per GPU + KV cache
# tp=2 does NOT fit (70 GB/GPU leaves no KV cache at mem-fraction 0.85).
#
# Prerequisite: download_ckpts.sh has landed
#   /scratch/euh7ys/nla_ckpts/Llama-3.3-70B-NLA-L53-av  (and the base model in HF cache)

set -euo pipefail

mkdir -p /scratch/euh7ys/logs

# ── Environment (identical to submit_gemma12b.sh) ───────────────────────────
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
source /home/euh7ys/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/euh7ys/envs/nla

module load gcc/11.4.0
export CC=$(which gcc) CXX=$(which g++)

export HF_HOME=/scratch/euh7ys/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/euh7ys/.cache/huggingface
export XDG_CACHE_HOME=/scratch/euh7ys/.cache
export TRITON_CACHE_DIR=/scratch/euh7ys/.cache/triton
export TORCH_EXTENSIONS_DIR=/scratch/euh7ys/.cache/torch_extensions
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN not set — needed for Llama-3.3 (gated)}
REPO=/scratch/euh7ys/NaturalLanguageAutoencoders
CKPT_ROOT=/scratch/euh7ys/nla_ckpts
OUT_ROOT=/scratch/euh7ys/adversarial_runs

cd $REPO

echo "=== GPU / toolchain ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "gcc : $(gcc --version | head -1)"
echo "nvcc: $(/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1 || echo 'not found at /usr/local/cuda')"
echo "======================="

[ -d $CKPT_ROOT/Llama-3.3-70B-NLA-L53-av ] || { echo "AV checkpoint missing — run download_ckpts.sh first"; exit 1; }
[ "$(nvidia-smi -L | wc -l)" -ge 4 ] || { echo "need 4 GPUs, got $(nvidia-smi -L | wc -l)"; exit 1; }

# Llama uses head_dim=128, so the default attention backend (what the Qwen
# run used on A100) is fine — no --attention-backend override.
# --dtype bfloat16 pinned in case the AV is stored fp32 like the 27B Gemma
# one (fp32 at tp=4 would be 70 GB/GPU — no room for KV cache).
python experiments/adversarial_extraction/run_all.py \
    --models llama70b-l53 \
    --datasets wmdp harmbench advbench truthfulqa \
    --n-samples 200 \
    --out-root $OUT_ROOT \
    --ckpt-root $CKPT_ROOT \
    --skip-extract \
    --device auto \
    --decode-tokens 64 \
    --sglang-port 30003 \
    --sglang-tp 4 \
    --sglang-mem-fraction 0.85 \
    --sglang-extra-args="--dtype bfloat16" \
    --temperature 0.0 \
    --max-new-tokens 200 \
    --workers 16

echo "=== DONE ==="
