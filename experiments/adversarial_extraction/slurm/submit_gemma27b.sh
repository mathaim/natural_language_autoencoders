#!/bin/bash
#SBATCH --job-name=nla_adv_gemma27b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=a100_80gb
#SBATCH --mem=128G
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/euh7ys/logs/nla_adv_gemma27b_%j.out
#SBATCH --error=/scratch/euh7ys/logs/nla_adv_gemma27b_%j.err
#
# Gemma-3-27B bf16 is ~54 GB of weights: fits one 80 GB A100 (extraction and
# the SGLang server run one after the other, never together) but NOT a 40 GB
# one, hence the a100_80gb constraint. If sbatch rejects the constraint name,
# check what the cluster calls it:  sinfo -p gpu -o "%N %G %f"
#
# Prerequisite: download_ckpts.sh has landed
#   /scratch/euh7ys/nla_ckpts/gemma3-27b-L41-av   (and the base model in HF cache)

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
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN not set — needed for Gemma}
REPO=/scratch/euh7ys/NaturalLanguageAutoencoders
CKPT_ROOT=/scratch/euh7ys/nla_ckpts
OUT_ROOT=/scratch/euh7ys/adversarial_runs

cd $REPO

echo "=== GPU / toolchain ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "gcc : $(gcc --version | head -1)"
echo "nvcc: $(/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1 || echo 'not found at /usr/local/cuda')"
echo "======================="

[ -d $CKPT_ROOT/gemma3-27b-L41-av ] || { echo "AV checkpoint missing — run download_ckpts.sh first"; exit 1; }

# --sglang-mem-fraction 0.80: 54 GB weights + KV cache for 16 concurrent
# 200-token requests comfortably inside 64 GB; leaves headroom for the
# triton JIT kernels.
#
# --dtype bfloat16: the 27B AV checkpoint is stored fp32 (101 GB, no
# torch_dtype in config.json). SGLang's default `auto` maps fp32 → fp16,
# which overflows in Gemma. Loading fp32 outright would need 108 GB.
python experiments/adversarial_extraction/run_all.py \
    --models gemma27b-l41 \
    --datasets wmdp harmbench advbench truthfulqa \
    --n-samples 200 \
    --out-root $OUT_ROOT \
    --ckpt-root $CKPT_ROOT \
    --skip-extract \
    --device cuda \
    --decode-tokens 64 \
    --sglang-port 30002 \
    --sglang-tp 1 \
    --sglang-mem-fraction 0.80 \
    --sglang-extra-args="--attention-backend triton --dtype bfloat16" \
    --temperature 0.0 \
    --max-new-tokens 200 \
    --workers 16

echo "=== DONE ==="
