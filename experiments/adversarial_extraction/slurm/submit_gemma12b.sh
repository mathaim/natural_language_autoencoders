#!/bin/bash
#SBATCH --job-name=nla_adv_gemma12b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/euh7ys/logs/nla_adv_gemma12b_%j.out
#SBATCH --error=/scratch/euh7ys/logs/nla_adv_gemma12b_%j.err

set -euo pipefail

mkdir -p /scratch/euh7ys/logs

# ── Environment ──────────────────────────────────────────────────────────────
source /home/euh7ys/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_HOME=/scratch/euh7ys/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/euh7ys/.cache/huggingface
# Gemma is a gated repo — HF_TOKEN must be set
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN not set — needed for Gemma}
REPO=/scratch/euh7ys/NaturalLanguageAutoencoders
CKPT_ROOT=/scratch/euh7ys/nla_ckpts
OUT_ROOT=/scratch/euh7ys/adversarial_runs

cd $REPO

echo "=== GPU info ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "================"

python experiments/adversarial_extraction/run_all.py \
    --models gemma12b-l32 \
    --datasets wmdp harmbench advbench truthfulqa \
    --n-samples 200 \
    --out-root $OUT_ROOT \
    --ckpt-root $CKPT_ROOT \
    --device cuda \
    --decode-tokens 64 \
    --sglang-port 30001 \
    --sglang-tp 1 \
    --sglang-mem-fraction 0.45 \
    --temperature 0.0 \
    --max-new-tokens 200

echo "=== DONE ==="
