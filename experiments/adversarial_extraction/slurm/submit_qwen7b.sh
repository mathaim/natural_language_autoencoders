#!/bin/bash
#SBATCH --job-name=nla_adv_qwen7b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/euh7ys/logs/nla_adv_qwen7b_%j.out
#SBATCH --error=/scratch/euh7ys/logs/nla_adv_qwen7b_%j.err

set -euo pipefail

mkdir -p /scratch/euh7ys/logs

# ── Environment ──────────────────────────────────────────────────────────────
export PYTHONNOUSERSITE=1   # ~/.local has broken packages that shadow the env
export PYTHONUNBUFFERED=1   # SLURM block-buffers stdout to file; stream it live (inherited by subprocesses)
source /home/euh7ys/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/euh7ys/envs/nla

# System gcc is 8.5 and lacks the C++20 <version> header that sglang's JIT
# kernels include. gcc 11 has it and is accepted as a host compiler by every
# CUDA 12.x nvcc (14/16 can be rejected). nvcc picks g++ from PATH; CC/CXX
# cover any build tool in the chain that reads them instead.
module load gcc/11.4.0
export CC=$(which gcc) CXX=$(which g++)

export HF_HOME=/scratch/euh7ys/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/euh7ys/.cache/huggingface
# Keep every JIT / compile cache off the over-quota /home filesystem.
export XDG_CACHE_HOME=/scratch/euh7ys/.cache        # tvm-ffi (sglang JIT kernels) and friends
export TRITON_CACHE_DIR=/scratch/euh7ys/.cache/triton
export TORCH_EXTENSIONS_DIR=/scratch/euh7ys/.cache/torch_extensions
REPO=/scratch/euh7ys/NaturalLanguageAutoencoders
CKPT_ROOT=/scratch/euh7ys/nla_ckpts
OUT_ROOT=/scratch/euh7ys/adversarial_runs

cd $REPO

echo "=== GPU / toolchain ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "gcc : $(gcc --version | head -1)"
echo "nvcc: $(/usr/local/cuda/bin/nvcc --version 2>/dev/null | tail -1 || echo 'not found at /usr/local/cuda')"
echo "======================="

python experiments/adversarial_extraction/run_all.py \
    --models qwen7b-l20 \
    --datasets wmdp harmbench advbench truthfulqa \
    --n-samples 200 \
    --out-root $OUT_ROOT \
    --ckpt-root $CKPT_ROOT \
    --skip-extract \
    --device cuda \
    --decode-tokens 64 \
    --sglang-port 30000 \
    --sglang-tp 1 \
    --sglang-mem-fraction 0.45 \
    --temperature 0.0 \
    --max-new-tokens 200 \
    --workers 16

echo "=== DONE ==="
