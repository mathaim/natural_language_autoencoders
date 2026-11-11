#!/bin/bash
#SBATCH --job-name=nla_xlayer_qwen7b
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/euh7ys/logs/nla_xlayer_qwen7b_%j.out
#SBATCH --error=/scratch/euh7ys/logs/nla_xlayer_qwen7b_%j.err

# All-layer NLA smoke test for Qwen2.5-7B: checks the adapter port on CPU,
# then runs the AV → AR round trip at all 28 layers, with the adapters and
# zero-shot, on the first N documents of the 2k eval corpus.
#
# Before the first submit, from a login node (needs network):
#   python experiments/cross_layer_transfer/run_multilayer.py fetch \
#       --out /scratch/euh7ys/nla_ckpts/nla-adapter --with-eval-data
#
# Idempotent: verbalizations resume, so resubmitting after a timeout continues.
# Override defaults with env vars:  ADAPTER=b_direct N_DOCS=200 sbatch submit_qwen7b.sh

set -euo pipefail

mkdir -p /scratch/euh7ys/logs

# ── Environment (same as experiments/adversarial_extraction/slurm) ──────────
export PYTHONNOUSERSITE=1   # ~/.local has broken packages that shadow the env
export PYTHONUNBUFFERED=1   # stream progress to the SLURM log
source /home/euh7ys/miniconda3/etc/profile.d/conda.sh
conda activate /scratch/euh7ys/envs/nla

# gcc 11 for sglang's JIT kernels (system gcc 8.5 lacks the C++20 <version> header).
module load gcc/11.4.0
export CC=$(which gcc) CXX=$(which g++)

export HF_HOME=/scratch/euh7ys/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/euh7ys/.cache/huggingface
export XDG_CACHE_HOME=/scratch/euh7ys/.cache
export TRITON_CACHE_DIR=/scratch/euh7ys/.cache/triton
export TORCH_EXTENSIONS_DIR=/scratch/euh7ys/.cache/torch_extensions

REPO=/scratch/euh7ys/NaturalLanguageAutoencoders
CKPT_ROOT=/scratch/euh7ys/nla_ckpts
ADAPTER_ROOT=$CKPT_ROOT/nla-adapter
OUT_ROOT=${OUT_ROOT:-/scratch/euh7ys/cross_layer_runs}
ADAPTER=${ADAPTER:-c_direct}
N_DOCS=${N_DOCS:-50}          # x 28 layers verbalizations per arm
PORT=${PORT:-30000}
CLI=experiments/cross_layer_transfer/run_multilayer.py

# Local checkpoint dir if present, else the HF id (downloads into HF_HOME).
resolve() {
    for d in "$CKPT_ROOT/$1" "$CKPT_ROOT/nla-$1"; do
        if [[ -d "$d" ]]; then echo "$d"; return; fi
    done
    echo "kitft/nla-$1"
}
AV_CKPT=$(resolve qwen2.5-7b-L20-av)
AR_CKPT=$(resolve qwen2.5-7b-L20-ar)

cd $REPO

echo "=== GPU / toolchain ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "gcc : $(gcc --version | head -1)"
echo "AV  : $AV_CKPT"
echo "AR  : $AR_CKPT"
echo "======================="

if [[ ! -s "$ADAPTER_ROOT/data/activations.npy" ]]; then
    echo "[setup] adapter checkpoints / eval corpus missing — fetching"
    python $CLI fetch --out "$ADAPTER_ROOT" --with-eval-data
fi

# ── 1. CPU check of the adapter port. Fails the job before any GPU time. ────
python $CLI check --ckpt-root "$ADAPTER_ROOT"

# ── 2. SGLang serving the layer-20 AV. Fail fast if it dies during startup. ─
SGLANG_LOG=$OUT_ROOT/sglang_${SLURM_JOB_ID:-local}.log
mkdir -p "$OUT_ROOT"
python -m sglang.launch_server \
    --model-path "$AV_CKPT" --port "$PORT" \
    --disable-radix-cache --trust-remote-code \
    --mem-fraction-static 0.45 > "$SGLANG_LOG" 2>&1 &
SGLANG_PID=$!
trap 'kill $SGLANG_PID 2>/dev/null || true' EXIT

echo "[sglang] pid $SGLANG_PID, log → $SGLANG_LOG"
for _ in $(seq 120); do          # 120 x 5s = 10 min: model load + first JIT compile
    if grep -q "The server is fired up and ready to roll" "$SGLANG_LOG" 2>/dev/null; then
        echo "[sglang] ready"; break
    fi
    if ! kill -0 $SGLANG_PID 2>/dev/null; then
        echo "[sglang] ✗ server exited before becoming ready:"; tail -30 "$SGLANG_LOG"; exit 1
    fi
    sleep 5
done
grep -q "The server is fired up and ready to roll" "$SGLANG_LOG" || {
    echo "[sglang] ✗ not ready after 10 min:"; tail -30 "$SGLANG_LOG"; exit 1; }

# ── 3. Round trip at all 28 layers: adapters, then zero-shot baseline. ──────
for ARM in "$ADAPTER" none; do
    python $CLI roundtrip \
        --acts "$ADAPTER_ROOT/data/activations.npy" \
        --out "$OUT_ROOT/eval2k_n${N_DOCS}_${ARM}" \
        --adapter "$ARM" --ckpt-root "$ADAPTER_ROOT" \
        --av-checkpoint "$AV_CKPT" --ar-checkpoint "$AR_CKPT" \
        --sglang-url "http://localhost:$PORT" \
        --n "$N_DOCS" --workers 16 --temperature 0.0 --max-new-tokens 200 \
        --device cuda
done

echo "=== DONE — compare $OUT_ROOT/eval2k_n${N_DOCS}_{${ADAPTER},none}/summary.json ==="
