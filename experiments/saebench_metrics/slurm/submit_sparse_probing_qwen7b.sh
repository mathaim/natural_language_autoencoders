#!/bin/bash
#SBATCH --job-name=nla_sparse_probe
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/scratch/euh7ys/logs/nla_sparse_probe_%j.out
#SBATCH --error=/scratch/euh7ys/logs/nla_sparse_probe_%j.err

# SAEBench sparse probing for the Qwen2.5-7B NLA at every layer.
#
#   prepare (CPU)  → examples.jsonl          class-balanced SAEBench examples
#   extract (GPU)  → activations.npy         last-token activation, all layers
#   roundtrip      → <arm>/verbalizations.jsonl, reconstructions.npy   (SGLang + AR)
#   probe (CPU)    → <arm>/sparse_probing.json, sparse_probing_by_layer.csv
#
# Every step is skipped or resumed if its output exists, so resubmitting
# after a timeout continues where it stopped.
#
# FIRST RUN — a small smoke test of the whole path (about 2,000 generations):
#   TAG=smoke TRAIN_PER_CLASS=10 TEST_PER_CLASS=6 LAYERS="4 12 20 26" \
#       sbatch experiments/saebench_metrics/slurm/submit_sparse_probing_qwen7b.sh
#
# FULL RUN (2,400 examples x 28 layers x 2 arms = 134k generations):
#   sbatch experiments/saebench_metrics/slurm/submit_sparse_probing_qwen7b.sh
#
# Env vars: TAG, ARMS, TRAIN_PER_CLASS, TEST_PER_CLASS, DATASETS, LAYERS,
#           WORKERS, BASE_MODEL, OUT_ROOT, PORT.

set -euo pipefail

mkdir -p /scratch/euh7ys/logs

# ── Environment (same as experiments/cross_layer_transfer/slurm) ────────────
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
OUT_ROOT=${OUT_ROOT:-/scratch/euh7ys/saebench_nla}
TAG=${TAG:-sp_v1}
ARMS=${ARMS:-"c_direct none"}             # adapter runs; "none" = zero-shot
TRAIN_PER_CLASS=${TRAIN_PER_CLASS:-100}
TEST_PER_CLASS=${TEST_PER_CLASS:-50}
DATASETS=${DATASETS:-}                    # empty = sparse_probing.py default (4 datasets)
LAYERS=${LAYERS:-}                        # empty = all 28
WORKERS=${WORKERS:-32}
BASE_MODEL=${BASE_MODEL:-Qwen/Qwen2.5-7B-Instruct}
PORT=${PORT:-30000}
XLAYER=experiments/cross_layer_transfer/run_multilayer.py
SP=experiments/saebench_metrics/sparse_probing.py
OUT=$OUT_ROOT/$TAG

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
mkdir -p "$OUT"

echo "=== GPU / toolchain ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo "AV   : $AV_CKPT"
echo "AR   : $AR_CKPT"
echo "OUT  : $OUT"
echo "ARMS : $ARMS    LAYERS: ${LAYERS:-all}"
echo "======================="

# ── 1. Examples. Sizes are fixed by the first run for this TAG. ─────────────
if [[ ! -s "$OUT/examples.jsonl" ]]; then
    python $SP prepare --out "$OUT" \
        --train-per-class "$TRAIN_PER_CLASS" --test-per-class "$TEST_PER_CLASS" \
        ${DATASETS:+--datasets $DATASETS}
fi

# ── 2. Activations at every layer. Runs before SGLang takes GPU memory. ─────
if [[ ! -s "$OUT/activations.npy" ]]; then
    python $XLAYER extract --base-model "$BASE_MODEL" \
        --prompts "$OUT/examples.jsonl" --out "$OUT" \
        --positions last --max-length 128 --device cuda
fi

# ── 3. SGLang serving the layer-20 AV. Fail fast if it dies during startup. ─
SGLANG_LOG=$OUT/sglang_${SLURM_JOB_ID:-local}.log
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

# ── 4. Per arm: verbalize + reconstruct at every layer, then probe. ─────────
for ARM in $ARMS; do
    echo "=== arm: $ARM ==="
    python $XLAYER roundtrip \
        --acts "$OUT/activations.npy" --out "$OUT/$ARM" \
        --adapter "$ARM" --ckpt-root "$ADAPTER_ROOT" \
        --av-checkpoint "$AV_CKPT" --ar-checkpoint "$AR_CKPT" \
        --sglang-url "http://localhost:$PORT" \
        ${LAYERS:+--layers $LAYERS} \
        --workers "$WORKERS" --temperature 0.0 --max-new-tokens 200 \
        --device cuda
    python $SP probe --examples "$OUT/examples.jsonl" \
        --acts "$OUT/activations.npy" --run "$OUT/$ARM"
done

echo "=== DONE — results in $OUT/<arm>/sparse_probing_by_layer.csv ==="
