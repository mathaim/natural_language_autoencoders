#!/bin/bash
# One-shot environment setup for the adversarial extraction pipeline on UVA HPC.
# Run from a login node:  bash experiments/adversarial_extraction/slurm/setup_env.sh
#
# Follows the repo README "Inference only" recipe: let sglang[all] pin torch and
# transformers to its tested versions. Everything lives in /scratch because
# /home is over quota.
set -euo pipefail

ENV_PREFIX=/scratch/euh7ys/envs/nla

# ~/.local/lib/python3.11/site-packages has broken leftover packages that
# shadow every conda env. Disable user-site for this script AND in the SLURM
# scripts that use this env.
export PYTHONNOUSERSITE=1
export UV_CACHE_DIR=/scratch/euh7ys/.cache/uv
export UV_LINK_MODE=copy          # cache and env are on the same FS, but be explicit
export PIP_CACHE_DIR=/scratch/euh7ys/.cache/pip

source /home/euh7ys/miniconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true

echo "══ 1/5  removing any previous env at $ENV_PREFIX"
conda env remove --prefix "$ENV_PREFIX" -y 2>/dev/null || true
rm -rf "$ENV_PREFIX"

echo "══ 2/5  creating fresh python 3.11 env"
conda create --prefix "$ENV_PREFIX" python=3.11 -y
conda activate "$ENV_PREFIX"
python -c "import sys; print('python:', sys.executable)"
python -m pip install -q uv

echo "══ 3/5  installing sglang[all] (pins torch + transformers)"
uv pip install "sglang[all]>=0.5.6"
python -c "import torch, transformers, sglang; print('torch', torch.__version__, '| transformers', transformers.__version__, '| sglang', sglang.__version__)"

echo "══ 4/5  installing remaining runtime deps"
uv pip install datasets requests pyarrow safetensors httpx orjson pyyaml numpy
python -c "import datasets, pyarrow, safetensors, httpx, orjson, yaml, numpy, requests; print('runtime deps ok')"

echo "══ 5/5  checking repo imports"
cd /scratch/euh7ys/NaturalLanguageAutoencoders
python -c "import nla_inference; print('nla_inference ok')"
python -c "
import sys; sys.path.insert(0, 'experiments/adversarial_extraction')
import adv_datasets
from datasets import load_dataset   # must be the HuggingFace lib, not ours
from nla_model_configs import NLA_MODELS, resolve_checkpoints
cfg = resolve_checkpoints(NLA_MODELS['qwen7b-l20'], '/scratch/euh7ys/nla_ckpts')
print('AV ->', cfg.av_checkpoint)
print('AR ->', cfg.ar_checkpoint)
"

echo
echo "✓ env ready: conda activate $ENV_PREFIX"
echo "  (CUDA can only be verified on a GPU node — the SLURM script does that)"
