"""Registry of all released NLA model families.

Source: kitft/nla-models collection on the HF Hub.
Each entry carries everything needed to run extract + verbalize for one
base-model family.

If checkpoints are already downloaded locally, pass --ckpt-root to run_all.py
and call resolve_checkpoints(cfg, ckpt_root) to remap HF IDs to local paths.
Expected layout under ckpt_root:
    {ckpt_root}/qwen2.5-7b-L20-av/
    {ckpt_root}/qwen2.5-7b-L20-ar/
    {ckpt_root}/gemma3-12b-L32-av/
    ...  (directory names match the HF repo slug after the owner prefix)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NLAModelConfig:
    name: str              # short slug used for the output directory
    base_model: str        # HF model ID for the base (instruct) model
    layer: int             # extraction layer K
    n_layers: int          # total transformer layers (for reference / docs)
    d_model: int
    av_checkpoint: str     # HF model ID for the AV (actor)
    ar_checkpoint: str     # HF model ID for the AR (critic)
    # Some models need more GPU memory — caller can use this to set
    # device_map or tensor_parallel_size in SGLang.
    approx_params_b: float  # billions of parameters (rough, for scheduling)
    # Gemma models are gated repos — HF_TOKEN must be set in the environment.
    requires_hf_token: bool = False


def resolve_checkpoints(cfg: "NLAModelConfig", ckpt_root: str | Path) -> "NLAModelConfig":
    """Return a copy of cfg with av/ar checkpoint paths remapped to local dirs.

    Strips the HF owner prefix (e.g. "kitft/") and looks for a matching
    subdirectory under ckpt_root. Falls back to the original HF ID if the
    local directory doesn't exist (so HF download still works as a fallback).
    """
    root = Path(ckpt_root)

    def _resolve(hf_id: str) -> str:
        slug = hf_id.split("/")[-1]           # "nla-qwen2.5-7b-L20-av"
        # also try without the "nla-" prefix — matches /scratch/.../qwen2.5-7b-L20-av
        candidates = [root / slug, root / slug.removeprefix("nla-")]
        for p in candidates:
            if p.is_dir():
                return str(p)
        return hf_id  # fall back to HF hub

    import dataclasses
    return dataclasses.replace(
        cfg,
        av_checkpoint=_resolve(cfg.av_checkpoint),
        ar_checkpoint=_resolve(cfg.ar_checkpoint),
    )


NLA_MODELS: dict[str, NLAModelConfig] = {
    "qwen7b-l20": NLAModelConfig(
        name="qwen7b-l20",
        base_model="Qwen/Qwen2.5-7B-Instruct",
        layer=20,
        n_layers=28,
        d_model=3584,
        av_checkpoint="kitft/nla-qwen2.5-7b-L20-av",
        ar_checkpoint="kitft/nla-qwen2.5-7b-L20-ar",
        approx_params_b=7.0,
    ),
    "gemma12b-l32": NLAModelConfig(
        name="gemma12b-l32",
        base_model="google/gemma-3-12b-it",
        layer=32,
        n_layers=48,
        d_model=3840,
        av_checkpoint="kitft/nla-gemma3-12b-L32-av",
        ar_checkpoint="kitft/nla-gemma3-12b-L32-ar",
        approx_params_b=12.0,
        requires_hf_token=True,
    ),
    "gemma27b-l41": NLAModelConfig(
        name="gemma27b-l41",
        base_model="google/gemma-3-27b-it",
        layer=41,
        n_layers=62,
        d_model=5376,
        av_checkpoint="kitft/nla-gemma3-27b-L41-av",
        ar_checkpoint="kitft/nla-gemma3-27b-L41-ar",
        approx_params_b=27.0,
        requires_hf_token=True,
    ),
    "llama70b-l53": NLAModelConfig(
        name="llama70b-l53",
        base_model="meta-llama/Llama-3.3-70B-Instruct",
        layer=53,
        n_layers=80,
        d_model=8192,
        av_checkpoint="kitft/Llama-3.3-70B-NLA-L53-av",
        ar_checkpoint="kitft/Llama-3.3-70B-NLA-L53-ar",
        approx_params_b=70.0,
    ),
}
