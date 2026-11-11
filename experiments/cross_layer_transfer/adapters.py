"""Bidirectional layer adapters that extend one anchor-layer NLA to every layer.

An NLA pair (AV verbalizer + AR reconstructor) is trained at ONE extraction
layer K (the "anchor"; K=20 for Qwen2.5-7B). These adapters let that single
pair serve all layers:

    h_l  ──input_adapter[l]──▶  anchor space  ──AV──▶  text
    text ──AR──▶  anchor space  ──output_adapter[l]──▶  ĥ_l

The module definitions are ported from chengjiali/nla-train
(`nla_train/multilayer/adapters.py`, Apache-2.0) and MUST stay
state-dict-compatible with the checkpoints in `jialicheng/nla-adapter`.
Do not rename submodules or change the forward math.

Three methods, all residual MLPs (identity at init):
    a / b : one `ResidualAdapter` per non-anchor layer (b = a + a second
            "direct" training stage through the frozen AV/AR)
    c     : one `SharedAdapter` for all layers, FiLM-conditioned on a learned
            layer embedding

Conventions the checkpoints were trained under (do not deviate silently):
  - Layer index l = raw output of decoder block l, captured with a forward
    hook. Same convention as `nla.datagen.extractors` in this repo. This
    equals HF `hidden_states[l + 1]` EXCEPT at the last block, where HF
    returns the post-final-LayerNorm state instead.
  - Inputs are RAW activations. No normalization before the adapter; the
    AV client rescales to `injection_scale` after `encode`.
  - Training loss is cosine only, so the NORM of an adapter output carries
    no information. `decode` output must be rescaled before it is patched
    back into a residual stream (see `rescale_to_norm`).
  - The anchor layer bypasses the adapters entirely, which preserves the
    pretrained NLA's behaviour at its own layer.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

# ─── Released checkpoints ───────────────────────────────────────────────────

ADAPTER_REPO = "jialicheng/nla-adapter"
ADAPTER_SUBDIR = "multilayer_checkpoints/10k"

# Final training step of each of the five released runs. Trained on a 10k
# document corpus against kitft/nla-qwen2.5-7b-L20-{av,ar}.
FINAL_CHECKPOINTS: dict[str, str] = {
    "a_align": "a_align_r256_step4000.pt",
    "b_align": "b_align_r512_step4000.pt",
    "b_direct": "b_direct_r512_step500.pt",
    "c_align": "c_align_r1024_step4000.pt",
    "c_direct": "c_direct_r1024_step500.pt",
}
# Best mean cosine and recall across layers in the upstream evaluation, and
# the smallest file (59 MB). See README.md for the per-layer table.
DEFAULT_RUN = "c_direct"


# ─── Modules (state-dict-compatible port) ───────────────────────────────────

class ResidualAdapter(nn.Module):
    """Residual MLP: d -> rank -> d; identity at initialization."""

    def __init__(self, d: int, rank: int):
        super().__init__()
        self.down = nn.Linear(d, rank)
        self.up = nn.Linear(rank, d)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.up(F.silu(self.down(x)))


class SharedAdapter(nn.Module):
    """Layer-conditioned residual MLP with learned FiLM modulation."""

    def __init__(self, d: int, rank: int, n_layers: int, embedding_dim: int):
        super().__init__()
        self.down = nn.Linear(d, rank)
        self.up = nn.Linear(rank, d)
        self.layer_embed = nn.Embedding(n_layers, embedding_dim)
        self.film = nn.Linear(embedding_dim, 2 * rank)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    def forward(self, x: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.film(self.layer_embed(layers)).chunk(2, dim=-1)
        u = F.silu(self.down(x))
        return x + self.up(u * (1.0 + gamma) + beta)


class MultiLayerAdapters(nn.Module):
    """a/b: per-layer adapters; c: one layer-conditioned adapter pair.

    `encode(h, layers)` maps layer-l activations into anchor space (before
    the AV). `decode(h, layers)` maps anchor-space vectors (AR outputs) back
    to layer l. Both take a MIXED batch: `h` is [N, d] and `layers` is [N]
    long, one layer index per row, so all layers run in one call.
    """

    def __init__(self, method: str, d: int = 3584, n_layers: int = 28,
                 anchor: int = 20, rank: int = 256, embedding_dim: int = 32):
        super().__init__()
        if method not in {"a", "b", "c"}:
            raise ValueError("method must be a, b, or c")
        if not (0 <= anchor < n_layers):
            raise ValueError("invalid anchor")
        self.config = dict(method=method, d=d, n_layers=n_layers,
                           anchor=anchor, rank=rank, embedding_dim=embedding_dim)
        self.method, self.d, self.n_layers, self.anchor = method, d, n_layers, anchor
        if method == "c":
            self.input_adapter = SharedAdapter(d, rank, n_layers, embedding_dim)
            self.output_adapter = SharedAdapter(d, rank, n_layers, embedding_dim)
        else:
            self.input_adapter = nn.ModuleDict(
                {str(i): ResidualAdapter(d, rank) for i in range(n_layers) if i != anchor})
            self.output_adapter = nn.ModuleDict(
                {str(i): ResidualAdapter(d, rank) for i in range(n_layers) if i != anchor})

    def encode(self, h: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
        return self._apply_adapters(h, layers, self.input_adapter)

    def decode(self, h: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
        return self._apply_adapters(h, layers, self.output_adapter)

    def _apply_adapters(self, x: torch.Tensor, layers: torch.Tensor,
                        module: nn.Module) -> torch.Tensor:
        if x.ndim != 2 or x.shape[1] != self.d:
            raise ValueError(f"Expected [batch,{self.d}], got {tuple(x.shape)}")
        if layers.ndim != 1 or layers.shape[0] != x.shape[0]:
            raise ValueError("Expected one layer index per activation")
        if bool(((layers < 0) | (layers >= self.n_layers)).any()):
            raise ValueError("Layer index out of range")
        keep = layers != self.anchor
        if not bool(keep.any()):
            return x
        result = x.clone()
        if self.method == "c":
            idx = keep.nonzero(as_tuple=True)[0]
            result = result.index_copy(
                0, idx, module(x.index_select(0, idx), layers.index_select(0, idx)))
        else:
            for layer in range(self.n_layers):
                if layer == self.anchor:
                    continue
                idx = (layers == layer).nonzero(as_tuple=True)[0]
                if idx.numel():
                    result = result.index_copy(
                        0, idx, module[str(layer)](x.index_select(0, idx)))
        return result


# ─── Loading ────────────────────────────────────────────────────────────────

def resolve_checkpoint(name_or_path: str, ckpt_root: str | Path | None = None,
                       repo: str = ADAPTER_REPO) -> Path:
    """Run name (`c_direct`), file name, or path → local .pt path.

    Lookup order: an existing path as given; `{ckpt_root}/{file}` or
    `{ckpt_root}/{ADAPTER_SUBDIR}/{file}` (the layout `run_multilayer.py
    fetch` writes); the HF Hub cache (downloads on first use). Pass
    `ckpt_root` on compute nodes without network access.
    """
    p = Path(name_or_path)
    if p.is_file():
        return p
    filename = FINAL_CHECKPOINTS.get(name_or_path, name_or_path)
    if not filename.endswith(".pt"):
        raise ValueError(
            f"unknown adapter {name_or_path!r}. Use a path to a .pt file, a file "
            f"name from {repo}, or one of {sorted(FINAL_CHECKPOINTS)}.")
    if ckpt_root is not None:
        for local in (Path(ckpt_root) / filename,
                      Path(ckpt_root) / ADAPTER_SUBDIR / filename):
            if local.is_file():
                return local
    from huggingface_hub import hf_hub_download
    return Path(hf_hub_download(repo_id=repo, filename=f"{ADAPTER_SUBDIR}/{filename}"))


def load_adapters(path: str | Path, device: str = "cpu",
                  ) -> tuple[MultiLayerAdapters, dict[str, Any]]:
    """Load a released checkpoint. Returns (frozen eval-mode adapters, provenance).

    `weights_only=True`: the checkpoint is plain tensors plus a small config
    dict, so no pickled code is ever executed. `strict=True`: a key mismatch
    means this port has drifted from the training code — fail loud.
    """
    ckpt = torch.load(str(path), map_location="cpu", weights_only=True)
    adapters = MultiLayerAdapters(**ckpt["config"])
    adapters.load_state_dict(ckpt["state_dict"], strict=True)
    adapters.to(device).eval()
    for p in adapters.parameters():
        p.requires_grad_(False)
    provenance = {k: ckpt.get(k) for k in ("stage", "step", "seed", "corpus")}
    provenance["path"] = str(path)
    return adapters, provenance


def assert_matches_sidecar(adapters: MultiLayerAdapters, sidecar: dict[str, Any],
                           what: str) -> None:
    """The NLA sidecar is the contract: anchor layer and width must agree.

    An adapter trained for anchor K only makes sense in front of an NLA whose
    extraction layer is K. A mismatch does not crash — it silently verbalizes
    the wrong space — so check before the first request.
    """
    d = sidecar["d_model"]
    assert adapters.d == d, (
        f"adapter d={adapters.d} != {what} sidecar d_model={d}. "
        f"Wrong adapter checkpoint for this NLA.")
    k = sidecar.get("extraction_layer_index",
                    (sidecar.get("critic") or {}).get("extraction_layer_index"))
    assert k is not None, f"{what} sidecar has no extraction_layer_index"
    assert adapters.anchor == k, (
        f"adapter anchor={adapters.anchor} != {what} sidecar "
        f"extraction_layer_index={k}. These adapters map into layer "
        f"{adapters.anchor}'s space; this NLA reads layer {k}.")


# ─── Norm handling ──────────────────────────────────────────────────────────

def rescale_to_norm(v: torch.Tensor, target_norm: torch.Tensor | float) -> torch.Tensor:
    """Rescale each row of v to the given L2 norm(s). Norm in fp32.

    Adapters and the AR are trained on direction only. Before a decoded
    vector replaces a real activation, give it that activation's norm (or the
    layer's typical norm); otherwise downstream blocks see an off-scale input.
    """
    norm = v.float().norm(dim=-1, keepdim=True).clamp_min(1e-12)
    target = torch.as_tensor(target_norm, dtype=torch.float32, device=v.device)
    if target.ndim == 1:
        target = target.unsqueeze(-1)
    return (v.float() / norm * target).to(v.dtype)
