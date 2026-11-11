"""All-layer NLA: one anchor-layer AV/AR pair plus layer adapters.

    mnla = MultiLayerNLA.from_checkpoints(
        "c_direct",
        av_checkpoint="kitft/nla-qwen2.5-7b-L20-av",
        ar_checkpoint="kitft/nla-qwen2.5-7b-L20-ar",
        sglang_url="http://localhost:30000", critic_device="cuda",
    )
    acts = extract_all_layers(model, input_ids)        # [n_layers, T, d]
    h = acts[:, -1]                                    # last token, every layer
    layers = torch.arange(h.shape[0])
    texts = mnla.verbalize(h, layers)                  # 28 explanations, one call
    h_hat = mnla.reconstruct(texts, layers, target_norms=h.norm(dim=-1))

Every method takes a MIXED batch — `acts` is [N, d] and `layers` is [N], one
layer index per row — so "all layers at once" is just a batch whose rows come
from different layers. There is exactly one AV server and one AR in memory
regardless of how many layers are in play.

`adapters=None` gives the zero-shot baseline: the anchor-layer NLA applied
directly to other layers' activations.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).parent))

from adapters import (  # noqa: E402
    MultiLayerAdapters, assert_matches_sidecar, load_adapters,
    rescale_to_norm, resolve_checkpoint,
)

ArrayLike = np.ndarray | torch.Tensor
LayerSpec = int | Sequence[int] | np.ndarray | torch.Tensor


def local_checkpoint_dir(id_or_path: str | Path) -> Path:
    """Local dir as-is; HF repo id → snapshot in the HF cache.

    `NLAClient` / `NLACritic` read `nla_meta.yaml` from a directory, so a Hub
    id has to be materialized first.
    """
    p = Path(id_or_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(str(id_or_path)))


@torch.inference_mode()
def extract_all_layers(model: Any, input_ids: torch.Tensor) -> torch.Tensor:
    """One forward pass → raw residual stream at every layer.

    input_ids: [1, T]. Returns [n_layers, T, d] fp32 on CPU, where row l is
    the output of decoder block l. Never normalized — normalization happens
    at injection time, per the repo invariant.

    Forward hooks, NOT `output_hidden_states=True`: HF applies the final
    LayerNorm before appending the last hidden state, so `hidden_states[-1]`
    is not the raw output of the last block. Hooks give the true residual
    stream at every layer, which is what the adapters were trained on.
    """
    from nla.arch_adapters import resolve_decoder_layers

    assert input_ids.ndim == 2 and input_ids.shape[0] == 1, "expected [1, T] input_ids"
    blocks = resolve_decoder_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def make_hook(idx: int):
        def hook(_module: Any, _inp: Any, out: Any) -> None:
            h = out[0] if isinstance(out, tuple) else out
            captured[idx] = h[0].detach().float().cpu()
        return hook

    handles = [blk.register_forward_hook(make_hook(i)) for i, blk in enumerate(blocks)]
    try:
        model(input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    assert len(captured) == len(blocks), (
        f"hooks fired on {len(captured)}/{len(blocks)} decoder blocks")
    return torch.stack([captured[i] for i in range(len(blocks))], dim=0)


class MultiLayerNLA:
    """Verbalize and reconstruct activations from any layer.

    client: `nla_inference.NLAClient` for the anchor-layer AV (needs a running
            SGLang server). Optional — only `verbalize*` needs it.
    critic: `nla_inference.NLACritic` for the anchor-layer AR. Optional —
            only `reconstruct` needs it.
    """

    def __init__(self, adapters: MultiLayerAdapters | None,
                 client: Any | None = None, critic: Any | None = None):
        self.adapters = adapters
        self.client = client
        self.critic = critic
        self.device = (next(adapters.parameters()).device
                       if adapters is not None else torch.device("cpu"))

    @classmethod
    def from_checkpoints(
        cls,
        adapter: str | None,
        av_checkpoint: str | Path | None = None,
        ar_checkpoint: str | Path | None = None,
        *,
        ckpt_root: str | Path | None = None,
        sglang_url: str = "http://localhost:30000",
        adapter_device: str = "cpu",
        critic_device: str = "cuda",
    ) -> "MultiLayerNLA":
        """adapter: run name (`c_direct`), .pt path, or None / "none" for zero-shot.

        Asserts the adapter's anchor layer and width against each NLA sidecar
        before anything runs.
        """
        from nla_inference import NLAClient, NLACritic

        adapters = None
        if adapter is not None and adapter != "none":
            path = resolve_checkpoint(adapter, ckpt_root)
            adapters, prov = load_adapters(path, device=adapter_device)
            print(f"[MultiLayerNLA] adapters {path.name}: method={adapters.method} "
                  f"anchor={adapters.anchor} n_layers={adapters.n_layers} "
                  f"stage={prov['stage']} step={prov['step']}")
        else:
            print("[MultiLayerNLA] no adapters — zero-shot baseline")

        client = critic = None
        if av_checkpoint is not None:
            av_dir = local_checkpoint_dir(av_checkpoint)
            if adapters is not None:
                assert_matches_sidecar(
                    adapters, yaml.safe_load((av_dir / "nla_meta.yaml").read_text()), "AV")
            client = NLAClient(av_dir, sglang_url=sglang_url)
        if ar_checkpoint is not None:
            ar_dir = local_checkpoint_dir(ar_checkpoint)
            if adapters is not None:
                assert_matches_sidecar(
                    adapters, yaml.safe_load((ar_dir / "nla_meta.yaml").read_text()), "AR")
            critic = NLACritic(ar_dir, device=critic_device)
        return cls(adapters, client, critic)

    # ─── Adapter maps ─────────────────────────────────────────────────────

    def _prep(self, acts: ArrayLike, layers: LayerSpec) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.as_tensor(np.asarray(acts, dtype=np.float32)
                            if not torch.is_tensor(acts) else acts).float()
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if isinstance(layers, int):
            lay = torch.full((x.shape[0],), layers, dtype=torch.long)
        else:
            lay = torch.as_tensor(np.asarray(layers) if not torch.is_tensor(layers)
                                  else layers).long().reshape(-1)
        assert lay.shape[0] == x.shape[0], (
            f"{x.shape[0]} activations but {lay.shape[0]} layer indices")
        assert torch.isfinite(x).all(), "activation has NaN/Inf"
        return x.to(self.device), lay.to(self.device)

    @torch.inference_mode()
    def encode(self, acts: ArrayLike, layers: LayerSpec) -> torch.Tensor:
        """Layer-l activations → anchor space. [N, d] fp32 on CPU."""
        x, lay = self._prep(acts, layers)
        if self.adapters is not None:
            x = self.adapters.encode(x, lay)
        return x.float().cpu()

    @torch.inference_mode()
    def decode(self, vecs: ArrayLike, layers: LayerSpec,
               target_norms: ArrayLike | float | None = None) -> torch.Tensor:
        """Anchor-space vectors → layer l. [N, d] fp32 on CPU.

        target_norms: scalar or [N]. The decoded direction is rescaled to
        this L2 norm. Pass the norm of the activation being replaced whenever
        the result goes back into a residual stream — adapter and AR outputs
        are direction-only, their raw norm is not meaningful.
        """
        x, lay = self._prep(vecs, layers)
        if self.adapters is not None:
            x = self.adapters.decode(x, lay)
        x = x.float().cpu()
        if target_norms is not None:
            t = (torch.as_tensor(np.asarray(target_norms, dtype=np.float32))
                 if not torch.is_tensor(target_norms) else target_norms.float().cpu())
            x = rescale_to_norm(x, t)
        return x

    # ─── AV: activations → text ───────────────────────────────────────────

    def verbalize_iter(self, acts: ArrayLike, layers: LayerSpec, *, workers: int = 16,
                       **sampling: object) -> Iterator[tuple[int, str | Exception]]:
        """Yield (row_index, explanation-or-exception) as requests complete.

        Completion order is nondeterministic. A failed request yields its
        exception instead of raising, so a caller writing incrementally can
        skip it and retry on resume.
        """
        assert self.client is not None, "no AV client — pass av_checkpoint"
        encoded = self.encode(acts, layers).numpy()
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {ex.submit(self.client.generate, encoded[i], **sampling): i
                       for i in range(encoded.shape[0])}
            try:
                for fut in as_completed(futures):
                    i = futures[fut]
                    try:
                        yield i, fut.result()
                    except Exception as e:  # noqa: BLE001 — surfaced to caller
                        yield i, e
            finally:
                # Caller stopped early (e.g. server down): drop queued requests.
                ex.shutdown(wait=False, cancel_futures=True)

    def verbalize(self, acts: ArrayLike, layers: LayerSpec, *, workers: int = 16,
                  **sampling: object) -> list[str]:
        """Explanations in input order. Raises on the first failed request."""
        out: dict[int, str] = {}
        for i, res in self.verbalize_iter(acts, layers, workers=workers, **sampling):
            if isinstance(res, Exception):
                raise res
            out[i] = res
        return [out[i] for i in range(len(out))]

    # ─── AR: text → activations ───────────────────────────────────────────

    def reconstruct_anchor(self, texts: Sequence[str]) -> torch.Tensor:
        """Explanations → AR outputs in ANCHOR space, before the output adapter."""
        assert self.critic is not None, "no AR critic — pass ar_checkpoint"
        return torch.stack([self.critic.reconstruct(t) for t in texts]).float()

    def reconstruct(self, texts: Sequence[str], layers: LayerSpec,
                    target_norms: ArrayLike | float | None = None) -> torch.Tensor:
        """Explanations → reconstructed layer-l activations. See `decode`."""
        return self.decode(self.reconstruct_anchor(texts), layers, target_norms)
