# Cross-layer transfer: one NLA at every layer (Qwen2.5-7B)

The released NLA pair for Qwen2.5-7B reads and writes **layer 20 only**. This
directory makes that one pair usable at **all 28 layers**, so an experiment can
verbalize, edit and reconstruct the residual stream at every layer in a single
run.

It does this with small bidirectional adapters trained by Jiali Cheng
([weights](https://huggingface.co/jialicheng/nla-adapter),
[training code](https://github.com/chengjiali/nla-train)):

```
h_l  ──input adapter──▶  layer-20 space  ──AV──▶  explanation
explanation  ──AR──▶  layer-20 space  ──output adapter──▶  ĥ_l
```

Layer 20 itself bypasses the adapters, so the pretrained NLA is unchanged
there. Nothing here touches `nla/` or `miles/`.

## Files

| File | What it is |
|---|---|
| `adapters.py` | Adapter modules (state-dict-compatible port), checkpoint registry, loader, sidecar check |
| `multilayer.py` | `MultiLayerNLA` (verbalize / reconstruct any layer) and `extract_all_layers` |
| `run_multilayer.py` | CLI: `fetch`, `check`, `extract`, `roundtrip` |
| `slurm/submit_qwen7b.sh` | End-to-end smoke test on one A100 |

## Checkpoints

`fetch` downloads the final training step of each of the five released runs
(2.1 GB total). Earlier steps stay on the Hub; pass a file name or path to
`--adapter` to use one.

| Run name | File | Method | Size |
|---|---|---|---|
| `a_align` | `a_align_r256_step4000.pt` | one adapter per layer, rank 256, activation alignment only | 397 MB |
| `b_align` | `b_align_r512_step4000.pt` | one adapter per layer, rank 512, alignment only | 794 MB |
| `b_direct` | `b_direct_r512_step500.pt` | `b_align` plus 500 steps trained through the frozen AV/AR | 794 MB |
| `c_align` | `c_align_r1024_step4000.pt` | one shared adapter, layer-conditioned (FiLM), rank 1024, alignment only | 59 MB |
| `c_direct` | `c_direct_r1024_step500.pt` | `c_align` plus 500 steps through the frozen AV/AR. **Default.** | 59 MB |

All were trained on 10k documents against `kitft/nla-qwen2.5-7b-L20-{av,ar}`.
Upstream's own evaluation (2k held-out documents, round-trip cosine, mean over
28 layers): zero-shot 0.555, `a` 0.670, `b` 0.690, `c` 0.721. `c` is also the
only method that beats zero-shot on retrieval, which is why it is the default.

## Setup

```bash
# login node (needs network); 2.1 GB of adapters + 404 MB eval corpus
python experiments/cross_layer_transfer/run_multilayer.py fetch \
    --out /scratch/$USER/nla_ckpts/nla-adapter --with-eval-data

# CPU only, ~1 min: confirms every checkpoint loads and behaves
python experiments/cross_layer_transfer/run_multilayer.py check \
    --ckpt-root /scratch/$USER/nla_ckpts/nla-adapter

# GPU: round trip at all 28 layers, adapters vs zero-shot
sbatch experiments/cross_layer_transfer/slurm/submit_qwen7b.sh
```

`check` needs no AV or AR. It uses paired activations (same token, all layers)
and verifies that `encode` moves layer-l activations toward the layer-20
activation and `decode` does the reverse. Expected mean cosine gain over no
adapter: about +0.12 to +0.15.

The SLURM job then writes `summary.json` for the adapter arm and the zero-shot
arm. Per-layer `cos_roundtrip` should land near upstream's numbers above;
`cjk_rate` should be near zero (a high rate means injection is failing).

## Using it in an experiment

```python
import sys; sys.path.insert(0, "experiments/cross_layer_transfer")
import torch
from multilayer import MultiLayerNLA, extract_all_layers

mnla = MultiLayerNLA.from_checkpoints(
    "c_direct", ckpt_root="/scratch/.../nla-adapter",
    av_checkpoint="kitft/nla-qwen2.5-7b-L20-av",     # SGLang must serve this
    ar_checkpoint="kitft/nla-qwen2.5-7b-L20-ar",
    sglang_url="http://localhost:30000", critic_device="cuda",
)

acts = extract_all_layers(model, input_ids)          # [28, T, d], raw
h = acts[:, -1]                                      # last token, all layers
layers = torch.arange(h.shape[0])

texts = mnla.verbalize(h, layers, temperature=0.0)   # 28 explanations, one call
edited = [my_edit(t) for t in texts]                 # e.g. remove a key phrase
h_hat = mnla.reconstruct(edited, layers, target_norms=h.norm(dim=-1))
```

Every method takes a mixed batch: `acts` is `[N, d]` and `layers` is `[N]`, one
layer index per row. "All layers at once" is a batch whose rows come from
different layers; there is one AV server and one AR in memory either way.

Pass `adapter=None` (or `--adapter none`) for the zero-shot baseline.

## Things to know before trusting a result

- **Norms are not learned.** Adapters and the AR are trained on cosine only.
  `reconstruct(..., target_norms=...)` rescales the output; always pass the
  norm of the activation you are replacing before patching into a forward pass.
- **Layer index = raw output of decoder block l**, captured by forward hook.
  Do not use `output_hidden_states` for this: HF returns the last layer
  *after* the final LayerNorm. `extract_all_layers` handles it.
- **Training distribution is narrow.** The adapters saw activations at token
  position 10 of raw web/wiki/pubmed/code/reddit text. Late positions and
  chat-formatted prompts are out of distribution; validate with `roundtrip`
  on your own activations before reading much into the explanations.
- **Protocol differs slightly from upstream's training loop.** Upstream fed
  the AV's raw 80-token greedy output, `<explanation>` tag included, to the
  AR. This repo strips the tags and allows 200 tokens. The `*_direct` output
  adapters were fitted to the upstream form; if round-trip cosine comes in
  below the upstream table, compare `--max-new-tokens 80` and the `*_align`
  checkpoints, which never saw AR outputs.
- **Early and last layers are weak** even with adapters (cosine about 0.6 to
  0.7 at layers 0 to 9 and at layer 27, against 0.79 at layer 20).
