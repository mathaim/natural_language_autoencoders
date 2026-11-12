# SAEBench metrics for NLAs

[SAEBench](https://github.com/adamkarvonen/SAEBench) scores sparse
autoencoders by what their latents let you do. An NLA has no latent
dictionary: its bottleneck is a piece of text. This folder adapts SAEBench
metrics to that bottleneck and runs them at every layer of Qwen2.5-7B, using
the all-layer setup in [`../cross_layer_transfer`](../cross_layer_transfer).

| Metric | Status |
|---|---|
| Sparse probing | `sparse_probing.py` |
| Spurious correlation removal (SCR) | planned; reuses the same examples, activations and verbalizations |

## Sparse probing

SAEBench: for a one-vs-rest concept (a profession, a news topic, a sentiment,
a language), pick the K latents whose mean differs most between the classes
and fit a logistic-regression probe on only those. High accuracy at K = 1, 2, 5
means the concept is isolated in a few latents.

NLA version: the latents are the **words of the verbalization** (binary
presence). Everything else follows SAEBench: same datasets and class sets,
same balanced one-vs-rest tasks, same top-K rule, K = 1, 2, 5.

| Column | Probe input | Reads as |
|---|---|---|
| `llm_test_accuracy` | raw activation, all dimensions | ceiling: is the concept in the activation at all |
| `llm_top_{k}_test_accuracy` | top-k activation dimensions | SAEBench's baseline |
| `nla_top_{k}_test_accuracy` | top-k verbalization words | **the metric** |
| `nla_test_accuracy` | all verbalization words | how much the text carries, sparse or not |
| `recon_test_accuracy` | AR reconstruction (probe trained on reconstructions) | what survives the text bottleneck |
| `recon_transfer_test_accuracy` | AR reconstruction, scored by the raw-activation probe | whether it survives in the model's own direction |

`sparse_probing.json` also records, per task and layer, the selected words
with `P(word | class) − P(word | not class)` — the NLA analogue of looking up
which latents were chosen.

### Run

```bash
# small smoke test of the whole path, about 2,000 generations (do this first)
TAG=smoke TRAIN_PER_CLASS=10 TEST_PER_CLASS=6 LAYERS="4 12 20 26" \
    sbatch experiments/saebench_metrics/slurm/submit_sparse_probing_qwen7b.sh

# full run: 2,400 examples x 28 layers, with the adapter and zero-shot
sbatch experiments/saebench_metrics/slurm/submit_sparse_probing_qwen7b.sh
```

Outputs land in `/scratch/$USER/saebench_nla/<TAG>/`:

```
examples.jsonl                     id, dataset, split, label, label_name, text, aux
activations.npy, samples.jsonl     [n_examples, 28, 3584] float16
<arm>/verbalizations.jsonl         one per (example, layer)
<arm>/reconstructions.npy          [n_examples, n_layers, 3584] float16
<arm>/summary.json                 per-layer round-trip cosine and CJK rate
<arm>/sparse_probing.json          per task, per dataset x layer, per layer
<arm>/sparse_probing_by_layer.csv  the headline table
```

`<arm>` is the adapter run (`c_direct` by default) or `none` for the layer-20
NLA applied zero-shot. Other adapter checkpoints: `ARMS="a_align b_direct"`.

### Differences from SAEBench

- **One activation per example**, the last token of the 128-token context.
  SAEBench mean-pools over tokens; an NLA verbalizes a single activation.
- **Texts are at least 300 characters** so that token is past position 50,
  the range the NLA was trained on. Europarl sentences are joined to reach it.
- **100 train / 50 test per class** by default (SAEBench: 2,000 / 500). Every
  example costs one generation per layer. Raise with `TRAIN_PER_CLASS` /
  `TEST_PER_CLASS`.
- **Probes are fit to convergence**, L2 logistic regression with C = 1 (the
  sklearn objective), activations standardized on the training split.
  SAEBench's full-dimension probe early-stops on test accuracy.
- **Four datasets by default**: bias_in_bios set 1, AG News, Amazon sentiment,
  Europarl. bias_in_bios sets 2–3 and Amazon category are available through
  `DATASETS`; `codeparrot/github-code` is not (script-based loader).
- `aux` in `examples.jsonl` keeps the balancing attribute (gender for
  bias_in_bios, rating for Amazon category). SCR needs it.
