# Verbalization Screening

The go/no-go test for the NLA-controllability idea: **when an activation encodes
a fact, does the AV actually verbalize that fact in editable form?**

Everything downstream (editing the verbalization to change behavior, generating
forget targets for unlearning) depends on the answer being yes. This measures
the hit-rate before we build any of that.

## The question, precisely

For a prompt like `The capital of France is`, the layer-K activation at the last
token should encode "Paris" (the model is about to predict it). We verbalize
that activation with the AV and check whether the text mentions **Paris** (the
*answer* / prediction) versus only **France** (the *subject* / input). Three
outcomes:

| result | meaning | next step |
|---|---|---|
| high **answer** hit-rate | fact is verbalized, one-token-editable | run the Tokyo↔Paris swap experiment |
| high **subject**, low answer | AV describes the input, not the prediction | the edit handle is the subject, rethink |
| low both | low ceiling for the whole approach | stop — cheaply learned |

Hit-rates are reported **only over facts the base model actually knows** (top-1
next token == answer). If the model can't predict Paris, the fact isn't in the
activation and a missed verbalization is uninformative.

## Run it

Two phases, decoupled so the HF base model and the SGLang AV server don't
contend for one GPU (and so you can re-verbalize without re-extracting).

```bash
cd experiments/verbalization_screening

# Phase 1 — base model -> activations (+ does-the-model-know check). ~15GB GPU.
python run_screening.py extract \
    --base-model Qwen/Qwen2.5-7B-Instruct --layer 20 --out runs/qwen_l20

# Start the AV server (separate process / GPU)
python -m sglang.launch_server --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 --disable-radix-cache --trust-remote-code

# Phase 2 — activations -> AV -> answer/subject detection -> report
python run_screening.py verbalize \
    --out runs/qwen_l20 \
    --av-checkpoint $HF_HOME/.../nla-qwen2.5-7b-L20-av \
    --sglang-url http://localhost:30000
```

`--av-checkpoint` must be a local HF-format dir containing `nla_meta.yaml`
(the sidecar with the injection token + scale). Download it with
`huggingface-cli download kitft/nla-qwen2.5-7b-L20-av --local-dir ...`.

## Two probe positions

Each fact is extracted at two positions and reported separately:

- **`last`** — final prompt token (`is`). Where the answer prediction is most
  concentrated. Probably your best shot at the AV saying "Paris".
- **`subject`** — last token of the subject (`France`). ROME's causal recall
  site.

Restrict with `--positions last` if you only care about one.

## ⚠️ Layer off-by-one

The repo disagrees with itself on layer indexing. The Qwen worked example states
layer 20 = `hidden_states[21]` (output of block 20); the README quickstart uses
`hidden_states[20]`. We default to `layer + 1` (the example's convention). **If
your verbalizations come out as CJK / unrelated garbage, the injection is seeing
OOD vectors — re-extract with `--hs-offset 0`** and compare. The screening
output itself is the diagnostic: wrong layer ⇒ near-zero hit-rate and incoherent
text.

## Output

- `runs/<name>/activations.npy` — `[N, d_model]` fp32 raw vectors
- `runs/<name>/records.jsonl` — one row per (fact, position): subject, answer,
  `model_knows`, `model_top1`, `vec_index`
- `runs/<name>/results.jsonl` — adds `verbalization`, `answer_hit`,
  `subject_hit`. Read these by hand — the substring detector is a screening
  signal, not a judge; eyeball whether near-misses ("a European capital",
  "Parisian") should count.

## Scaling past the built-ins

`facts.py` ships ~40 country→capital pairs (filtered to single-token answers at
runtime). For more, point `extract` at CounterFact:

```bash
python run_screening.py extract ... --counterfact /path/to/counterfact.json
```

It keeps `capital of` relations by default — the same minimal-pair structure,
plus pre-specified counterfactual targets you'll want for the swap experiment.
