# ROME Ablation Sweep

Measures whether NLA verbalization edits actually suppress factual recall — and
whether they leak across unpatched token positions.

## The leakage problem this fixes

Earlier experiments patched only the "subject" or "last" token position.
The model could recover the fact from the **other T−1 positions** that were
never touched. Every condition here replaces **all** prompt token embeddings,
so there is no unpatched residual for the model to fall back on.

## Conditions

| name | what is injected at every position | what it tests |
|---|---|---|
| `baseline` | no injection (raw input_ids) | gold reference |
| `empty` | AR("") | ablation floor — does erasing content suppress recall? |
| `control` | AR(AV(h_l)) | round-trip faithfulness — does text bottleneck preserve info? |
| `forget` | AR(AV(h_l) + " Forget {gold}.") | does a forget instruction transmit? |
| `ground_truth` | AR("The answer to this prompt is {gold}.") | upper bound — can AR amplify an explicit label? |

## Usage

```bash
# Phase 1 — extract activations at EVERY token position
python sweep.py extract \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --layer 20 \
    --counterfact /data/counterfact.json \
    --n-samples 200 \
    --out runs/qwen_l20

# Start the AV server (separate process / GPU)
python -m sglang.launch_server \
    --model-path kitft/nla-qwen2.5-7b-L20-av \
    --port 30000 --disable-radix-cache --trust-remote-code

# Phase 2 — run all ablation conditions
python sweep.py sweep \
    --out runs/qwen_l20 \
    --av-checkpoint kitft/nla-qwen2.5-7b-L20-av \
    --ar-checkpoint kitft/nla-qwen2.5-7b-L20-ar \
    --base-model Qwen/Qwen2.5-7B-Instruct \
    --layer 20 \
    --sglang-url http://localhost:30000

# Run a subset of conditions
python sweep.py sweep ... --conditions baseline empty control
```

## Interpreting results

```
empty   ≈ 0%         → erasing ALL positions fully suppresses recall (ideal floor)
empty   > 30%        → AR("") inadvertently reconstructs a meaningful vector
control ≈ baseline   → round-trip faithful; text bottleneck preserves the fact
forget  < baseline   → forget instruction transmitted  ✓
forget  ≈ baseline   → instruction NOT transmitted (bottleneck lossy)
ground_truth > baseline → AR amplifies explicit text cues (useful ceiling)
```

## Outputs

- `runs/<name>/activations.npy` — float32 matrix [Σ T_i, d_model], all positions
- `runs/<name>/samples.jsonl`  — per-fact metadata + vec_start / n_tokens slice
- `runs/<name>/sweep_results.jsonl` — per-fact results for every condition,
  including the verbalization text that was fed to AR

## Connection to unlearning

The WMDP/MMLU results showed MMLU collapsing alongside WMDP under the NLA
forget intervention. The most likely cause is that `forget` only partially
transmits through the AR bottleneck (so WMDP recall doesn't drop enough) OR
that patching positions outside the "answer" sites disrupts general-capability
circuits. This sweep isolates which: if `control` (no edit) also hurts MMLU
recall, the issue is round-trip lossyness, not the forget instruction per se.
