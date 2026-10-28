"""Phase 1: extract base-model activations at fact-bearing positions.

For each fact prompt ("The capital of France is") we capture the layer-K
residual stream at two positions and record whether the base model actually
knows the fact (top-1 next token == answer). The second part matters: if the
model doesn't predict "Paris", the fact isn't in the activation and a missed
verbalization tells us nothing. We only trust hit-rates over facts the model
knows.

Two probe positions, because the fact lives in two places:
  - "last"    : the final prompt token (" is"). Where next-token prediction —
                and therefore the answer — is most concentrated.
  - "subject" : the last token of the subject mention ("France"). ROME's causal
                recall site, where the MLP writes the looked-up fact.
We extract both and let the screening report compare them.

Layer indexing — read this, it's the #1 silent-failure source here:
  datagen extracts `model.model.layers[K]`'s OUTPUT, which HF exposes as
  `output_hidden_states=True -> hidden_states[K+1]` (hidden_states[0] is the
  embedding). The released Qwen-L20 AV therefore expects hidden_states[21].
  We default to hs_index = layer + 1. The repo README's quickstart uses
  hidden_states[K] instead — if your verbalizations come out as CJK / garbage,
  flip --hs-offset to 0 and re-extract.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from facts import Fact


@dataclass
class Extraction:
    fact: Fact
    position: str            # "last" or "subject"
    vector: np.ndarray       # [d_model] fp32, raw (NLAClient rescales it)
    model_top1: str          # base model's greedy next token at "last" position
    model_knows: bool        # model_top1 (stripped, lowercased) == answer


def _subject_last_index(tokenizer, prompt: str, subject: str) -> int:
    """Token index of the last token of `subject` within the tokenized prompt.

    Diff the token length of the prompt up-to-and-including the subject against
    the prefix before it. Robust to leading-space BPE quirks because both sides
    are tokenized the same way.
    """
    head = prompt.split(subject)[0]                 # "The capital of "
    head_with_subj = head + subject                 # "The capital of France"
    n_prefix = len(tokenizer(head, add_special_tokens=True)["input_ids"])
    n_with = len(tokenizer(head_with_subj, add_special_tokens=True)["input_ids"])
    assert n_with > n_prefix, (
        f"subject {subject!r} produced no tokens in prompt {prompt!r}"
    )
    return n_with - 1                               # last token of the subject


class BaseModelExtractor:
    def __init__(self, model_name: str, device: str = "cuda",
                 dtype: torch.dtype = torch.bfloat16):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=dtype, trust_remote_code=True,
        ).to(device).eval()
        self.device = device
        print(f"[extractor] {model_name}: "
              f"{self.model.config.num_hidden_layers} layers, "
              f"d_model={self.model.config.hidden_size}")

    @torch.inference_mode()
    def extract(self, fact: Fact, layer: int, hs_offset: int = 1) -> list[Extraction]:
        """Return Extraction records at the 'last' and 'subject' positions."""
        ids = self.tokenizer(fact.prompt, return_tensors="pt",
                             add_special_tokens=True)["input_ids"].to(self.device)
        out = self.model(ids, output_hidden_states=True, use_cache=False)

        hs = out.hidden_states[layer + hs_offset][0]        # [T, d_model]
        last_logits = out.logits[0, -1]                     # next-token at " is"
        top1_id = int(last_logits.argmax())
        top1 = self.tokenizer.decode([top1_id])
        knows = top1.strip().lower() == fact.answer.strip().lower()

        positions = {
            "last": ids.shape[1] - 1,
            "subject": _subject_last_index(self.tokenizer, fact.prompt, fact.subject),
        }
        return [
            Extraction(
                fact=fact,
                position=name,
                vector=hs[idx].float().cpu().numpy(),
                model_top1=top1,
                model_knows=knows,
            )
            for name, idx in positions.items()
        ]
