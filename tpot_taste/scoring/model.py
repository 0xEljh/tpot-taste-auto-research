"""Load / score with the Qwen2.5-3B reward model (the Taste Scorer).

The Scorer is `AutoModelForSequenceClassification(num_labels=1)` on the SAME base as
the Writer (Qwen2.5-3B-Instruct), 4-bit, trained with a Bradley-Terry pairwise loss.
A higher scalar = predicted to do better with the tpot audience (relative to author).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

DEFAULT_BASE = "Qwen/Qwen2.5-3B-Instruct"


def load_tokenizer(model_id: str = DEFAULT_BASE):
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def _bnb4() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )


def load_reward_model(model_id: str = DEFAULT_BASE, *, four_bit: bool = True):
    """Fresh reward model (random scalar head) for training."""
    kwargs = {"num_labels": 1, "dtype": torch.bfloat16}
    if four_bit:
        kwargs["quantization_config"] = _bnb4()
    model = AutoModelForSequenceClassification.from_pretrained(model_id, **kwargs)
    tok = load_tokenizer(model_id)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def load_trained_scorer(adapter_dir: str, base_id: str = DEFAULT_BASE, *, four_bit: bool = True):
    """Base + trained LoRA adapter, for inference/eval."""
    from peft import PeftModel

    base, tok = load_reward_model(base_id, four_bit=four_bit)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model, tok


@torch.no_grad()
def score_texts(model, tok, texts, *, batch_size: int = 32, max_length: int = 256) -> np.ndarray:
    """Scalar reward per text."""
    device = next(model.parameters()).device
    out: list[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = [str(t) for t in texts[i : i + batch_size]]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        logits = model(**enc).logits.squeeze(-1)
        out.append(logits.float().cpu().numpy())
    return np.concatenate(out)
