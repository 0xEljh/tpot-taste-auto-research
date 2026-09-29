"""Load / score with the Taste Scorer sequence-classification adapter."""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, BitsAndBytesConfig

from tpot_taste.adapters import DEFAULT_BASE, require_matching_adapter_base, resolve_adapter_base


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


def _device_map_has_cpu_or_disk(model) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        base_model = getattr(model, "base_model", None)
        device_map = getattr(base_model, "hf_device_map", None)
    if not device_map:
        return False
    for value in device_map.values():
        if isinstance(value, int):
            continue
        normalized = str(value).lower()
        if normalized == "disk" or normalized.startswith("cpu"):
            return True
    return False


def assert_cuda_ready(model) -> None:
    """Reject CPU/disk placement before a worker reports ready."""
    try:
        device_type = next(model.parameters()).device.type
    except StopIteration as exc:
        raise RuntimeError("scorer model has no parameters; cannot verify cuda placement") from exc
    if device_type != "cuda":
        raise RuntimeError(f"scorer model must be on cuda before serving, got {device_type!r}")
    if _device_map_has_cpu_or_disk(model):
        raise RuntimeError("scorer model device map contains cpu/disk offload; refusing CPU fallback")


def load_trained_scorer(adapter_dir: str, base_id: str | None = None, *, four_bit: bool = True, require_cuda: bool = False):
    """Base + trained LoRA adapter, for inference/eval."""
    resolved_base = resolve_adapter_base(adapter_dir, DEFAULT_BASE) if base_id is None else require_matching_adapter_base(adapter_dir, base_id)
    from peft import PeftModel

    base, tok = load_reward_model(resolved_base, four_bit=four_bit)
    model = PeftModel.from_pretrained(base, adapter_dir).eval()
    if require_cuda:
        assert_cuda_ready(model)
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
