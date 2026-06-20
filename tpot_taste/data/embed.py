"""Lightweight sentence embeddings for topic-matched pairing (MiniLM, mean-pooled).

Uses plain transformers (no sentence-transformers dependency). Returns L2-normalized
vectors, so cosine similarity = dot product.
"""
from __future__ import annotations

import numpy as np

DEFAULT_EMBEDDER = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedder(model_id: str = DEFAULT_EMBEDDER):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModel.from_pretrained(model_id, dtype=dtype)
    if torch.cuda.is_available():
        model = model.cuda()
    model.eval()
    return model, tok


def embed_texts(model, tok, texts, *, batch_size: int = 256, max_length: int = 128) -> np.ndarray:
    import torch

    device = next(model.parameters()).device
    out: list[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = [str(t) for t in texts[i : i + batch_size]]
            enc = tok(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(device)
            hidden = model(**enc).last_hidden_state  # [B, T, H]
            mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)  # [B, T, 1]
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
            pooled = torch.nn.functional.normalize(pooled, dim=-1)
            out.append(pooled.float().cpu().numpy())
    return np.concatenate(out)
