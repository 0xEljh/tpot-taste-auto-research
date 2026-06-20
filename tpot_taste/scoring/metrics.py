"""Ranking metrics for the Taste Scorer (pure, GPU-free, unit-tested).

Primary: pairwise accuracy on held-out same-author pairs.
Secondary: per-author Spearman (honest — global Spearman just re-predicts reach) and NDCG.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def pairwise_accuracy(chosen, rejected) -> float:
    """Fraction of pairs where score(chosen) > score(rejected)."""
    c = np.asarray(chosen, dtype=float)
    r = np.asarray(rejected, dtype=float)
    return float((c > r).mean())


def ndcg_at_k(*, scores, relevances, k: int) -> float:
    """NDCG@k of a single ranked list: rank items by `scores`, score the `relevances`."""
    s = np.asarray(scores, dtype=float)
    rel = np.asarray(relevances, dtype=float)
    order = np.argsort(-s)
    gains = rel[order][:k]
    disc = 1.0 / np.log2(np.arange(2, 2 + len(gains)))
    dcg = float((gains * disc).sum())
    ideal = np.sort(rel)[::-1][:k]
    idisc = 1.0 / np.log2(np.arange(2, 2 + len(ideal)))
    idcg = float((ideal * idisc).sum())
    return dcg / idcg if idcg > 0 else 0.0


def per_author_spearman(authors, scores, targets, *, min_items: int = 5) -> float:
    """Mean within-author Spearman ρ(score, target). Skips authors with < min_items
    or zero variance. NaN if no author qualifies."""
    a = np.asarray(authors)
    s = np.asarray(scores, dtype=float)
    t = np.asarray(targets, dtype=float)
    vals: list[float] = []
    for au in np.unique(a):
        m = a == au
        if int(m.sum()) < min_items:
            continue
        if np.std(s[m]) == 0 or np.std(t[m]) == 0:
            continue
        rho = spearmanr(s[m], t[m])[0]
        if not np.isnan(rho):
            vals.append(float(rho))
    return float(np.mean(vals)) if vals else float("nan")
