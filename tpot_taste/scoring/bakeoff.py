"""Pure, GPU-free helpers for the scorer bake-off (doc 05 §7).

Shared across every arm so the comparison is apples-to-apples: each arm produces a 1-D
score per held-out item, and `human_eval` turns it into the SAME three metrics
`eval_scorer_vs_human.py` reports for the v7.1/v7.2 BT baselines (pairwise-acc(tpot>not),
Spearman, precision@taste-rate). Comparability is the whole point of the bake-off.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

# emoji calibration label -> ordinal taste target (matches eval_scorer_vs_human.py + pull script)
LABEL_MAP = {
    "👍 tpot": 1.0, "🤔 borderline": 0.5, "👎 not tpot": 0.0,
    "tpot": 1.0, "borderline": 0.5, "not tpot": 0.0,
}


def load_labels(labels_path) -> dict[int, float]:
    """Read calibration_labels.json ({id_str: emoji-label}) -> {id_int: float in {0,0.5,1}}.

    Unknown / blank label strings are dropped (only the three known classes survive)."""
    raw = json.loads(Path(labels_path).read_text())
    return {int(k): LABEL_MAP[v] for k, v in raw.items() if v in LABEL_MAP}


def pointwise_rows(calib_rows, labels: dict[int, float], ids) -> list[tuple[int, str, float]]:
    """Build [(id, text, target)] for ids that are both in `ids` and labeled.

    calib_rows: dicts with 'id' and 'text'. Used to assemble train / held-out pointwise sets."""
    keep = set(ids)
    return [(r["id"], r["text"], labels[r["id"]])
            for r in calib_rows if r["id"] in keep and r["id"] in labels]


def human_eval(scores, human) -> dict:
    """The shared arm metric. scores/human: 1-D arrays aligned by item; human in {0,0.5,1}.

    - pairwise: acc over all (tpot, not) cross-pairs that score(tpot) > score(not); ties=0.5;
      random=0.5. Highest-power metric at small N. Borderline items are excluded here.
    - spearman: rank corr over ALL items (incl. borderline); NaN if scores are degenerate.
    - precision: precision@(human tpot-rate) over decided (non-borderline) items.
    """
    s = np.asarray(scores, dtype=float)
    h = np.asarray(human, dtype=float)
    hi, lo = s[h == 1.0], s[h == 0.0]
    pw = (float(np.mean([(1.0 if a > b else 0.5 if a == b else 0.0) for a in hi for b in lo]))
          if len(hi) and len(lo) else float("nan"))
    rho = float(spearmanr(s, h)[0]) if len(s) > 2 and np.std(s) > 0 else float("nan")
    dec = h != 0.5
    if int(dec.sum()) and bool((h == 1.0).any()):
        thr = np.percentile(s, 100 * (1 - (h == 1.0).mean()))
        denom = max(1, int((s[dec] >= thr).sum()))
        prec = float(((h[dec] == 1.0) & (s[dec] >= thr)).sum() / denom)
    else:
        prec = float("nan")
    return {"pairwise": pw, "spearman": rho, "precision": prec,
            "n": int(len(s)), "n_tpot": int((h == 1.0).sum()), "n_not": int((h == 0.0).sum())}


def cv_fold_indices(targets, n_splits: int = 5, seed: int = 0):
    """Stratified k-fold over pointwise taste targets (3 classes 0/0.5/1).

    Returns [(train_idx, test_idx), ...] as int arrays. Every index lands in exactly one
    TEST fold, so concatenating the per-fold held-out predictions gives one out-of-fold
    prediction per sample — the de-noised CV signal the frontier bake-off needs (n=46 on a
    single fixed split is ±0.07; CV over all 177 is the robust read). Stratifying by the
    three label classes keeps every fold from dropping tpot/borderline/not.
    """
    from sklearn.model_selection import StratifiedKFold

    y = np.asarray(targets, dtype=float)
    strat = np.rint(y * 2).astype(int)  # {0.0,0.5,1.0} -> {0,1,2}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in skf.split(np.zeros(len(y)), strat)]


def assemble_oof(fold_results, n: int):
    """Glue per-fold out-of-fold predictions back into ONE length-n score vector.

    For heavy bases (8B) the in-process CV loop OOMs across reloads, so each fold runs in its
    own process and dumps {"idx": [...], "score": [...]} (test-fold global indices + scores).
    `assemble_oof` scatters those back into all_pw order. Indices never written stay NaN — a
    crashed/partial run then reads as partial (aggregate over non-NaN, report n<N) instead of
    silently fabricating a score. cv_fold_indices guarantees a full run covers every index once.
    """
    oof = np.full(int(n), np.nan)
    for fr in fold_results:
        idx = np.asarray(fr["idx"], dtype=int)
        oof[idx] = np.asarray(fr["score"], dtype=float)
    return oof
