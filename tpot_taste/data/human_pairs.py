"""Direct human-pair BT building (Phase 6f, Approach B) — pure, no model.

The Phase-6e finding: v7.1 (distilled from the rubric judge) is ~uncorrelated with the *human's* taste
(Spearman 0.06 on 137 labels). The most direct fix is to make the human's labels the supervision itself:
turn the calibration labels into Bradley-Terry pairs (chosen out-ranks rejected by human label) and train the
scorer on them directly — no judge approximation in the loop.

Why this is safe here (it was NOT for v7): v7 paired judge-high vs judge-low *real* tweets, and real tpot is
systematically terser → the BT head learned register, not taste (the v7 confound). The human's calibration
labels are length-balanced (tpot/borderline/not all ~110 chars; len↔label r=-0.036), so cross-class pairing
carries no length signal. length_tol is therefore optional, not load-bearing.
"""
from __future__ import annotations

import random
from collections import defaultdict

# label floats: tpot / borderline / not
_TYPE_MAP = {"hi_lo": (1.0, 0.0), "hi_mid": (1.0, 0.5), "mid_lo": (0.5, 0.0)}


def stratified_heldout(items, n_heldout: int, seed: int = 0):
    """items: list of (id, label). Return (heldout_ids:set, train_ids:set), stratified by label class.

    The held-out slice is the shared, fixed eval set for the A-vs-B head-to-head; neither approach may train
    on it. Each class contributes ~its population share so the eval covers tpot, borderline, and not alike.
    """
    rng = random.Random(seed)
    by_cls: dict = defaultdict(list)
    for i, lab in items:
        by_cls[lab].append(i)
    total = max(1, len(items))
    held: set = set()
    for lab in sorted(by_cls):
        ids = sorted(by_cls[lab])
        rng.shuffle(ids)
        k = round(n_heldout * len(ids) / total)
        held.update(ids[:k])
    train = {i for i, _ in items if i not in held}
    return held, train


def build_pairs(train_items, *, pair_types=("hi_lo", "hi_mid", "mid_lo"),
                length_tol: int | None = None, reuse_cap: int | None = None, seed: int = 0):
    """train_items: list of dict(id, text, label in {0.0, 0.5, 1.0}). Return BT pair dicts.

    For each requested contrast (hi>lo, hi>mid, mid>lo) form cross-class pairs (chosen=higher label). Optional
    length_tol drops |len(chosen)-len(rejected)|>tol pairs; optional reuse_cap bounds how many pairs any single
    text may appear in (so the handful of tpot texts don't dominate the gradient / get memorized).
    """
    rng = random.Random(seed)
    cls: dict = {0.0: [], 0.5: [], 1.0: []}
    for r in train_items:
        cls[r["label"]].append(r)
    pairs: list = []
    used: dict = defaultdict(int)
    for pt in pair_types:
        hi_lab, lo_lab = _TYPE_MAP[pt]
        combos = [(c, r) for c in cls[hi_lab] for r in cls[lo_lab]]
        rng.shuffle(combos)
        for c, r in combos:
            if length_tol is not None and abs(len(c["text"]) - len(r["text"])) > length_tol:
                continue
            if reuse_cap is not None and (used[c["id"]] >= reuse_cap or used[r["id"]] >= reuse_cap):
                continue
            pairs.append({"text_clean_w": c["text"], "text_clean_l": r["text"],
                          "label_w": hi_lab, "label_l": lo_lab,
                          "len_w": len(c["text"]), "len_l": len(r["text"])})
            used[c["id"]] += 1
            used[r["id"]] += 1
    rng.shuffle(pairs)
    return pairs
