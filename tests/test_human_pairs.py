"""TDD for direct human-pair BT building (Phase 6f) — pure, no model.

Approach B: turn the human's calibration labels (tpot=1.0 / borderline=0.5 / not=0.0) directly into BT
preference pairs (no judge in the loop). Two invariants matter: (1) the train/held-out split must not leak,
so the head-to-head eval vs Approach A is fair; (2) chosen must always out-rank rejected by human label.
The human's labels are length-balanced (len↔label r=-0.036), so length-matching is optional, not required.
"""
from __future__ import annotations

from tpot_taste.data.human_pairs import build_pairs, resolve_split, stratified_heldout


def _items(n_hi, n_mid, n_lo):
    items = []
    i = 0
    for lab, n in [(1.0, n_hi), (0.5, n_mid), (0.0, n_lo)]:
        for _ in range(n):
            items.append({"id": i, "text": f"text-{i} " + "x" * (i % 50), "label": lab})
            i += 1
    return items


def test_stratified_heldout_no_overlap_and_covers_all():
    items = [(r["id"], r["label"]) for r in _items(35, 42, 60)]
    held, train = stratified_heldout(items, n_heldout=45, seed=0)
    assert held & train == set()                       # no leak
    assert held | train == {i for i, _ in items}       # partition
    assert abs(len(held) - 45) <= 3                     # roughly the requested size


def test_stratified_heldout_is_proportional_per_class():
    items = [(r["id"], r["label"]) for r in _items(35, 42, 60)]
    held, _ = stratified_heldout(items, n_heldout=45, seed=0)
    lab = {i: l for i, l in items}
    # each class should contribute ~ its share of the held-out (45/137 ~ 33%)
    for cls, n_cls in [(1.0, 35), (0.5, 42), (0.0, 60)]:
        got = sum(1 for i in held if lab[i] == cls)
        assert abs(got - round(45 * n_cls / 137)) <= 1


def test_build_pairs_chosen_always_outranks_rejected():
    pairs = build_pairs(_items(10, 8, 12), seed=0)
    assert pairs                                        # non-empty
    assert all(p["label_w"] > p["label_l"] for p in pairs)


def test_build_pairs_covers_all_three_contrasts():
    pairs = build_pairs(_items(10, 8, 12), seed=0)
    contrasts = {(p["label_w"], p["label_l"]) for p in pairs}
    assert contrasts == {(1.0, 0.0), (1.0, 0.5), (0.5, 0.0)}


def test_build_pairs_reuse_cap_limits_text_reuse():
    pairs = build_pairs(_items(10, 8, 12), reuse_cap=3, seed=0)
    from collections import Counter
    c = Counter()
    for p in pairs:
        c[p["text_clean_w"]] += 1
        c[p["text_clean_l"]] += 1
    assert max(c.values()) <= 3


def test_build_pairs_length_tol_respected():
    pairs = build_pairs(_items(10, 8, 12), length_tol=5, seed=0)
    assert all(abs(p["len_w"] - p["len_l"]) <= 5 for p in pairs)


def test_build_pairs_only_hi_lo_when_restricted():
    pairs = build_pairs(_items(10, 8, 12), pair_types=("hi_lo",), seed=0)
    assert {(p["label_w"], p["label_l"]) for p in pairs} == {(1.0, 0.0)}


# --- held-out FREEZE for the active-learning loop (round 3+): the eval set must not drift ---

def _idlab(ids_labels):
    return [(i, l) for i, l in ids_labels]


def test_resolve_split_freezes_existing_heldout():
    items = _idlab([(1, 1.0), (2, 0.0), (3, 1.0), (4, 0.0), (5, 0.5), (6, 0.5)])
    held, train = resolve_split(items, n_heldout=2, existing_heldout={1, 2})
    assert held == {1, 2}
    assert train == {3, 4, 5, 6}


def test_resolve_split_new_items_go_to_train_not_heldout():
    # THE active-learning property: adding ids 7,8 leaves held-out unchanged; they land in train
    items = _idlab([(1, 1.0), (2, 0.0), (3, 1.0), (4, 0.0), (7, 1.0), (8, 0.0)])
    held, train = resolve_split(items, n_heldout=2, existing_heldout={1, 2})
    assert held == {1, 2}
    assert {7, 8} <= train


def test_resolve_split_drops_heldout_ids_absent_from_items():
    items = _idlab([(1, 1.0), (3, 0.0)])
    held, train = resolve_split(items, n_heldout=2, existing_heldout={1, 99})
    assert held == {1}            # 99 not currently labeled -> dropped
    assert train == {3}


def test_resolve_split_none_matches_stratified():
    items = _idlab([(i, float(i % 3) / 2) for i in range(1, 31)])
    a = resolve_split(items, n_heldout=9, seed=0, existing_heldout=None)
    b = stratified_heldout(items, n_heldout=9, seed=0)
    assert a == b


def test_resolve_split_empty_existing_falls_back_to_stratified():
    # an empty set is falsy -> stratify a fresh held-out (not "freeze nothing")
    items = _idlab([(i, float(i % 3) / 2) for i in range(1, 31)])
    a = resolve_split(items, n_heldout=9, seed=0, existing_heldout=set())
    b = stratified_heldout(items, n_heldout=9, seed=0)
    assert a == b
