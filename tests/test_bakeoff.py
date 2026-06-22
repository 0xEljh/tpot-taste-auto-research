"""Tests for the scorer bake-off pure helpers (doc 05 §7).

These guarantee every bake-off arm is evaluated IDENTICALLY on the human held-out
(comparability is the whole point) and that the train/held-out pointwise data is built
correctly. Written before tpot_taste/scoring/bakeoff.py exists — must fail on import first.
"""
from __future__ import annotations

import json

import numpy as np

from tpot_taste.scoring.bakeoff import LABEL_MAP, human_eval, load_labels, pointwise_rows


def test_label_map_three_classes():
    assert LABEL_MAP["👍 tpot"] == 1.0
    assert LABEL_MAP["🤔 borderline"] == 0.5
    assert LABEL_MAP["👎 not tpot"] == 0.0


def test_load_labels_maps_and_drops_unknown(tmp_path):
    p = tmp_path / "labels.json"
    p.write_text(json.dumps({"1": "👍 tpot", "2": "👎 not tpot", "3": "??? unlabeled"}))
    assert load_labels(p) == {1: 1.0, 2: 0.0}


def test_pointwise_rows_filters_to_labeled_and_id_subset():
    calib = [{"id": 1, "text": "a"}, {"id": 2, "text": "b"}, {"id": 3, "text": "c"}]
    labels = {1: 1.0, 2: 0.0}  # id 3 unlabeled
    # full id set -> only labeled rows returned
    rows = pointwise_rows(calib, labels, ids={1, 2, 3})
    assert sorted(rows) == [(1, "a", 1.0), (2, "b", 0.0)]
    # restricting ids subsets further
    assert pointwise_rows(calib, labels, ids={1}) == [(1, "a", 1.0)]


def test_human_eval_perfect_separation():
    human = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([2.0, 1.5, -1.0, -2.0])
    m = human_eval(scores, human)
    assert m["pairwise"] == 1.0
    # tied human classes cap a perfectly-ordered scorer at rho≈0.894, not 1.0
    assert m["spearman"] > 0.85
    assert m["n_tpot"] == 2 and m["n_not"] == 2


def test_human_eval_reversed_is_zero():
    human = np.array([1.0, 1.0, 0.0, 0.0])
    scores = np.array([-2.0, -1.0, 1.5, 2.0])
    m = human_eval(scores, human)
    assert m["pairwise"] == 0.0
    assert m["spearman"] < -0.85


def test_human_eval_ties_count_half():
    human = np.array([1.0, 0.0])
    scores = np.array([0.5, 0.5])
    assert human_eval(scores, human)["pairwise"] == 0.5


def test_human_eval_excludes_borderline_from_pairwise():
    human = np.array([1.0, 0.5, 0.0])
    scores = np.array([3.0, 99.0, -3.0])  # borderline score is irrelevant to tpot>not
    m = human_eval(scores, human)
    assert m["n_tpot"] == 1 and m["n_not"] == 1
    assert m["pairwise"] == 1.0


def test_human_eval_handles_degenerate_scores():
    # zero-variance scores -> spearman NaN, pairwise all-ties = 0.5, no crash
    human = np.array([1.0, 0.0, 0.5])
    scores = np.array([0.0, 0.0, 0.0])
    m = human_eval(scores, human)
    assert m["pairwise"] == 0.5
    assert np.isnan(m["spearman"])
