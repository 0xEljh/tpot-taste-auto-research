"""TDD spec for Scorer ranking metrics (pure functions). Run: uv run pytest -q"""
from __future__ import annotations

import math

import pytest

from tpot_taste.scoring import metrics


def test_pairwise_accuracy():
    # chosen beats rejected in 2 of 3 rows
    assert metrics.pairwise_accuracy([2, 3, 1], [1, 2, 5]) == pytest.approx(2 / 3)


def test_pairwise_accuracy_perfect_and_zero():
    assert metrics.pairwise_accuracy([1, 2, 3], [0, 0, 0]) == 1.0
    assert metrics.pairwise_accuracy([0, 0], [1, 1]) == 0.0


def test_ndcg_at_k_perfect_is_one_and_reversed_is_low():
    rel = [3, 2, 1, 0]
    assert metrics.ndcg_at_k(scores=[4, 3, 2, 1], relevances=rel, k=4) == pytest.approx(1.0)
    rev = metrics.ndcg_at_k(scores=[1, 2, 3, 4], relevances=rel, k=4)
    assert 0.0 < rev < 1.0


def test_per_author_spearman_averages_within_author():
    # author a: score perfectly aligned with target (+1); author b: reversed (-1) -> mean 0
    authors = ["a", "a", "a", "b", "b", "b"]
    scores = [1, 2, 3, 3, 2, 1]
    targets = [1, 2, 3, 1, 2, 3]
    val = metrics.per_author_spearman(authors, scores, targets, min_items=3)
    assert val == pytest.approx(0.0, abs=1e-9)


def test_per_author_spearman_skips_thin_and_constant_authors():
    # author a has only 2 items (< min_items); author c is constant-score -> both skipped;
    # only author b (aligned, +1) counts.
    authors = ["a", "a", "b", "b", "b", "c", "c", "c"]
    scores = [1, 2, 5, 6, 7, 9, 9, 9]
    targets = [1, 2, 1, 2, 3, 1, 2, 3]
    val = metrics.per_author_spearman(authors, scores, targets, min_items=3)
    assert val == pytest.approx(1.0, abs=1e-9)
