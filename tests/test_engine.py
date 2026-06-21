"""TDD for TasteEngine pure helpers (Phase 5). No torch/model import."""
from __future__ import annotations

import pytest

from tpot_taste.engine import build_prompt, rank_topk


def test_build_prompt_ideate_default():
    assert "tpot" in build_prompt("ideate").lower()


def test_build_prompt_ideate_topic():
    assert "recursion" in build_prompt("ideate", topic="recursion")


def test_build_prompt_improve_embeds_draft():
    assert "my bad tweet" in build_prompt("improve", draft="my bad tweet")


def test_build_prompt_improve_requires_draft():
    with pytest.raises(ValueError):
        build_prompt("improve")


def test_build_prompt_unknown_kind():
    with pytest.raises(ValueError):
        build_prompt("nonsense")


def test_rank_topk_orders_by_score_desc():
    out = rank_topk(["a", "b", "c"], [1.0, 3.0, 2.0], 2)
    assert [t for t, _ in out] == ["b", "c"]
    assert out[0][1] == 3.0


def test_rank_topk_dedups_and_drops_blanks():
    out = rank_topk(["x", "X ", "", "y"], [5.0, 4.0, 9.0, 1.0], 5)
    assert [t for t, _ in out] == ["x", "y"]  # blank dropped, "X " is a dup of "x"


def test_rank_topk_respects_k():
    out = rank_topk(["a", "b", "c", "d"], [1, 2, 3, 4], 2)
    assert [t for t, _ in out] == ["d", "c"]
