"""TDD for TasteEngine pure helpers (Phase 5). No torch/model import."""
from __future__ import annotations

import json

import pytest

from tpot_taste.engine import build_prompt, rank_topk, resolve_scorer_base


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


def test_resolve_scorer_base_reads_adapter_config(tmp_path):
    # A scorer carries its own base (PEFT records it) — so an 8B scorer loads on an 8B base
    # even when the Writer (and the engine default) is the 3B. This is the shared-base decouple.
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "Qwen/Qwen3-8B"}))
    assert resolve_scorer_base(tmp_path, "Qwen/Qwen2.5-3B-Instruct") == "Qwen/Qwen3-8B"


def test_resolve_scorer_base_falls_back_when_no_config(tmp_path):
    # No adapter_config (or no recorded base) -> fall back to the caller's base, preserving
    # the old shared-base behaviour for legacy adapters.
    assert resolve_scorer_base(tmp_path, "Qwen/Qwen2.5-3B-Instruct") == "Qwen/Qwen2.5-3B-Instruct"
    (tmp_path / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": None}))
    assert resolve_scorer_base(tmp_path, "fallback-base") == "fallback-base"
