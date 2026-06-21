"""TDD for the taste judge's PURE helpers (Phase 6c, D29) — no model required.

The judge separates tpot-taste from generic-competent-virality WITHOUT engagement. These tests pin the
prompt contract + robust verdict/score parsing; the model path + rubric calibration are evaluated against
the dipstick panel (scripts/judge_eval.py), not here.
"""
from __future__ import annotations

from tpot_taste.scoring.judge import (
    agree_winner,
    build_pairwise_prompt,
    build_score_prompt,
    build_score_prompt_fewshot,
    parse_score,
    parse_verdict,
)


def test_fewshot_prompt_embeds_exemplars_then_query():
    msgs = build_score_prompt_fewshot("the QUERY post", [("a witty one", 9), ("a corporate one", 1)])
    assert msgs[0]["role"] == "system" and "tpot" in msgs[0]["content"].lower()
    joined = " ".join(m["content"] for m in msgs)
    assert "a witty one" in joined and "SCORE: 9" in joined
    assert "a corporate one" in joined and "SCORE: 1" in joined
    assert msgs[-1]["role"] == "user" and "the QUERY post" in msgs[-1]["content"]  # query is last


def test_pairwise_prompt_contains_both_and_rubric():
    msgs = build_pairwise_prompt("post A text", "post B text")
    assert msgs[0]["role"] == "system"
    joined = " ".join(m["content"] for m in msgs)
    assert "post A text" in joined and "post B text" in joined
    assert "tpot" in joined.lower()  # rubric is present
    assert "VERDICT" in joined  # the output contract is stated


def test_parse_verdict_canonical():
    assert parse_verdict("VERDICT: A") == "A"
    assert parse_verdict("VERDICT: B") == "B"
    assert parse_verdict("reasoning here...\nVERDICT: B\n") == "B"


def test_parse_verdict_loose_fallback():
    assert parse_verdict("I think A is more tpot.") == "A"
    assert parse_verdict("The more tpot one is B.") == "B"


def test_parse_verdict_garbage_is_none():
    assert parse_verdict("neither is good") is None
    assert parse_verdict("") is None


def test_score_prompt_and_parse():
    msgs = build_score_prompt("some post")
    assert "some post" in msgs[-1]["content"]
    assert parse_score("SCORE: 7") == 7.0
    assert parse_score("I'd rate this 8/10.") == 8.0
    assert parse_score("SCORE: 12") == 10.0  # clamp to [0,10]
    assert parse_score("SCORE: -3") == 0.0
    assert parse_score("no number here") is None


def test_agree_winner_order_invariance():
    # judge run twice with sides swapped; agree only if the SAME post wins both times.
    # round1 (a,b): winner 'A' = post a.  round2 (b,a): winner 'B' = post a.  -> agree on a.
    assert agree_winner("A", "B") == "a"
    assert agree_winner("B", "A") == "b"
    assert agree_winner("A", "A") is None  # contradiction (position bias) -> no decision
    assert agree_winner("B", "B") is None
    assert agree_winner("A", None) is None
