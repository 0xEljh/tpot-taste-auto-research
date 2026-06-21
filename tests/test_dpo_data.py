"""TDD for DPO pair formation (Phase 4). Pure logic — no TRL/torch import."""
from __future__ import annotations

from tpot_taste.writer.dpo_data import form_dpo_pairs

SYS = "sys"


def _g(user, cands, task="ideate"):
    return {"user": user, "task": task, "candidates": cands}


def test_forms_pair_when_margin_met():
    recs = form_dpo_pairs([_g("u", [("good", 3.0), ("bad", 0.0)])], sys_prompt=SYS, min_margin=1.0)
    assert len(recs) == 1
    r = recs[0]
    assert r["chosen"][0]["content"] == "good"
    assert r["rejected"][0]["content"] == "bad"
    assert [m["role"] for m in r["prompt"]] == ["system", "user"]
    assert r["prompt"][1]["content"] == "u"
    assert r["margin"] == 3.0
    assert r["task"] == "ideate"


def test_skips_below_margin():
    recs = form_dpo_pairs([_g("u", [("a", 1.0), ("b", 0.8)])], sys_prompt=SYS, min_margin=1.0)
    assert recs == []


def test_picks_extremes_among_many():
    cands = [("mid", 1.0), ("best", 5.0), ("worst", -2.0), ("mid2", 2.0)]
    recs = form_dpo_pairs([_g("u", cands)], sys_prompt=SYS, min_margin=1.0)
    assert recs[0]["chosen"][0]["content"] == "best"
    assert recs[0]["rejected"][0]["content"] == "worst"
    assert recs[0]["margin"] == 7.0


def test_skips_identical_text():
    recs = form_dpo_pairs([_g("u", [("same", 5.0), ("Same ", 0.0)])], sys_prompt=SYS, min_margin=1.0)
    assert recs == []


def test_skips_too_few_candidates():
    assert form_dpo_pairs([_g("u", [("only", 5.0)])], sys_prompt=SYS, min_margin=1.0) == []


def test_drops_empty_candidates_then_too_few():
    recs = form_dpo_pairs([_g("u", [("good", 3.0), ("", 0.0), ("  ", -1.0)])], sys_prompt=SYS, min_margin=1.0)
    assert recs == []
