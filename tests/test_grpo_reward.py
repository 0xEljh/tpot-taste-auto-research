"""TDD for the multi-objective GRPO reward components (Phase 4b). Pure — no torch."""
from __future__ import annotations

from tpot_taste.writer.grpo_reward import is_baity, repetition_ratio, reward_components


def test_clean_post_unpenalized():
    assert reward_components("a sharp little aphorism", 3.0) == 3.0


def test_length_penalty_over_target():
    assert reward_components("x" * 300, 3.0, length_target=200, length_penalty=0.01) == 3.0 - 1.0


def test_under_target_no_length_penalty():
    assert reward_components("x" * 150, 2.0, length_target=200, length_penalty=0.01) == 2.0


def test_bait_phrase_penalized():
    assert reward_components("RT if you agree", 2.0, bait_penalty=2.0) == 0.0


def test_hashtag_spam_is_baity():
    assert is_baity("cool thought #mindblown #tryhard")
    assert not is_baity("one #hashtag is fine")


def test_emoji_spam_is_baity():
    assert is_baity("wow 🚀🔥😭")
    assert not is_baity("just one emoji 🚀 here")


def test_length_and_bait_combine():
    r = reward_components("like if you agree " + "x" * 250, 1.0,
                          length_target=200, length_penalty=0.01, bait_penalty=2.0)
    assert r < 0  # over-length AND baity


def test_repetition_ratio():
    assert repetition_ratio("i will tell you i will tell you i will tell you") > 0.5
    assert repetition_ratio("a varied sentence with all distinct words right here") < 0.1
    assert repetition_ratio("too short here") == 0.0  # < 6 words


def test_rep_penalty_demotes_repetitive():
    repetitive = "go away go away go away go away go away"
    clean = "a thoughtful and varied little remark about modern life"
    assert reward_components(repetitive, 3.0, rep_penalty=3.0) < reward_components(clean, 3.0, rep_penalty=3.0)
