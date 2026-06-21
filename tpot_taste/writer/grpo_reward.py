"""Multi-objective GRPO reward (Phase 4b, design §3): v6 taste score MINUS guardrails.

The whole project shows raw reward gets gamed (scorer v5, DPO v1 length-hack). So the GRPO reward
is v6 score − length penalty (over a target) − a bait/cringe penalty, computed per completion. The
pure component math is unit-tested; the v6 scorer is wired in by scripts/train_grpo.py.
"""
from __future__ import annotations

import re

_BAIT_PHRASE = re.compile(
    r"(?i)(\brt if\b|retweet if|like if|reply (yes|below|with)|agree\?|tag someone|drop a |comment below|"
    r"smash that|link in bio)"
)
_HASHTAG = re.compile(r"#\w+")
_EMOJI = re.compile(r"[\U0001F300-\U0001FAFF☀-➿]")


def is_baity(text: str) -> bool:
    if _BAIT_PHRASE.search(text):
        return True
    if len(_HASHTAG.findall(text)) >= 2:  # hashtag spam
        return True
    if len(_EMOJI.findall(text)) >= 3:  # emoji spam
        return True
    return False


def reward_components(
    text: str,
    base_score: float,
    *,
    length_target: int = 200,
    length_penalty: float = 0.01,
    bait_penalty: float = 2.0,
) -> float:
    """v6 score − λ·max(0, len−target) − bait_penalty·[is_baity]."""
    r = float(base_score)
    r -= length_penalty * max(0, len(text) - length_target)
    if is_baity(text):
        r -= bait_penalty
    return r
