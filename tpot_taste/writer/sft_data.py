"""Assemble Writer SFT supervision from artifacts we already have (Phase 3, D17 → Phase 3).

Two tasks, both built without any new generation:
  - ideate : a paraphrased "write a tpot post" prompt -> a real curated good tweet (voice cloning)
  - improve: a deopt pair's `rejected` (weak draft) -> its `chosen` (good)   [free bad->good supervision]

Records are chat-message dicts ready for TRL SFTTrainer (assistant turn = the real tweet, masked-prompt SFT).
Topic-conditioned ideate + make-it-tpot are deferred (need a topic-extraction pass) — see 03-writer-design.md.
"""
from __future__ import annotations

import random
from collections.abc import Iterable

import polars as pl

from tpot_taste.writer.grpo_reward import is_baity

SYS_PROMPT = (
    "You write posts for tpot — the introspective, intellectually playful corner of tech Twitter. "
    "Voice: specific, earnest or wry, a little surprising. Never engagement-bait, corporate, or platitudes."
)

IDEATE_PROMPTS = [
    "Write a tpot post.",
    "Write a post that would land well on tpot.",
    "Post something tpot would like.",
    "Write a short, original tpot-style post.",
]

IMPROVE_PROMPTS = [
    "Improve this post so it lands on tpot. Keep the idea; sharpen the voice.\n\n{draft}",
    "Rewrite this so it would do well on tpot:\n\n{draft}",
    "This post is weak. Make it a strong tpot post:\n\n{draft}",
]


def _record(task: str, user: str, assistant: str) -> dict:
    return {
        "task": task,
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ],
    }


def make_sft_records(
    goods: Iterable[str],
    pairs: pl.DataFrame,
    *,
    n_ideate: int,
    n_improve: int,
    seed: int = 0,
    forbidden: set[str] | None = None,
) -> list[dict]:
    """Build a mixed ideate+improve SFT set.

    Caps at available data (no duplication). `forbidden` excludes any example whose completion
    (the real tweet) is in it — the leakage guard against held-out/eval text. Engagement-bait is
    dropped from both the ideate goods and the improve targets (is_baity) — the writer voice-clones
    its targets, so a "$5,000 giveaway, RT to win" target teaches bait.
    """
    rng = random.Random(seed)
    forbidden = forbidden or set()
    records: list[dict] = []

    # ideate: unique goods (dedup, preserve order), drop forbidden + bait, shuffle, cap
    goods_u = [g for g in dict.fromkeys(goods) if g not in forbidden and not is_baity(g)]
    rng.shuffle(goods_u)
    for g in goods_u[:n_ideate]:
        records.append(_record("ideate", rng.choice(IDEATE_PROMPTS), g))

    # improve: deopt pairs, rejected (weak) -> chosen (good); drop forbidden + baity targets
    rows = [
        (w, l)
        for w, l in zip(pairs["text_clean_w"].to_list(), pairs["text_clean_l"].to_list())
        if w not in forbidden and not is_baity(w)
    ]
    rng.shuffle(rows)
    for w, l in rows[:n_improve]:
        records.append(_record("improve", rng.choice(IMPROVE_PROMPTS).format(draft=l), w))

    return records
