"""TDD for the Writer SFT data builder (Phase 3, see docs/design/03-writer-design.md).

Writer SFT v1 = two tasks assembled from artifacts we already have:
  - ideate: fixed/paraphrased prompt -> a real curated good tweet (voice cloning)
  - improve: a deopt pair's `rejected` (weak draft) -> its `chosen` (good)  [free supervision]
Topic-conditioned ideate + make-it-tpot are deferred (need a topic-extraction pass).
"""
from __future__ import annotations

import polars as pl

from tpot_taste.writer import sft_data


GOODS = [
    "every codebase is a haunted house and you are the ghost",
    "spent 3 years thinking my problem was discipline. it was sleep.",
    "the map is not the territory but the territory is also kind of a map",
    "i keep a list of ideas i'm too scared to try. it's my best writing.",
]

PAIRS = pl.DataFrame(
    {
        "text_clean_w": [
            "teslas are charged with direct current and that's kind of poetic",
            "the best abstractions feel inevitable once you see them",
        ],
        "text_clean_l": [
            "Does anyone else find it ironic that Teslas use DC? #Tesla #Facts",
            "Abstractions are useful tools in software engineering, generally speaking.",
        ],
        "style": [
            "Rewrite this post as engagement-bait that fishes for replies.",
            "Rewrite this post to be bland, generic and forgettable.",
        ],
    }
)


def _records(**kw):
    return sft_data.make_sft_records(GOODS, PAIRS, seed=0, **kw)


def test_record_schema():
    recs = _records(n_ideate=4, n_improve=2)
    assert recs, "builder returned no records"
    for r in recs:
        assert r["task"] in {"ideate", "improve"}
        msgs = r["messages"]
        assert [m["role"] for m in msgs] == ["system", "user", "assistant"]
        assert all(m["content"].strip() for m in msgs), "empty message content"


def test_ideate_completion_is_a_real_good():
    recs = _records(n_ideate=4, n_improve=0)
    assert {r["task"] for r in recs} == {"ideate"}
    goodset = set(GOODS)
    for r in recs:
        assert r["messages"][-1]["content"] in goodset


def test_improve_maps_rejected_to_chosen():
    recs = _records(n_ideate=0, n_improve=2)
    assert {r["task"] for r in recs} == {"improve"}
    pair_by_chosen = dict(zip(PAIRS["text_clean_w"], PAIRS["text_clean_l"]))
    for r in recs:
        chosen = r["messages"][-1]["content"]
        user = r["messages"][1]["content"]
        assert chosen in pair_by_chosen, "improve target is not a pair's chosen"
        assert pair_by_chosen[chosen] in user, "the weak draft (rejected) must appear in the prompt"


def test_counts_match_request():
    recs = _records(n_ideate=3, n_improve=2)
    by = {}
    for r in recs:
        by[r["task"]] = by.get(r["task"], 0) + 1
    assert by.get("ideate", 0) == 3
    assert by.get("improve", 0) == 2


def test_caps_at_available_data():
    # asking for more than exists must not duplicate or crash
    recs = _records(n_ideate=999, n_improve=999)
    ideate = [r for r in recs if r["task"] == "ideate"]
    improve = [r for r in recs if r["task"] == "improve"]
    assert len(ideate) == len(GOODS)
    assert len(improve) == PAIRS.height
    completions = [r["messages"][-1]["content"] for r in ideate]
    assert len(completions) == len(set(completions)), "ideate completions duplicated"


def test_forbidden_completions_excluded():
    # leakage guard: nothing whose completion is in the eval set
    forbidden = {GOODS[0]}
    recs = _records(n_ideate=4, n_improve=0, forbidden=forbidden)
    assert all(r["messages"][-1]["content"] != GOODS[0] for r in recs)
    assert len(recs) == len(GOODS) - 1


def test_deterministic_under_seed():
    a = sft_data.make_sft_records(GOODS, PAIRS, seed=7, n_ideate=4, n_improve=2)
    b = sft_data.make_sft_records(GOODS, PAIRS, seed=7, n_ideate=4, n_improve=2)
    assert a == b


# ---- Phase 7: drop engagement-bait from the SFT corpus the writer voice-clones ----

def test_ideate_drops_baity_goods():
    # the writer voice-clones its targets, so a bait good must never become an ideate completion
    goods = GOODS + ["drop a 🔥 in the replies if you agree, tag a friend who needs this 👇👇👇"]
    recs = sft_data.make_sft_records(goods, PAIRS, seed=0, n_ideate=99, n_improve=0)
    completions = [r["messages"][-1]["content"] for r in recs]
    assert goods[-1] not in completions
    assert len(recs) == len(GOODS)  # only the clean goods survive


def test_improve_drops_baity_targets():
    pairs = pl.DataFrame({
        "text_clean_w": [
            "the best abstractions feel inevitable once you see them",       # clean target -> kept
            "I'm giving away $5,000 to 3 people who retweet this! ❤️🔥👇",     # baity target -> dropped
        ],
        "text_clean_l": ["abstractions are useful tools generally", "money giveaway, generic version"],
        "style": ["bland", "bland"],
    })
    recs = sft_data.make_sft_records(GOODS, pairs, seed=0, n_ideate=0, n_improve=2)
    targets = [r["messages"][-1]["content"] for r in recs]
    assert pairs["text_clean_w"][1] not in targets
    assert len(recs) == 1
