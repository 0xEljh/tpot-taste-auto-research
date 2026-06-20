"""TDD spec for the Phase-1 data pipeline (clean -> labels -> pairs -> splits).

These tests are written before the implementation and pin the *behaviour* that the
design (docs/design/02-data-findings.md) requires. Run: uv run pytest -q
"""
from __future__ import annotations

import math
from datetime import datetime

import polars as pl
import pytest

from tpot_taste.data import clean, labels, pairs, splits


# ----------------------------- fixtures -----------------------------------

def _raw(rows: list[dict]) -> pl.DataFrame:
    """Build a raw-schema frame (subset of columns the pipeline reads)."""
    schema = {
        "tweet_id": pl.Int64,
        "account_id": pl.Int64,
        "username": pl.Utf8,
        "created_at": pl.Utf8,
        "full_text": pl.Utf8,
        "favorite_count": pl.Int64,
        "retweet_count": pl.Int64,
        "reply_to_tweet_id": pl.Int64,
        "quoted_tweet_id": pl.Int64,
        "archive_upload_id": pl.Int64,
    }
    return pl.DataFrame(rows, schema=schema)


# ----------------------------- clean --------------------------------------

def test_clean_parses_dates_marks_uploaders_and_drops_rt_and_dups():
    df = _raw([
        # uploader (has archive_upload_id), normal tweet
        {"tweet_id": 1, "account_id": 10, "username": "up", "created_at": "2020-05-01 12:00:00+00",
         "full_text": "hello world", "favorite_count": 5, "retweet_count": 1,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": 99},
        # duplicate tweet_id -> dropped
        {"tweet_id": 1, "account_id": 10, "username": "up", "created_at": "2020-05-01 12:00:00+00",
         "full_text": "hello world", "favorite_count": 5, "retweet_count": 1,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": 99},
        # pure retweet -> dropped
        {"tweet_id": 2, "account_id": 10, "username": "up", "created_at": "2021-01-01 00:00:00+00",
         "full_text": "RT @someone: lol", "favorite_count": 0, "retweet_count": 0,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": 99},
        # liked/context tweet (no archive_upload_id) -> kept, is_uploader False
        {"tweet_id": 3, "account_id": 77, "username": "other", "created_at": "2019-03-03 03:03:03+00",
         "full_text": "a liked tweet", "favorite_count": 100, "retweet_count": 20,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": None},
        # empty text -> dropped
        {"tweet_id": 4, "account_id": 10, "username": "up", "created_at": "2020-06-01 00:00:00+00",
         "full_text": "   ", "favorite_count": 1, "retweet_count": 0,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": 99},
        # reply by uploader -> kept, is_reply True
        {"tweet_id": 5, "account_id": 10, "username": "up", "created_at": "2020-07-01 00:00:00+00",
         "full_text": "a reply", "favorite_count": 2, "retweet_count": 0,
         "reply_to_tweet_id": 1, "quoted_tweet_id": None, "archive_upload_id": None},
    ])
    out = clean.clean_corpus(df)
    ids = set(out["tweet_id"].to_list())
    assert ids == {1, 3, 5}  # dup(1), RT(2), empty(4) removed
    assert "era" in out.columns and "created_dt" in out.columns
    row1 = out.filter(pl.col("tweet_id") == 1)
    assert row1["era"].item() == 2020
    assert isinstance(row1["created_dt"].item(), datetime)
    # account 10 uploaded an archive -> ALL its tweets (incl. the reply) are uploader rows
    assert row1["is_uploader"].item() is True
    assert out.filter(pl.col("tweet_id") == 5)["is_uploader"].item() is True
    assert out.filter(pl.col("tweet_id") == 5)["is_reply"].item() is True
    # account 77 never uploaded -> not an uploader
    assert out.filter(pl.col("tweet_id") == 3)["is_uploader"].item() is False


def test_clean_strips_urls_unescapes_entities_and_flags_links():
    df = _raw([
        {"tweet_id": 1, "account_id": 10, "username": "u", "created_at": "2020-01-01 00:00:00+00",
         "full_text": "great read &amp; worth it https://t.co/abc123", "favorite_count": 1, "retweet_count": 0,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": 1},
        {"tweet_id": 2, "account_id": 10, "username": "u", "created_at": "2020-01-02 00:00:00+00",
         "full_text": "no link here", "favorite_count": 1, "retweet_count": 0,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": 1},
    ])
    out = clean.clean_corpus(df)
    r1 = out.filter(pl.col("tweet_id") == 1)
    assert "http" not in r1["text_clean"].item()
    assert "&amp;" not in r1["text_clean"].item() and "&" in r1["text_clean"].item()
    assert r1["has_link"].item() is True
    assert r1["text_clean"].item() == "great read & worth it"
    assert out.filter(pl.col("tweet_id") == 2)["has_link"].item() is False


def test_uploader_ids_are_accounts_with_any_upload():
    df = _raw([
        {"tweet_id": 1, "account_id": 10, "username": "u", "created_at": "2020-01-01 00:00:00+00",
         "full_text": "x", "favorite_count": 0, "retweet_count": 0,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": 5},
        {"tweet_id": 2, "account_id": 10, "username": "u", "created_at": "2020-01-02 00:00:00+00",
         "full_text": "y", "favorite_count": 0, "retweet_count": 0,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": None},
        {"tweet_id": 3, "account_id": 20, "username": "v", "created_at": "2020-01-03 00:00:00+00",
         "full_text": "z", "favorite_count": 0, "retweet_count": 0,
         "reply_to_tweet_id": None, "quoted_tweet_id": None, "archive_upload_id": None},
    ])
    up = set(clean.uploader_ids(df).to_list())
    assert up == {10}


# ----------------------------- labels -------------------------------------

def test_engagement_score_log1p_weighted():
    df = pl.DataFrame({"favorite_count": [0, 9, 99], "retweet_count": [0, 0, 1]})
    out = labels.add_engagement(df, rt_weight=0.5)
    e = out["e"].to_list()
    assert e[0] == pytest.approx(0.0)
    assert e[1] == pytest.approx(math.log1p(9))
    assert e[2] == pytest.approx(math.log1p(99) + 0.5 * math.log1p(1))


def test_author_era_z_is_zero_mean_within_cell_and_monotone():
    # one author, one era, increasing engagement -> z increasing, ~zero mean
    df = pl.DataFrame({
        "account_id": [1, 1, 1, 1, 1],
        "era": [2020] * 5,
        "e": [0.0, 1.0, 2.0, 3.0, 4.0],
    })
    out = labels.add_author_era_z(df).sort("e")
    z = out["z"].to_list()
    assert z == sorted(z)  # monotone in e
    assert sum(z) / len(z) == pytest.approx(0.0, abs=1e-6)


def test_author_era_z_shrinks_thin_era_cell_toward_author_mean():
    # era 2020: ten low tweets; era 2021: a single big hit.
    df = pl.DataFrame({
        "account_id": [1] * 11,
        "era": [2020] * 10 + [2021],
        "e": [0.0] * 10 + [10.0],
    })
    out = labels.add_author_era_z(df)
    z_hit = out.filter(pl.col("era") == 2021)["z"].item()
    z_low = out.filter(pl.col("era") == 2020)["z"].max()
    # the lone hit beat its (shrunk) era baseline -> positive and the top score
    assert z_hit > 0
    assert z_hit > z_low


def test_author_era_z_handles_singleton_author_without_crash():
    df = pl.DataFrame({"account_id": [1], "era": [2020], "e": [3.0]})
    out = labels.add_author_era_z(df)
    assert out["z"].item() is not None
    assert math.isfinite(out["z"].item())


# ----------------------------- pairs --------------------------------------

def _labelled(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"tweet_id": pl.Int64, "account_id": pl.Int64, "era": pl.Int64,
                "full_text": pl.Utf8, "favorite_count": pl.Int64, "e": pl.Float64,
                "z": pl.Float64, "is_reply": pl.Boolean},
    )


def test_mine_pairs_respects_margins_same_author_and_winner_beats_loser():
    rows = [
        {"tweet_id": 1, "account_id": 1, "era": 2020, "full_text": "hit", "favorite_count": 100,
         "e": 4.6, "z": 2.0, "is_reply": False},
        {"tweet_id": 2, "account_id": 1, "era": 2020, "full_text": "meh", "favorite_count": 2,
         "e": 1.1, "z": -1.0, "is_reply": False},
        {"tweet_id": 3, "account_id": 1, "era": 2020, "full_text": "mid", "favorite_count": 90,
         "e": 4.5, "z": 1.8, "is_reply": False},  # vs #1 no margin -> not paired together
    ]
    pr = pairs.mine_pairs(_labelled(rows), min_ratio=2.0, min_abs_gap=5.0, min_z_gap=0.75,
                          max_pairs_per_author=10)
    assert pr.height >= 1
    for r in pr.iter_rows(named=True):
        assert r["account_id_w"] == r["account_id_l"]  # same author
        assert r["favorite_count_w"] > r["favorite_count_l"]
        assert r["favorite_count_w"] >= 2.0 * r["favorite_count_l"]
        assert (r["z_w"] - r["z_l"]) >= 0.75


def test_mine_pairs_returns_empty_when_no_margin():
    rows = [
        {"tweet_id": 1, "account_id": 1, "era": 2020, "full_text": "a", "favorite_count": 10,
         "e": 2.4, "z": 0.1, "is_reply": False},
        {"tweet_id": 2, "account_id": 1, "era": 2020, "full_text": "b", "favorite_count": 9,
         "e": 2.3, "z": 0.0, "is_reply": False},
    ]
    pr = pairs.mine_pairs(_labelled(rows), min_ratio=2.0, min_abs_gap=5.0, min_z_gap=0.75)
    assert pr.height == 0


def test_mine_pairs_min_winner_fav_floor():
    rows = [
        {"tweet_id": 1, "account_id": 1, "era": 2020, "full_text": "low hit", "favorite_count": 8,
         "e": 2.2, "z": 2.0, "is_reply": False},
        {"tweet_id": 2, "account_id": 1, "era": 2020, "full_text": "zero", "favorite_count": 0,
         "e": 0.0, "z": -1.0, "is_reply": False},
    ]
    # winner has only 8 favs; floor of 10 should reject the pair as too-noisy
    pr = pairs.mine_pairs(_labelled(rows), min_ratio=2.0, min_abs_gap=5.0, min_z_gap=0.75, min_winner_fav=10)
    assert pr.height == 0


def test_mine_topic_pairs_prefers_topically_matched_loser():
    import numpy as np

    # author 1: winner W; two eligible losers — "near" (topically similar to W) and "far".
    # The pair should match W with the NEAR loser, not the far one.
    df = pl.DataFrame(
        {"tweet_id": [1, 2, 3], "account_id": [1, 1, 1], "era": [2020, 2020, 2020],
         "text_clean": ["w", "near", "far"], "favorite_count": [100, 2, 1],
         "e": [4.6, 1.1, 0.7], "z": [2.0, -1.0, -1.0]},
        schema_overrides={"favorite_count": pl.Int64},
    )
    emb = np.array([[1.0, 0.0], [0.98, 0.2], [0.0, 1.0]])
    emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    pr = pairs.mine_topic_pairs(df, emb, min_topic_cos=0.5, min_ratio=2.0, min_abs_gap=5.0,
                                min_z_gap=0.75, min_winner_fav=10)
    assert pr.height == 1
    r = pr.row(0, named=True)
    assert r["tweet_id_w"] == 1 and r["tweet_id_l"] == 2


def test_mine_topic_pairs_empty_when_off_topic():
    import numpy as np

    df = pl.DataFrame(
        {"tweet_id": [1, 2], "account_id": [1, 1], "era": [2020, 2020],
         "text_clean": ["w", "l"], "favorite_count": [100, 2], "e": [4.6, 1.1], "z": [2.0, -1.0]},
        schema_overrides={"favorite_count": pl.Int64},
    )
    emb = np.array([[1.0, 0.0], [0.0, 1.0]])  # orthogonal -> cos 0 < 0.5
    pr = pairs.mine_topic_pairs(df, emb, min_topic_cos=0.5)
    assert pr.height == 0


def test_mine_pairs_caps_per_author():
    rows = [{"tweet_id": i, "account_id": 1, "era": 2020, "full_text": f"t{i}",
             "favorite_count": (1000 if i < 20 else 1), "e": (6.0 if i < 20 else 0.0),
             "z": (3.0 if i < 20 else -1.0), "is_reply": False} for i in range(40)]
    pr = pairs.mine_pairs(_labelled(rows), max_pairs_per_author=7)
    assert pr.height <= 7


# ----------------------------- splits -------------------------------------

def _dated(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows, schema={"tweet_id": pl.Int64, "account_id": pl.Int64, "created_dt": pl.Datetime}
    )


def test_temporal_split_respects_cutoff_and_maturity():
    df = _dated([
        {"tweet_id": 1, "account_id": 1, "created_dt": datetime(2024, 1, 1)},   # train
        {"tweet_id": 2, "account_id": 1, "created_dt": datetime(2025, 8, 1)},   # test
        {"tweet_id": 3, "account_id": 1, "created_dt": datetime(2026, 2, 1)},   # immature -> dropped
    ])
    train, test = splits.temporal_split(df, cutoff=datetime(2025, 6, 1), maturity_cutoff=datetime(2026, 1, 17))
    assert set(train["tweet_id"]) == {1}
    assert set(test["tweet_id"]) == {2}


def test_author_holdout_is_disjoint_and_leakage_check_raises():
    df = pl.DataFrame({"tweet_id": [1, 2, 3, 4], "account_id": [1, 1, 2, 3]})
    train, test = splits.author_holdout_split(df, holdout_ids={2, 3})
    assert set(test["account_id"]) == {2, 3}
    assert set(train["account_id"]) == {1}
    with pytest.raises(AssertionError):
        splits.assert_no_author_leakage(df, df)  # full overlap -> must raise
