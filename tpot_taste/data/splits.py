"""Leakage-proof splits: temporal (predict the future) + author-disjoint (unseen voices).

See docs/design/01-system-design.md §1. Temporal split also drops immature tweets
(newer than the maturity cutoff) whose engagement hasn't saturated.
"""
from __future__ import annotations

import random
from datetime import datetime

import polars as pl


def temporal_split(
    df: pl.DataFrame,
    *,
    cutoff: datetime,
    maturity_cutoff: datetime,
    time_col: str = "created_dt",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    train = df.filter(pl.col(time_col) < cutoff)
    test = df.filter((pl.col(time_col) >= cutoff) & (pl.col(time_col) < maturity_cutoff))
    return train, test


def author_holdout_split(
    df: pl.DataFrame,
    holdout_ids: set[int] | list[int],
    *,
    author_col: str = "account_id",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    hold = list(holdout_ids)
    test = df.filter(pl.col(author_col).is_in(hold))
    train = df.filter(~pl.col(author_col).is_in(hold))
    return train, test


def assert_no_author_leakage(
    train: pl.DataFrame, test: pl.DataFrame, *, author_col: str = "account_id"
) -> None:
    overlap = set(train[author_col].to_list()) & set(test[author_col].to_list())
    assert not overlap, f"author leakage: {len(overlap)} authors in both splits"


def pick_holdout_authors(
    df: pl.DataFrame, n: int, *, author_col: str = "account_id", seed: int = 0
) -> set[int]:
    ids = df[author_col].unique().sort().to_list()
    random.Random(seed).shuffle(ids)
    return set(ids[:n])
