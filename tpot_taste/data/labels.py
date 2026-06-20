"""Engagement labels: log1p engagement + within-(author, era) EB-shrunk z-score.

Raw likes are confounded by follower count and tweet age (2008→2026 span). We:
  1. e = log1p(fav) + rt_weight·log1p(rt)            (favorites primary; RT sparse)
  2. z = (e − μ*) / scale, where μ* is the author's era baseline shrunk toward the
     author's global mean (empirical Bayes), so a single hit in a thin era cell still
     scores high; scale is the author's overall std (floored), so z is "how much this
     tweet beat the author's typical tweet that year".
"""
from __future__ import annotations

import polars as pl


def add_engagement(
    df: pl.DataFrame,
    *,
    rt_weight: float = 0.5,
    fav_col: str = "favorite_count",
    rt_col: str = "retweet_count",
    out: str = "e",
) -> pl.DataFrame:
    return df.with_columns(
        (pl.col(fav_col).cast(pl.Float64).log1p() + rt_weight * pl.col(rt_col).cast(pl.Float64).log1p()).alias(out)
    )


def add_author_era_z(
    df: pl.DataFrame,
    *,
    author_col: str = "account_id",
    era_col: str = "era",
    score_col: str = "e",
    k_cell: float = 10.0,
    sigma_floor: float = 0.25,
    out: str = "z",
) -> pl.DataFrame:
    auth = df.group_by(author_col).agg(
        pl.col(score_col).mean().alias("_a_mu"),
        pl.col(score_col).std().alias("_a_sd"),
    )
    cell = df.group_by([author_col, era_col]).agg(
        pl.col(score_col).mean().alias("_c_mu"),
        pl.len().alias("_c_n"),
    )
    res = (
        df.join(auth, on=author_col, how="left")
        .join(cell, on=[author_col, era_col], how="left")
        .with_columns(
            (
                (pl.col("_c_n") * pl.col("_c_mu") + k_cell * pl.col("_a_mu"))
                / (pl.col("_c_n") + k_cell)
            ).alias("_mu_shrunk")
        )
        .with_columns(
            pl.when(pl.col("_a_sd").is_null() | (pl.col("_a_sd") < sigma_floor))
            .then(pl.lit(sigma_floor))
            .otherwise(pl.col("_a_sd"))
            .alias("_scale")
        )
        .with_columns(((pl.col(score_col) - pl.col("_mu_shrunk")) / pl.col("_scale")).alias(out))
    )
    return res.drop("_a_mu", "_a_sd", "_c_mu", "_c_n", "_mu_shrunk", "_scale")
