"""Clean the raw Community Archive corpus into a typed, deduped working frame.

See docs/design/02-data-findings.md. Key moves:
  * dedup on tweet_id
  * drop pure retweets (`RT @…`) and empty text
  * parse created_at (slice 19 chars; polars can't infer the `+00` offset)
  * mark uploaders (accounts with ANY non-null archive_upload_id → full timelines)
  * derive era (year), is_reply, is_pure_rt
"""
from __future__ import annotations

import polars as pl

_URL_RE = r"https?://\S+"
_ENTITIES = {"&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"', "&#39;": "'"}


def _text_clean_expr(col: str = "full_text") -> pl.Expr:
    """URL-stripped, HTML-unescaped, whitespace-collapsed text for the model.

    URLs carry no wording signal and would let a scorer cheat on "has link"; media
    tweets (URL-only) collapse to short/empty text and are filtered downstream.
    """
    e = pl.col(col)
    for k, v in _ENTITIES.items():
        e = e.str.replace_all(k, v, literal=True)
    return e.str.replace_all(_URL_RE, "").str.replace_all(r"\s+", " ").str.strip_chars()


def uploader_ids(df: pl.DataFrame) -> pl.Series:
    """account_ids that uploaded an archive (any non-null archive_upload_id).

    Those accounts' *entire* timeline is present, so they are the only unbiased
    corpus for within-author normalization and same-author pairing.
    """
    return (
        df.filter(pl.col("archive_upload_id").is_not_null())
        .select(pl.col("account_id").unique())
        .to_series()
    )


def clean_corpus(df: pl.DataFrame) -> pl.DataFrame:
    up = uploader_ids(df).to_list()

    out = df.unique(subset=["tweet_id"], keep="first", maintain_order=True).with_columns(
        pl.col("created_at").str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False).alias("created_dt"),
        pl.col("full_text").str.strip_chars().alias("text"),
        _text_clean_expr().alias("text_clean"),
        pl.col("full_text").str.starts_with("RT @").alias("is_pure_rt"),
        pl.col("full_text").str.contains("http").alias("has_link"),
        pl.col("reply_to_tweet_id").is_not_null().alias("is_reply"),
        pl.col("account_id").is_in(up).alias("is_uploader"),
    )
    out = out.with_columns(
        pl.col("created_dt").dt.year().alias("era"),
        pl.col("text_clean").str.len_chars().alias("text_clean_len"),
    )

    return out.filter(
        (~pl.col("is_pure_rt"))
        & pl.col("text").is_not_null()
        & (pl.col("text").str.len_chars() > 0)
        & pl.col("created_dt").is_not_null()
    )
