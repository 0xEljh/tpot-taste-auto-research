"""Exploratory data analysis of the Community Archive corpus.

Informs Phase-1 label/normalization/pairing thresholds (min author history, pair
margins, dedup). Writes a markdown summary + plots to outputs/eda/.

Usage:
    uv run python scripts/eda.py
    uv run python scripts/eda.py --parquet data/raw/community-archive.parquet
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

QUANTILES = [0.5, 0.75, 0.9, 0.95, 0.99, 0.999]


def _has(df: pl.DataFrame, col: str) -> bool:
    return col in df.columns


def _engagement_stats(df: pl.DataFrame, col: str) -> str:
    s = df[col].drop_nulls()
    parts = [
        f"mean={float(s.mean()):.1f}",
        f"p50={float(s.quantile(0.5)):.0f}",
        f"p90={float(s.quantile(0.9)):.0f}",
        f"p99={float(s.quantile(0.99)):.0f}",
        f"max={float(s.max()):.0f}",
        f"pct_zero={float((s == 0).mean()):.1%}",
    ]
    return f"{col}: " + " ".join(parts)


@app.command()
def main(
    parquet: Path = Path("data/raw/community-archive.parquet"),
    out_dir: Path = Path("outputs/eda"),
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[read] {parquet}")
    df = pl.read_parquet(parquet)
    n = df.height
    out: list[str] = ["# Community Archive — EDA\n", f"- rows: **{n:,}**", f"- columns: {df.columns}\n"]

    def emit(line: str) -> None:
        out.append(line)
        print(line)

    print(f"rows={n:,}")

    # --- authors ---
    per_auth = (
        df.group_by("account_id")
        .agg(pl.col("username").first().alias("username"), pl.len().alias("n"))
        .sort("n", descending=True)
    )
    n_auth = per_auth.height
    emit(f"## authors\n- unique authors: **{n_auth:,}**  (avg {n / n_auth:.1f} tweets/author)")
    emit(
        f"- tweets/author: median={per_auth['n'].median():.0f} mean={per_auth['n'].mean():.0f} "
        f"p90={per_auth['n'].quantile(0.9):.0f} max={per_auth['n'].max():,}"
    )
    for thr in (20, 50, 100, 200, 500, 1000):
        k = int((per_auth["n"] >= thr).sum())
        emit(f"  - authors with ≥{thr:>4} tweets: {k:,}")
    # cumulative tweet share of the most prolific authors
    cum = per_auth["n"].cum_sum()
    for topk in (300, 1000, 5000):
        if topk <= n_auth:
            emit(f"  - top {topk:>5} authors hold {float(cum[topk - 1]) / n * 100:.1f}% of all tweets")
    emit("- top 15 authors by volume: " + ", ".join(
        f"{u}({c:,})" for u, c in zip(per_auth['username'][:15].to_list(), per_auth['n'][:15].to_list())
    ))

    # --- engagement ---
    emit("\n## engagement")
    for col in ("favorite_count", "retweet_count"):
        if _has(df, col):
            emit("- " + _engagement_stats(df, col))

    # --- structure ---
    emit("\n## structure")
    fr = lambda c: float(df[c].is_not_null().mean()) if _has(df, c) else float("nan")
    emit(f"- replies (reply_to_tweet_id not null): {fr('reply_to_tweet_id'):.1%}")
    emit(f"- quotes (quoted_tweet_id not null): {fr('quoted_tweet_id'):.1%}")
    if _has(df, "full_text"):
        is_rt = df["full_text"].str.starts_with("RT @")
        emit(f"- pure retweets (text starts 'RT @'): {float(is_rt.mean()):.1%}")
        has_url = df["full_text"].str.contains("http")
        emit(f"- contains URL: {float(has_url.mean()):.1%}")
        clen = df["full_text"].str.len_chars()
        wlen = df["full_text"].str.split(" ").list.len()
        emit(f"- text chars: p50={clen.median():.0f} p90={clen.quantile(0.9):.0f} max={clen.max()}")
        emit(f"- text words: p50={wlen.median():.0f} p90={wlen.quantile(0.9):.0f}")

    # --- time ---
    emit("\n## time")
    try:
        dt = df["created_at"].str.slice(0, 19).str.to_datetime("%Y-%m-%d %H:%M:%S", strict=False)
        ok = float(dt.is_not_null().mean())
        emit(f"- created_at parse rate: {ok:.1%}")
        if ok > 0:
            emit(f"- date range: {dt.min()} → {dt.max()}")
            by_year = pl.DataFrame({"year": dt.dt.year()}).drop_nulls().group_by("year").len().sort("year")
            emit("- tweets by year: " + ", ".join(f"{y}:{c // 1000}k" for y, c in zip(by_year['year'], by_year['len'])))
    except Exception as e:  # noqa: BLE001
        emit(f"- created_at parse FAILED: {e!r}")
        dt = None

    # --- nulls ---
    nulls = {c: int(df[c].null_count()) for c in df.columns if df[c].null_count()}
    emit(f"\n## nulls\n- {nulls}")

    # --- plots ---
    fav = df["favorite_count"].drop_nulls().to_numpy()
    plt.figure(figsize=(6, 4)); plt.hist(np.log1p(fav), bins=60)
    plt.xlabel("log1p(favorite_count)"); plt.ylabel("tweets"); plt.title("Favorite count (log1p)")
    plt.tight_layout(); plt.savefig(out_dir / "favorite_log1p_hist.png", dpi=110); plt.close()

    counts = per_auth["n"].to_numpy()
    plt.figure(figsize=(6, 4)); plt.hist(np.log10(counts), bins=40)
    plt.xlabel("log10(tweets per author)"); plt.ylabel("authors"); plt.title("Tweets per author")
    plt.tight_layout(); plt.savefig(out_dir / "tweets_per_author_hist.png", dpi=110); plt.close()

    agg = (
        df.group_by("account_id")
        .agg(pl.col("favorite_count").median().alias("med_fav"), pl.len().alias("n"))
        .filter(pl.col("n") >= 50)
    )
    plt.figure(figsize=(6, 4))
    plt.scatter(np.log10(agg["n"].to_numpy()), np.log1p(agg["med_fav"].to_numpy()), s=6, alpha=0.35)
    plt.xlabel("log10(author tweet count)"); plt.ylabel("log1p(author median favorite)")
    plt.title("Per-author baseline engagement (the confound)")
    plt.tight_layout(); plt.savefig(out_dir / "author_engagement_scatter.png", dpi=110); plt.close()

    (out_dir / "summary.md").write_text("\n".join(out) + "\n")
    print(f"\n[done] wrote {out_dir}/summary.md + 3 plots")


if __name__ == "__main__":
    app()
