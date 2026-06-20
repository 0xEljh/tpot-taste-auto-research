"""Build the Phase-1 modeling datasets from the raw corpus.

raw parquet -> clean -> {liked_taste, uploader_labeled} -> leakage-proof splits
-> same-author preference pairs (train / test_temporal / test_unseen_authors).

Usage:
    uv run python scripts/build_dataset.py
Outputs: data/processed/*.parquet, data/splits/*.parquet, data/processed/manifest.json
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import polars as pl
import typer

from tpot_taste.data import clean, labels, pairs, splits

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    raw: Path = Path("data/raw/community-archive.parquet"),
    out_processed: Path = Path("data/processed"),
    out_splits: Path = Path("data/splits"),
    cutoff: str = "2025-06-01",       # train < cutoff ; test_temporal >= cutoff
    maturity: str = "2026-01-17",     # drop tweets newer than this (engagement immature)
    n_holdout_authors: int = 30,
    max_pairs_per_author: int = 150,
    min_author_tweets: int = 50,      # min ORIGINAL tweets for a stable author baseline
    min_text_chars: int = 15,         # substantive text (URL-stripped) — cuts media-only tweets
    min_winner_fav: int = 10,         # absolute engagement floor — cuts 5-vs-0 noise
    seed: int = 0,
) -> None:
    cutoff_dt, maturity_dt = datetime.fromisoformat(cutoff), datetime.fromisoformat(maturity)
    out_processed.mkdir(parents=True, exist_ok=True)
    out_splits.mkdir(parents=True, exist_ok=True)

    print("[1/6] load + clean")
    cdf = clean.clean_corpus(pl.read_parquet(raw))
    print(f"  cleaned rows: {cdf.height:,}")

    liked = cdf.filter(~pl.col("is_uploader"))
    liked.write_parquet(out_processed / "liked_taste.parquet")
    print(f"  liked_taste (tpot-endorsed context): {liked.height:,}")

    print("[2/6] uploader modeling corpus = non-reply, substantive-text originals")
    U = cdf.filter(
        pl.col("is_uploader") & (~pl.col("is_reply")) & (pl.col("text_clean_len") >= min_text_chars)
    )
    counts = U.group_by("account_id").len()
    keep = counts.filter(pl.col("len") >= min_author_tweets)["account_id"].to_list()
    U = U.filter(pl.col("account_id").is_in(keep))
    print(f"  uploader originals (text>={min_text_chars}): {U.height:,} across {U['account_id'].n_unique()} authors (>= {min_author_tweets})")

    print("[3/6] labels: log1p engagement + within-(author,era) EB z-score")
    U = labels.add_author_era_z(labels.add_engagement(U))
    U.write_parquet(out_processed / "uploader_labeled.parquet")

    print("[4/6] splits (author-holdout then temporal)")
    holdout = splits.pick_holdout_authors(U, n_holdout_authors, seed=seed)
    U_seen, U_held = splits.author_holdout_split(U, holdout)
    train_tweets, test_temporal = splits.temporal_split(U_seen, cutoff=cutoff_dt, maturity_cutoff=maturity_dt)
    test_unseen = U_held.filter(pl.col("created_dt") < maturity_dt)
    splits.assert_no_author_leakage(train_tweets, test_unseen)
    for name, d in [("train_tweets", train_tweets), ("test_temporal_tweets", test_temporal),
                    ("test_unseen_authors_tweets", test_unseen)]:
        d.write_parquet(out_splits / f"{name}.parquet")
        print(f"  {name}: {d.height:,}")

    print("[5/6] mine same-author preference pairs")
    _pk = dict(text_col="text_clean", min_winner_fav=min_winner_fav, max_pairs_per_author=max_pairs_per_author)
    pair_sets = {
        "train_pairs": pairs.mine_pairs(train_tweets, **_pk),
        "test_pairs": pairs.mine_pairs(test_temporal, **_pk),
        "test_unseen_pairs": pairs.mine_pairs(test_unseen, **_pk),
    }
    for name, d in pair_sets.items():
        d.write_parquet(out_splits / f"{name}.parquet")
        print(f"  {name}: {d.height:,}")

    print("[6/6] manifest")
    manifest = {
        "raw": str(raw), "built_at_cutoff": cutoff, "maturity": maturity,
        "cleaned_rows": cdf.height, "liked_rows": liked.height,
        "uploader_originals": U.height, "uploader_authors": int(U["account_id"].n_unique()),
        "min_author_tweets": min_author_tweets, "min_text_chars": min_text_chars,
        "min_winner_fav": min_winner_fav, "max_pairs_per_author": max_pairs_per_author,
        "n_holdout_authors": n_holdout_authors, "holdout_author_ids": sorted(int(x) for x in holdout),
        "seed": seed,
        "counts": {"train_tweets": train_tweets.height, "test_temporal": test_temporal.height,
                   "test_unseen": test_unseen.height,
                   **{k: v.height for k, v in pair_sets.items()}},
    }
    (out_processed / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest["counts"], indent=2))
    print("[done]")


if __name__ == "__main__":
    app()
