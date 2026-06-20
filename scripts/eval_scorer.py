"""Evaluate the Taste Scorer: pairwise accuracy + per-author rank correlation.

  uv run python scripts/eval_scorer.py
Writes outputs/scorer/eval.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    adapter: Path = Path("outputs/scorer/qwen3b-bt-v1"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    splits_dir: Path = Path("data/splits"),
    out: Path = Path("outputs/scorer/eval.json"),
    pairs: str = "test_pairs,test_unseen_pairs",
    tweets: bool = True,
    max_eval_tweets: int = 8000,
    batch_size: int = 48,
) -> None:
    from tpot_taste.scoring import metrics
    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    model, tok = load_trained_scorer(str(adapter), base)
    results: dict[str, float] = {}

    print("== pairwise accuracy (the primary, confound-robust metric) ==")
    for name in [p for p in pairs.split(",") if p]:
        df = pl.read_parquet(splits_dir / f"{name}.parquet")
        sc = score_texts(model, tok, df["text_clean_w"].to_list(), batch_size=batch_size)
        sr = score_texts(model, tok, df["text_clean_l"].to_list(), batch_size=batch_size)
        acc = metrics.pairwise_accuracy(sc, sr)
        results[f"{name}_pairwise_acc"] = acc
        results[f"{name}_n"] = df.height
        print(f"  {name:22s} acc={acc:.4f}  (n={df.height:,})")

    print("== per-author Spearman (score vs within-author z / raw favourites) ==" if tweets else "(skipping tweet Spearman)")
    for name in (["test_temporal_tweets", "test_unseen_authors_tweets"] if tweets else []):
        df = pl.read_parquet(splits_dir / f"{name}.parquet")
        if df.height > max_eval_tweets:
            df = df.sample(max_eval_tweets, seed=0)
        scores = score_texts(model, tok, df["text_clean"].to_list(), batch_size=batch_size)
        authors = df["account_id"].to_list()
        sp_z = metrics.per_author_spearman(authors, scores, df["z"].to_list(), min_items=10)
        sp_f = metrics.per_author_spearman(authors, scores, df["favorite_count"].to_list(), min_items=10)
        results[f"{name}_spearman_z"] = sp_z
        results[f"{name}_spearman_fav"] = sp_f
        print(f"  {name:28s} vs_z={sp_z:.4f}  vs_fav={sp_f:.4f}  (n={df.height:,})")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    app()
