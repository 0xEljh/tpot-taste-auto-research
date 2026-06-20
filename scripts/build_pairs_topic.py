"""Build topic-controlled same-author pairs (Scorer v2, D12).

For each split: pick top/bottom-z candidates per author, embed (MiniLM), then match
each high-z winner to its most topically-similar low-z loser. Writes *_pairs_topic.parquet.

  uv run python scripts/build_pairs_topic.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

SPLITS = {
    "train": "train_tweets.parquet",
    "test": "test_temporal_tweets.parquet",
    "test_unseen": "test_unseen_authors_tweets.parquet",
}


@app.command()
def main(
    splits_dir: Path = Path("data/splits"),
    cand_per_side: int = 300,
    min_topic_cos: float = 0.5,
    max_pairs_per_author: int = 150,
) -> None:
    from tpot_taste.data import embed as E
    from tpot_taste.data import pairs as P

    model, tok = E.load_embedder()
    for name, fn in SPLITS.items():
        df = pl.read_parquet(splits_dir / fn)
        top = df.sort("z", descending=True).group_by("account_id", maintain_order=True).head(cand_per_side)
        bot = df.sort("z", descending=False).group_by("account_id", maintain_order=True).head(cand_per_side)
        cand = pl.concat([top, bot]).unique(subset=["tweet_id"], keep="first")
        emb = E.embed_texts(model, tok, cand["text_clean"].to_list())
        pr = P.mine_topic_pairs(cand, emb, min_topic_cos=min_topic_cos, max_pairs_per_author=max_pairs_per_author)
        out = splits_dir / f"{name}_pairs_topic.parquet"
        pr.write_parquet(out)
        if pr.height:
            cos = pr["topic_cos"]
            print(f"  {name:11s}: {pr.height:,} pairs | authors={pr['account_id_w'].n_unique()} | "
                  f"topic_cos mean={float(cos.mean()):.3f} min={float(cos.min()):.3f} | cand={cand.height:,}")
        else:
            print(f"  {name:11s}: 0 pairs | cand={cand.height:,}")
    print("[done]")


if __name__ == "__main__":
    app()
