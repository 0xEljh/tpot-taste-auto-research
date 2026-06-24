"""Build Writer SFT data (Phase 3): ideate (curated goods) + improve (deopt pairs).

The deopt pairs already contain the curated good tweets as `text_clean_w`, so ideate goods =
unique chosen, and improve = (rejected -> chosen). No new generation. Leakage-guarded against
the held-out/unseen eval tweet sets.

  uv run python scripts/build_writer_sft.py
Writes data/splits/writer_sft_train.jsonl (chat-message records for TRL SFTTrainer).
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer

from tpot_taste.writer import sft_data

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    pairs_path: Path = Path("data/splits/deopt_v6_train_pairs.parquet"),
    out: Path = Path("data/splits/writer_sft_train.jsonl"),
    splits_dir: Path = Path("data/splits"),
    goods_path: Path = typer.Option(None, help="ideate goods source; default = pairs' chosen.unique()"),
    n_ideate: int = 10000,
    n_improve: int = 10000,
    seed: int = 0,
) -> None:
    pairs = pl.read_parquet(pairs_path)
    # ideate goods: a v8-curated clean set (Phase 7, scripts/curate_goods.py) if given, else pairs' chosen.
    goods = (pl.read_parquet(goods_path)["text_clean"].to_list() if goods_path
             else pairs["text_clean_w"].unique().to_list())

    forbidden: set[str] = set()
    for name in ["test_temporal_tweets", "test_unseen_authors_tweets"]:
        p = splits_dir / f"{name}.parquet"
        if p.exists():
            forbidden |= set(pl.read_parquet(p)["text_clean"].to_list())

    # make_sft_records drops engagement-bait from BOTH ideate goods and improve targets (is_baity) — the
    # improve pairs' `chosen` come from the raw engagement-selected pool (e.g. "$5,000 giveaway, RT to win").
    recs = sft_data.make_sft_records(
        goods, pairs, n_ideate=n_ideate, n_improve=n_improve, seed=seed, forbidden=forbidden
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    by: dict[str, int] = {}
    for r in recs:
        by[r["task"]] = by.get(r["task"], 0) + 1
    print(f"[write] {out} : {len(recs):,} records  {by}")
    print(f"  unique goods={len(goods):,}  pairs={pairs.height:,}  forbidden(eval)={len(forbidden):,}")
    for r in recs[:2] + recs[-1:]:
        print(f"  --- task={r['task']}")
        for m in r["messages"]:
            print(f"    {m['role']:9s}: {m['content'][:88]}")


if __name__ == "__main__":
    app()
