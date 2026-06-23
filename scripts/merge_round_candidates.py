"""Merge an active-learning round's candidate texts into the calibration set (doc 05 §8).

The labels live in Notion (pulled to outputs/calibration_labels.json by id), but the *text* for a new
round (e.g. round-3 ids 141-180) lives only in its candidates parquet. build_human_pairs / the bake-off /
eval all join label->text via calibration_set.parquet, so a round's texts must be folded in once the
human has labeled them. Idempotent: ids already present are skipped. judge_score is null for merged rows
(they were selected by the embed+Ridge / committee probes, not the judge-score bins — and nothing in the
loop-close or bake-off reads judge_score).

  uv run python scripts/merge_round_candidates.py                                   # default: round-3
  uv run python scripts/merge_round_candidates.py --cands data/splits/round4_candidates.parquet
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    calib: Path = Path("data/splits/calibration_set.parquet"),
    cands: Path = Path("data/splits/round3_candidates.parquet"),
) -> None:
    import polars as pl

    base = pl.read_parquet(calib)
    new = pl.read_parquet(cands)
    have = set(base["id"].to_list())
    add = new.filter(~pl.col("id").is_in(list(have))).select(
        pl.col("id"), pl.col("text"), pl.lit(None, dtype=base.schema["judge_score"]).alias("judge_score")
    )
    if add.height == 0:
        print(f"[merge] nothing to add — all {new.height} candidate ids already in {calib} ({base.height} rows)")
        return
    out = pl.concat([base.select("id", "text", "judge_score"), add]).sort("id")
    out.write_parquet(calib)
    print(f"[merge] +{add.height} rows (ids {add['id'].min()}-{add['id'].max()}) -> {calib} "
          f"now {out.height} rows (was {base.height})")


if __name__ == "__main__":
    app()
