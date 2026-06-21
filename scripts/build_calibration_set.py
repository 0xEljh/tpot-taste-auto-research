"""Build a human-calibration set for the taste judge (Phase 6e, D36).

The judge's taste = base-model prior × OUR rubric, graded on OUR archetypes — the one thing we can't
bootstrap is whether it matches the *human's* taste. So: sample real tweets STRATIFIED across the judge's
score range (so we test it everywhere, not just the extremes), hand them to the human in Notion WITHOUT the
judge's score (no anchoring), collect labels, then measure judge↔human agreement + surface the judge's errors.

  uv run python scripts/build_calibration_set.py
Writes data/splits/calibration_set.parquet (id, text, judge_score) — judge_score is for the later join only.
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    pool: Path = Path("data/splits/taste_v7_pool_scored.parquet"),
    out: Path = Path("data/splits/calibration_set.parquet"),
    per_bin: int = 22,
    seed: int = 0,
) -> None:
    import random

    import polars as pl

    df = pl.read_parquet(pool).filter(pl.col("score").is_not_null())
    rng = random.Random(seed)
    # stratify by judge score so the human tests the judge across its whole range
    bins = [(-0.1, 2.0), (2.0, 4.0), (4.0, 6.0), (6.0, 8.0), (8.0, 10.1)]
    picked: list[dict] = []
    for lo, hi in bins:
        rows = df.filter((pl.col("score") >= lo) & (pl.col("score") < hi)).to_dicts()
        rng.shuffle(rows)
        for r in rows[:per_bin]:
            picked.append({"text": r["text"], "judge_score": float(r["score"])})
    rng.shuffle(picked)  # randomize order so the user can't infer the score from position
    for i, r in enumerate(picked):
        r["id"] = i + 1
    pl.DataFrame(picked).select(["id", "text", "judge_score"]).write_parquet(out)
    print(f"[write] {out}: {len(picked)} items across {len(bins)} judge-score bins (~{per_bin}/bin)")
    import numpy as np
    s = np.array([r["judge_score"] for r in picked])
    print(f"  judge score spread: min {s.min():.1f} p50 {np.percentile(s,50):.1f} max {s.max():.1f}")
    print("  (judge_score is hidden from the human; used only to score agreement afterward)")


if __name__ == "__main__":
    app()
