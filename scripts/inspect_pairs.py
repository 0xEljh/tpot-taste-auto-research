"""Eyeball mined preference pairs — a qualitative gate before training the Scorer.

Usage: uv run python scripts/inspect_pairs.py --n 10
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _short(s: str, n: int = 200) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


@app.command()
def main(path: Path = Path("data/splits/train_pairs.parquet"), n: int = 10, seed: int = 0) -> None:
    df = pl.read_parquet(path)
    tw, tl = "text_clean_w", "text_clean_l"  # pairs carry URL-stripped model text
    print(f"pairs: {df.height:,}")
    print(f"winner fav > loser fav in all rows: {bool((df['favorite_count_w'] > df['favorite_count_l']).all())}")
    print(f"identical winner/loser text: {int((df[tw] == df[tl]).sum())}")
    print(
        f"fav_w: p50={df['favorite_count_w'].median():.0f} p90={df['favorite_count_w'].quantile(0.9):.0f} | "
        f"fav_l: p50={df['favorite_count_l'].median():.0f} p90={df['favorite_count_l'].quantile(0.9):.0f}"
    )
    print(f"z-gap (w-l): mean={float((df['z_w'] - df['z_l']).mean()):.2f} min={float((df['z_w'] - df['z_l']).min()):.2f}")
    print(f"distinct authors in pairs: {df['account_id_w'].n_unique()}\n")

    sample = df.sample(n=min(n, df.height), seed=seed)
    for i, r in enumerate(sample.iter_rows(named=True), 1):
        print(f"── pair {i} (author {r['account_id_w']}, era {r['era']}) "
              f"z-gap {r['z_w'] - r['z_l']:.2f} ──")
        print(f"  WIN  [{r['favorite_count_w']:>6} ♥]  {_short(r[tw])}")
        print(f"  LOSE [{r['favorite_count_l']:>6} ♥]  {_short(r[tl])}")
        print()


if __name__ == "__main__":
    app()
