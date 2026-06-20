"""Download the Community Archive tpot tweet corpus (HF mirror) to data/raw/.

Primary source: https://huggingface.co/datasets/Rabrg/community-archive
~11.5M tweets with native favorite_count / retweet_count / created_at / author.

Usage:
    uv run python scripts/download_data.py
    uv run python scripts/download_data.py --repo-id Rabrg/community-archive

Idempotent: skips if the output parquet already exists (use --force to re-pull).
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    repo_id: str = "Rabrg/community-archive",
    out: Path = Path("data/raw/community-archive.parquet"),
    force: bool = False,
) -> None:
    from datasets import load_dataset

    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists() and not force:
        print(f"[skip] {out} already exists ({out.stat().st_size / 1e6:.1f} MB). Use --force to re-pull.")
        return

    print(f"[load] load_dataset({repo_id!r}, split='train') ...")
    ds = load_dataset(repo_id, split="train")

    print("\n[schema] features:")
    for name, feat in ds.features.items():
        print(f"  {name}: {feat}")
    print(f"\n[rows] {len(ds):,}")
    print("[sample] first row:")
    sample = ds[0]
    for k, v in sample.items():
        sval = str(v)
        if len(sval) > 120:
            sval = sval[:117] + "..."
        print(f"  {k} = {sval}")

    print(f"\n[write] -> {out}")
    ds.to_parquet(str(out))
    print(f"[done] {out} ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    app()
