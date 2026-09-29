"""Qualitative look at the Taste Scorer.

(1) Direction check: score hand-written probes (good tpot vs cringe/bait/low-effort).
(2) Show the highest- and lowest-scored held-out tweets.

  uv run python scripts/inspect_scorer.py
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# Direction probes — what the Scorer *should* rank high (insight/punch/earnest) vs
# low (engagement-bait/corporate/low-effort). Not ground truth, just a sniff test.
PROBES: list[tuple[str, str]] = [
    ("aphorism", "the best ideas feel obvious in retrospect and impossible in advance"),
    ("specific-insight", "spent 3 years thinking my problem was discipline. it was sleep. that's the whole post."),
    ("earnest-vulnerable", "i think 'i'm busy' has become how i avoid finding out what i'd actually do with free time"),
    ("curious-question", "what's a belief you held strongly at 22 that now embarrasses you, and what changed it?"),
    ("funny-punchy", "every codebase is a haunted house and you are the ghost"),
    ("engagement-bait", "RT if you agree, like if you don't. let the algorithm settle this once and for all 👇"),
    ("corporate-cringe", "Thrilled and humbled to share some thoughts on synergy, growth mindset, and crushing it 🚀 #blessed"),
    ("low-effort", "lol same"),
    ("rage-bait", "nobody wants to admit it but everyone who disagrees with me is simply not very smart"),
    ("generic-platitude", "work hard, stay positive, and good things will happen. believe in yourself!"),
]


def _short(s: str, n: int = 110) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 1] + "…"


@app.command()
def main(
    adapter: Path = Path("outputs/scorer/qwen3b-bt-v1"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    split: Path = Path("data/splits/test_temporal_tweets.parquet"),
    n: int = 15,
    sample: int = 4000,
    seed: int = 0,
) -> None:
    from tpot_taste.engine import resolve_adapter_base
    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    base = resolve_adapter_base(adapter, base)
    model, tok = load_trained_scorer(str(adapter), base)

    print("== direction check (hand-written probes, sorted by score) ==")
    ps = score_texts(model, tok, [t for _, t in PROBES], batch_size=16)
    for (lab, t), s in sorted(zip(PROBES, ps), key=lambda x: -x[1]):
        print(f"  {s:+6.2f}  [{lab:18s}] {_short(t, 80)}")

    df = pl.read_parquet(split)
    if df.height > sample:
        df = df.sample(sample, seed=seed)
    sc = score_texts(model, tok, df["text_clean"].to_list(), batch_size=64)
    df = df.with_columns(pl.Series("score", sc))

    import numpy as np
    lens = df["text_clean"].str.len_chars().to_numpy()
    corr = float(np.corrcoef(sc, lens)[0, 1])
    print(f"\n== score-vs-length confound check: pearson(score, chars) = {corr:+.3f} "
          f"(≈0 is good; strongly negative = 'shorter is better' bias) ==")

    print(f"\n== highest-scored held-out tweets (of {df.height:,} sampled) ==")
    for r in df.sort("score", descending=True).head(n).iter_rows(named=True):
        print(f"  {r['score']:+6.2f} ({r['favorite_count']:>5}♥ z={r['z']:+.1f}) {_short(r['text_clean'])}")
    print("\n== lowest-scored held-out tweets ==")
    for r in df.sort("score").head(n).iter_rows(named=True):
        print(f"  {r['score']:+6.2f} ({r['favorite_count']:>5}♥ z={r['z']:+.1f}) {_short(r['text_clean'])}")


if __name__ == "__main__":
    app()
