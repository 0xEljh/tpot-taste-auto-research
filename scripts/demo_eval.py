"""Demonstrative dipstick (Phase 6b, D28) — human qualitative sensing of the Scorer + Writer.

Three sections, all scored by the locked v6 Taste Scorer:
  1. ARCHETYPE SPECTRUM — the fixed hand-authored panel (tpot/aphorism vs platitude/promo/bait/...).
     The key test: does the scorer keep taste ABOVE generic-virality, or blur the boundary (D27/D28)?
  2. REAL GOODS at score levels — top/median/bottom of a sample of the actual training goods, so we
     see what the scorer loves/hates in real data (and where promo/platitude creep ranks).
  3. WRITER panel — fixed ideate topics + improve drafts (best-of-N), incl. deliberate platitude traps.

Writes a monospace report to `outputs/demo_eval.md` (push to Notion with `notion-cat`). Re-run after every
model iteration and diff — the panel is a constant ruler.

  uv run python scripts/demo_eval.py
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _sample_real_goods(splits_dir: Path, z_min: float, lo: int, hi: int, n: int, seed: int) -> list[str]:
    import random

    import polars as pl

    d = pl.read_parquet(splits_dir / "train_tweets.parquet")
    d = d.filter((pl.col("text_clean_len") >= lo) & (pl.col("text_clean_len") <= hi))
    d = d.filter(~pl.col("is_reply") & (pl.col("z") >= z_min))
    texts = d["text_clean"].to_list()
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:n]


@app.command()
def main(
    out: Path = Path("outputs/demo_eval.md"),
    splits_dir: Path = Path("data/splits"),
    writer: Path = Path("outputs/writer/qwen3b-writer-LOCKED"),
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    n_real: int = 600,
    best_of: int = 8,
    k: int = 3,
    seed: int = 0,
) -> None:
    import numpy as np

    from tpot_taste.engine import TasteEngine
    from tpot_taste.eval.demo_panel import (
        EXPECTED_HIGH,
        EXPECTED_LOW,
        IDEATE_TOPICS,
        IMPROVE_DRAFTS,
        PANEL,
    )

    eng = TasteEngine(writer=writer, scorer=scorer)
    L: list[str] = []  # report lines
    def w(s: str = "") -> None:
        L.append(s)

    w(f"# tpot-taste — demonstrative dipstick ({datetime.now():%Y-%m-%d %H:%M})")
    w(f"Scorer: `{scorer.name}` (higher = more tpot) · Writer: `{writer.name}`. For human qualitative sensing.")
    w("Re-run `scripts/demo_eval.py` after each model iteration and diff. See decision-log D28.")
    w("")

    # ---- 1. archetype spectrum ----
    w("## 1. Scorer — archetype spectrum")
    w("Does the scorer rank taste (tpot_canon, aphorism) ABOVE generic-virality (platitude, promo, ...)?")
    w("")
    cat_scores: dict[str, np.ndarray] = {}
    for cat, texts in PANEL.items():
        cat_scores[cat] = np.asarray(eng.score(texts))
    w("```")
    w(f"{'category':16s} {'mean':>7s} {'min':>7s} {'max':>7s}  n  expected")
    for cat in PANEL:
        s = cat_scores[cat]
        exp = "HIGH" if cat in EXPECTED_HIGH else "low"
        note = "  <- watch: platitude creep" if cat == "generic_viral" else ("  <- the audit leak" if cat == "promo" else "")
        w(f"{cat:16s} {s.mean():+7.2f} {s.min():+7.2f} {s.max():+7.2f} {len(s):2d}  {exp}{note}")
    high = np.concatenate([cat_scores[c] for c in EXPECTED_HIGH])
    low = np.concatenate([cat_scores[c] for c in EXPECTED_LOW])
    worst_high = min(cat_scores[c].mean() for c in EXPECTED_HIGH)
    best_low = max(cat_scores[c].mean() for c in EXPECTED_LOW)
    clean = worst_high > best_low
    w("")
    w(f"HIGH probes mean {high.mean():+.2f}   LOW probes mean {low.mean():+.2f}   gap {high.mean()-low.mean():+.2f}")
    w(f"worst-HIGH category {worst_high:+.2f}  vs  best-LOW category {best_low:+.2f}  -> "
      f"{'CLEAN separation' if clean else 'BLURRED (a low category outranks a high one)'}")
    w("```")
    w("")
    w("Per-probe (within category, sorted desc):")
    w("```")
    for cat, texts in PANEL.items():
        s = cat_scores[cat]
        w(f"[{cat}]")
        for i in np.argsort(-s):
            w(f"  {s[i]:+6.2f}  {texts[i][:96]}")
    w("```")
    w("")

    # ---- 2. real goods at score levels ----
    w("## 2. Scorer — REAL training goods at score levels")
    w(f"Top/median/bottom of {n_real} sampled real goods (uploader z>=1.0). What does it actually love/hate?")
    w("")
    real = _sample_real_goods(splits_dir, 1.0, 40, 240, n_real, seed)
    rs = np.asarray(eng.score(real))
    order = np.argsort(-rs)
    w("```")
    for label, idxs in [("TOP-8", order[:8]), ("MEDIAN-8", order[len(order)//2 - 4: len(order)//2 + 4]),
                        ("BOTTOM-8", order[-8:][::-1])]:
        w(f"--- {label} ---")
        for i in idxs:
            w(f"  {rs[i]:+6.2f}  {real[i][:100]}")
    w("```")
    w("")

    # ---- 3. writer panel ----
    w("## 3. Writer — ideate (best-of-N, top candidates)")
    w("`discipline` is a deliberate platitude-bait topic — does the Writer drift motivational or stay specific?")
    w("")
    w("```")
    for topic in IDEATE_TOPICS:
        w(f"topic: {topic}")
        for text, sc in eng.ideate(topic, k=k, best_of=best_of):
            w(f"  {sc:+6.2f}  {text[:100]}")
        w("")
    w("```")
    w("")
    w("## 3b. Writer — improve (best-of-N)")
    w('`"work hard and you will succeed"` is a platitude draft — does `improve` de-platitude or amplify?')
    w("")
    w("```")
    for draft in IMPROVE_DRAFTS:
        w(f'draft: "{draft}"')
        for text, sc in eng.improve(draft, k=k, best_of=best_of):
            w(f"  {sc:+6.2f}  {text[:100]}")
        w("")
    w("```")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n")
    print(f"[write] {out}  ({len(L)} lines)")
    print(f"[spectrum] HIGH {high.mean():+.2f} vs LOW {low.mean():+.2f} (gap {high.mean()-low.mean():+.2f}); "
          f"{'CLEAN' if clean else 'BLURRED'} separation")


if __name__ == "__main__":
    app()
