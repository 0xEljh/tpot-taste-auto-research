"""Score the dipstick panel with a given BT scorer — the fast v6-vs-v7 before/after meter (Phase 6c).

No writer / no generation: just the scorer's category means on the fixed archetype panel (the same §1 the
full dipstick uses). The pass test: worst-HIGH > best-LOW (taste above generic-virality). v6 FAILS this
(generic_viral is its top category); v7 should pass if it distilled the judge.

  uv run python scripts/score_panel.py --scorer outputs/scorer/qwen3b-bt-taste-LOCKED        # v6
  uv run python scripts/score_panel.py --scorer outputs/scorer/qwen3b-bt-v7-judge            # v7
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
) -> None:
    import numpy as np

    from tpot_taste.eval.demo_panel import EXPECTED_HIGH, EXPECTED_LOW, PANEL
    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    sm, stok = load_trained_scorer(str(scorer), base)
    print(f"[scorer] {scorer}")
    means = {}
    for cat, texts in PANEL.items():
        s = np.asarray(score_texts(sm, stok, texts, batch_size=16))
        means[cat] = s
        exp = "HIGH" if cat in EXPECTED_HIGH else "low"
        print(f"  {cat:16s} mean={s.mean():+6.2f}  [{s.min():+.2f},{s.max():+.2f}]  {exp}")
    worst_high = min(means[c].mean() for c in EXPECTED_HIGH)
    best_low = max(means[c].mean() for c in EXPECTED_LOW)
    hi = np.concatenate([means[c] for c in EXPECTED_HIGH]).mean()
    lo = np.concatenate([means[c] for c in EXPECTED_LOW]).mean()
    clean = worst_high > best_low
    print(f"\n  HIGH mean {hi:+.2f} vs LOW mean {lo:+.2f} (gap {hi-lo:+.2f})")
    print(f"  worst-HIGH {worst_high:+.2f} vs best-LOW {best_low:+.2f} -> "
          f"{'CLEAN ✓ (taste > generic-virality)' if clean else 'BLURRED ✗ (a low category outranks a high one)'}")
    # which low category is the worst offender
    low_sorted = sorted(((means[c].mean(), c) for c in EXPECTED_LOW), reverse=True)
    print(f"  top low category: {low_sorted[0][1]} ({low_sorted[0][0]:+.2f})")


if __name__ == "__main__":
    app()
