"""Scorer-only de-platitude acceptance gate (generalises demo_eval §1 to ANY base).

Scores the fixed, versioned archetype panel (tpot_taste/eval/demo_panel.py) with a candidate
scorer adapter and reports the HIGH (tpot_canon, aphorism) vs LOW (platitude/promo/bait/...)
separation — the qualitative bar a scorer must clear before being LOCKED. The hard case is
keeping `aphorism` above `generic_viral` (sharp sayings vs motivational platitudes).

Unlike demo_eval.py this takes an explicit `--base`, so it works for a scorer whose base differs
from the Writer's (e.g. a Qwen3-8B scorer alongside a Qwen2.5-3B writer). Scorer-only: it never
loads the Writer, so there is no two-model co-residency cost — fits the candidate easily in 12 GB.

  uv run python scripts/accept_scorer.py --scorer outputs/scorer/v8-qwen3-8b --base Qwen/Qwen3-8B
  uv run python scripts/accept_scorer.py --scorer outputs/scorer/qwen3b-bt-taste-LOCKED  # v7.2 baseline
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
    from tpot_taste.engine import resolve_adapter_base
    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    base = resolve_adapter_base(scorer, base)
    model, tok = load_trained_scorer(str(scorer), base)
    cat_scores = {cat: np.asarray(score_texts(model, tok, texts)) for cat, texts in PANEL.items()}

    print(f"[accept] scorer={scorer.name}  base={base}")
    print(f"{'category':16s} {'mean':>7s} {'min':>7s} {'max':>7s}  n  expected")
    for cat in PANEL:
        s = cat_scores[cat]
        exp = "HIGH" if cat in EXPECTED_HIGH else "low"
        note = "  <- platitude" if cat == "generic_viral" else ("  <- audit leak" if cat == "promo" else "")
        print(f"{cat:16s} {s.mean():+7.2f} {s.min():+7.2f} {s.max():+7.2f} {len(s):2d}  {exp}{note}")

    high = np.concatenate([cat_scores[c] for c in EXPECTED_HIGH])
    low = np.concatenate([cat_scores[c] for c in EXPECTED_LOW])
    worst_high = min(cat_scores[c].mean() for c in EXPECTED_HIGH)
    best_low = max(cat_scores[c].mean() for c in EXPECTED_LOW)
    clean = worst_high > best_low
    print()
    print(f"HIGH mean {high.mean():+.2f}   LOW mean {low.mean():+.2f}   gap {high.mean() - low.mean():+.2f}")
    print(f"worst-HIGH category {worst_high:+.2f}  vs  best-LOW category {best_low:+.2f}  -> "
          f"{'CLEAN separation' if clean else 'BLURRED (a low category outranks a high one)'}")
    print()
    print("Per-probe (sorted desc within category):")
    for cat, texts in PANEL.items():
        s = cat_scores[cat]
        print(f"[{cat}]")
        for i in np.argsort(-s):
            print(f"  {s[i]:+6.2f}  {texts[i][:96]}")
    # machine-readable one-liner for diffing candidates
    print(f"\n[verdict] {'CLEAN' if clean else 'BLURRED'} gap={high.mean() - low.mean():+.2f} "
          f"worstHIGH={worst_high:+.2f} bestLOW={best_low:+.2f}")


if __name__ == "__main__":
    app()
