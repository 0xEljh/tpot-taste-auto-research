"""Score the judge against human calibration labels (Phase 6e, D36).

After the human labels the Notion calibration DB, I pull (id -> Taste) via the Notion MCP and write it to a
labels JSON; this joins it to the hidden judge_score and MEASURES the judge:
  - rank agreement (Spearman) between judge_score and the human taste label,
  - binary agreement / precision / recall at a tpot threshold,
  - the judge's ERRORS — judge-high but human-not-tpot (rubric over-includes) and judge-low but human-tpot
    (rubric misses) — which are the concrete cases to fix the rubric on.

  uv run python scripts/score_calibration.py --labels outputs/calibration_labels.json
labels JSON: {"1": "👍 tpot", "2": "👎 not tpot", ...}  (id -> Taste select value)
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_MAP = {"👍 tpot": 1.0, "🤔 borderline": 0.5, "👎 not tpot": 0.0,
        "tpot": 1.0, "borderline": 0.5, "not tpot": 0.0}


@app.command()
def main(
    labels: Path = Path("outputs/calibration_labels.json"),
    calib: Path = Path("data/splits/calibration_set.parquet"),
    tpot_thresh: float = 6.0,   # judge score >= this => judge predicts "tpot"
    not_thresh: float = 4.0,    # judge score <= this => judge predicts "not tpot"
) -> None:
    import numpy as np
    import polars as pl

    lab = {int(k): _MAP[v] for k, v in json.loads(labels.read_text()).items() if v in _MAP}
    df = pl.read_parquet(calib)
    rows = [(r["id"], r["text"], r["judge_score"], lab[r["id"]]) for r in df.to_dicts() if r["id"] in lab]
    print(f"[calib] {len(rows)}/{df.height} labeled")
    if not rows:
        return
    js = np.array([r[2] for r in rows])
    hl = np.array([r[3] for r in rows])

    def spearman(a, b):
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    print(f"\n[rank agreement] Spearman(judge, human) = {spearman(js, hl):+.3f}")
    print(f"  human label mix: tpot={int((hl==1).sum())} borderline={int((hl==0.5).sum())} not={int((hl==0).sum())}")

    # binary agreement on the decisive (non-borderline) human labels
    dec = hl != 0.5
    h_tpot = hl[dec] == 1.0
    j_pred = js[dec] >= tpot_thresh
    if dec.sum():
        agree = (h_tpot == j_pred).mean()
        tp = (h_tpot & j_pred).sum(); fp = (~h_tpot & j_pred).sum(); fn = (h_tpot & ~j_pred).sum()
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        rec = tp / (tp + fn) if (tp + fn) else float("nan")
        print(f"\n[binary @ judge>={tpot_thresh:.0f}] agreement={agree:.2f} on {int(dec.sum())} decisive labels  "
              f"precision={prec:.2f} recall={rec:.2f}")

    print(f"\n=== judge FALSE POSITIVES (judge>={tpot_thresh:.0f} but human = not tpot) — rubric OVER-includes ===")
    for i, t, j, h in sorted(rows, key=lambda r: -r[2]):
        if j >= tpot_thresh and h == 0.0:
            print(f"  judge {j:4.1f} | {t[:96]}")
    print(f"\n=== judge FALSE NEGATIVES (judge<={not_thresh:.0f} but human = tpot) — rubric MISSES ===")
    for i, t, j, h in sorted(rows, key=lambda r: r[2]):
        if j <= not_thresh and h == 1.0:
            print(f"  judge {j:4.1f} | {t[:96]}")
    print("\n[done] — these error cases are the rubric-tuning targets (D36).")


if __name__ == "__main__":
    app()
