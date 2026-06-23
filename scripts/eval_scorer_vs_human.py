"""Score a BT scorer against the human's held-out calibration labels (Phase 6e, D39).

The linchpin metric: does a scorer distilled from the few-shot (aligned) judge agree with the HUMAN better
than v7.1 (distilled from the rubric judge)? Scores the held-out items (NOT used as few-shot exemplars) with
the given scorer and reports Spearman / precision vs the human.

  uv run python scripts/eval_scorer_vs_human.py --scorer outputs/scorer/qwen3b-bt-v71-judge-deopt
  uv run python scripts/eval_scorer_vs_human.py --scorer outputs/scorer/qwen3b-bt-v72fs --heldout outputs/fs_heldout_ids.json
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_MAP = {"👍 tpot": 1.0, "🤔 borderline": 0.5, "👎 not tpot": 0.0}


@app.command()
def main(
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    base: str = typer.Option(None, help="base model id; default: read from the scorer's adapter_config"),
    calib: Path = Path("data/splits/calibration_set.parquet"),
    labels: Path = Path("outputs/calibration_labels.json"),
    heldout: Path = typer.Option(None, help="json list of ids to restrict to (e.g. few-shot held-out)"),
) -> None:
    import numpy as np
    import polars as pl

    from tpot_taste.engine import resolve_scorer_base
    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    lab = {int(k): _MAP[v] for k, v in json.loads(labels.read_text()).items() if v in _MAP}
    keep = set(json.loads(heldout.read_text())) if heldout and heldout.exists() else set(lab)
    rows = [r for r in pl.read_parquet(calib).to_dicts() if r["id"] in lab and r["id"] in keep]
    human = np.array([lab[r["id"]] for r in rows])

    base = base or resolve_scorer_base(scorer, "Qwen/Qwen2.5-3B-Instruct")
    sm, stok = load_trained_scorer(str(scorer), base)
    s = np.asarray(score_texts(sm, stok, [r["text"] for r in rows], batch_size=16))

    def spearman(a, b):
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    dec = human != 0.5
    thr = np.percentile(s, 100 * (1 - (human == 1.0).mean()))  # top-k by human tpot rate
    prec = ((human[dec] == 1.0) & (s[dec] >= thr)).sum() / max(1, (s[dec] >= thr).sum())
    # pairwise accuracy on tpot-vs-not held-out pairs — the highest-power metric at small N (random=0.5)
    hi, lo = s[human == 1.0], s[human == 0.0]
    pw = np.mean([(1.0 if h > l else 0.5 if h == l else 0.0) for h in hi for l in lo]) if len(hi) and len(lo) else float("nan")
    print(f"[{scorer.name}] on {len(rows)} human-labeled items "
          f"({int((human==1).sum())} tpot / {int((human==0).sum())} not / {int((human==0.5).sum())} borderline)")
    print(f"  Spearman(scorer, human) = {spearman(s, human):+.3f}   "
          f"pairwise-acc(tpot>not) = {pw:.2f} over {len(hi)*len(lo)} pairs   precision@taste-rate = {prec:.2f}")


if __name__ == "__main__":
    app()
