"""Re-test the (retuned) judge rubric against the human labels (Phase 6e, D36).

Re-judges the 60 calibration items with the CURRENT judge.RUBRIC and compares to the human labels:
new Spearman / precision vs the old (0.126 / 0.43). NOTE: in-sample — the rubric was tuned by looking at
these same 60 errors, so improvement here is a SANITY check (did the change move the judge as intended?),
not an unbiased win; an unbiased number needs a fresh human-labeled set. Also re-judges the dipstick panel
negatives to confirm the wit-retune didn't re-admit platitudes/promo/bait.

  uv run python scripts/rejudge_calibration.py
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_MAP = {"👍 tpot": 1.0, "🤔 borderline": 0.5, "👎 not tpot": 0.0}


@app.command()
def main(
    calib: Path = Path("data/splits/calibration_set.parquet"),
    labels: Path = Path("outputs/calibration_labels.json"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
) -> None:
    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.eval.demo_panel import PANEL
    from tpot_taste.scoring.judge import judge_score_batch

    lab = {int(k): _MAP[v] for k, v in json.loads(labels.read_text()).items() if v in _MAP}
    df = pl.read_parquet(calib)
    rows = [r for r in df.to_dicts() if r["id"] in lab]
    texts = [r["text"] for r in rows]
    human = np.array([lab[r["id"]] for r in rows])
    old = np.array([r["judge_score"] for r in rows])

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    jm = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16).eval()

    new = np.array([s if s is not None else np.nan for s in judge_score_batch(jm, tok, texts, batch_size=16)])
    ok = ~np.isnan(new)

    def spearman(a, b):
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    def prec_at(scores, thr=6.0):
        dec = human != 0.5
        h, j = human[dec] == 1.0, scores[dec] >= thr
        tp, fp = (h & j).sum(), (~h & j).sum()
        return tp / (tp + fp) if (tp + fp) else float("nan")

    print("=== judge alignment with human (n={}) ===".format(int(ok.sum())))
    print(f"  OLD rubric: Spearman {spearman(old, human):+.3f}  precision@6 {prec_at(old):.2f}")
    print(f"  NEW rubric: Spearman {spearman(new[ok], human[ok]):+.3f}  precision@6 {prec_at(new):.2f}  "
          f"(in-sample sanity — tuned on these)")

    print("\n=== still-wrong with NEW rubric ===")
    print("  judge HIGH (>=6) but human NOT tpot:")
    for t, j, h in sorted(zip(texts, new, human), key=lambda x: -x[1]):
        if j >= 6 and h == 0.0:
            print(f"    {j:4.1f} | {t[:88]}")
    print("  judge LOW (<=4) but human tpot:")
    for t, j, h in sorted(zip(texts, new, human), key=lambda x: x[1]):
        if j <= 4 and h == 1.0:
            print(f"    {j:4.1f} | {t[:88]}")

    print("\n=== panel sanity (NEW rubric must still reject platitude/promo/bait) ===")
    for cat in ["tpot_canon", "aphorism", "generic_viral", "promo", "corporate", "bait", "assistant_slop"]:
        s = [x for x in judge_score_batch(jm, tok, PANEL[cat], batch_size=16) if x is not None]
        print(f"  {cat:16s} mean judge {np.mean(s):4.1f}")


if __name__ == "__main__":
    app()
