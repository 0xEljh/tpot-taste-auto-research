"""Grade the taste judge on the dipstick panel (Phase 6c, D29) — the judge's exam.

The dipstick (D28) showed v6 FAILS: generic_viral is its top category (+2.66 > tpot +1.68). A judge is only
worth trusting if it fixes that. Two measures on the SAME panel:
  - pointwise: judge_score each probe -> category means (directly comparable to the v6 dipstick table)
  - pairwise crux: every (tpot|aphorism) vs generic_viral pair, order-robust (agree_winner) -> % the judge
    correctly prefers the tpot/aphorism over the platitude. This is the platitude-blind-spot test.

Default judge = base Qwen (no adapter). `--judge-adapter outputs/writer/qwen3b-writer-LOCKED` tries the
tpot-voiced Writer as judge ("use the same model in scoring", D29-C).

  uv run python scripts/judge_eval.py
  uv run python scripts/judge_eval.py --judge-adapter outputs/writer/qwen3b-writer-LOCKED
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


def _load_judge(base: str, adapter: Path | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16)
    if adapter is not None:
        from peft import PeftModel
        m = PeftModel.from_pretrained(m, str(adapter))
    return m.eval(), tok


@app.command()
def main(
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    judge_adapter: Path = typer.Option(None),
    max_new_tokens: int = 80,
) -> None:
    import numpy as np

    from tpot_taste.eval.demo_panel import EXPECTED_HIGH, EXPECTED_LOW, PANEL
    from tpot_taste.scoring.judge import judge_pairwise_2x, judge_score

    m, tok = _load_judge(base, judge_adapter)
    kw = dict(max_new_tokens=max_new_tokens, temperature=0.0)
    tag = str(judge_adapter) if judge_adapter else "base Qwen"
    print(f"[judge] {tag}")

    # --- pointwise ---
    print("\n=== JUDGE pointwise scores (0-10) by category ===")
    cat_means: dict[str, float] = {}
    miss = 0
    for cat, texts in PANEL.items():
        vals = []
        for t in texts:
            s = judge_score(m, tok, t, **kw)
            if s is None:
                miss += 1
            else:
                vals.append(s)
        cat_means[cat] = float(np.mean(vals)) if vals else float("nan")
        exp = "HIGH" if cat in EXPECTED_HIGH else "low"
        print(f"  {cat:16s} mean={cat_means[cat]:5.2f}  n={len(vals)}  {exp}")
    worst_high = min(cat_means[c] for c in EXPECTED_HIGH)
    best_low = max(cat_means[c] for c in EXPECTED_LOW)
    print(f"  -> worst-HIGH {worst_high:.2f} vs best-LOW {best_low:.2f} : "
          f"{'CLEAN' if worst_high > best_low else 'BLURRED'}  (parse-misses={miss})")

    # --- pairwise crux: tpot/aphorism vs generic_viral platitudes ---
    print("\n=== JUDGE pairwise crux: (tpot|aphorism) vs generic_viral platitude ===")
    highs = [t for c in EXPECTED_HIGH for t in PANEL[c]]
    plats = PANEL["generic_viral"]
    correct = wrong = undecided = 0
    for h in highs:
        for p in plats:
            w = judge_pairwise_2x(m, tok, h, p, **kw)  # 'a'=h preferred, 'b'=p preferred, None=disagree
            if w == "a":
                correct += 1
            elif w == "b":
                wrong += 1
            else:
                undecided += 1
    decided = correct + wrong
    acc = correct / decided if decided else float("nan")
    print(f"  prefers tpot/aphorism over platitude: {correct}/{decided} = {acc:.2f}  "
          f"(undecided/position-biased: {undecided}/{len(highs)*len(plats)})")
    print(f"\n[verdict] judge {'PASSES' if (worst_high > best_low and acc > 0.6) else 'is WEAK'} the panel "
          f"(v6 baseline: BLURRED, generic_viral was the top category).")


if __name__ == "__main__":
    app()
