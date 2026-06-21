"""LINCHPIN test (Phase 6e, D39): does the few-shot judge's alignment SURVIVE distillation into the scorer?

The whole plan is: human labels -> few-shot judge (aligned) -> distill -> fast aligned scorer. The unproven
link is the distill step. This builds deopt pairs from a FEW-SHOT-judged pool (anchors = few-shot-high), so a
scorer trained on them should inherit the few-shot judge's taste. Held-out human labels (not used as exemplars)
then test whether the trained scorer aligns with the human better than v7.1 (which used the rubric judge).

  uv run python scripts/fewshot_distill_test.py        # writes pairs + held-out eval ids; then train + eval
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)
_MAP = {"👍 tpot": 1.0, "🤔 borderline": 0.5, "👎 not tpot": 0.0}


@app.command()
def main(
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    calib: Path = Path("data/splits/calibration_set.parquet"),
    labels: Path = Path("outputs/calibration_labels.json"),
    pool: Path = Path("data/splits/taste_v7_pool_scored.parquet"),
    out_pairs: Path = Path("data/splits/taste_fs_deopt_train_pairs.parquet"),
    out_heldout: Path = Path("outputs/fs_heldout_ids.json"),
    n_shot: int = 8,        # per class -> 16 exemplars; the rest are held-out for eval
    n_pool: int = 1500,
    n_styles: int = 2,
    seed: int = 0,
) -> None:
    import random

    import polars as pl
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.data.degrade import degrade_texts
    from tpot_taste.scoring.judge import judge_score_batch

    lab = {int(k): _MAP[v] for k, v in json.loads(labels.read_text()).items() if v in _MAP}
    crows = [r for r in pl.read_parquet(calib).to_dicts() if r["id"] in lab]
    tpot = [(r["id"], r["text"]) for r in crows if lab[r["id"]] == 1.0]
    nott = [(r["id"], r["text"]) for r in crows if lab[r["id"]] == 0.0]
    exemplars = [(t, 9.0) for _, t in tpot[:n_shot]] + [(t, 1.0) for _, t in nott[:n_shot]]
    held = [r["id"] for r in crows if r["id"] not in {i for i, _ in (tpot[:n_shot] + nott[:n_shot])}]
    out_heldout.parent.mkdir(parents=True, exist_ok=True)
    out_heldout.write_text(json.dumps(held))
    print(f"[exemplars] {len(exemplars)} ({n_shot}+{n_shot})  held-out eval ids: {len(held)}")

    rng = random.Random(seed)
    pdf = pl.read_parquet(pool).to_dicts()
    rng.shuffle(pdf)
    anchors_pool = pdf[:n_pool]
    texts = [r["text"] for r in anchors_pool]
    print(f"[pool] few-shot judging {len(texts)} items with {len(exemplars)} exemplars ...")

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    m = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16).eval()

    fs = judge_score_batch(m, tok, texts, exemplars=exemplars, batch_size=8, max_length=2048)
    anchors = [r["text"] for r, s in zip(anchors_pool, fs) if s is not None and s >= 7.0]
    print(f"[anchors] {len(anchors)} few-shot-high (>=7) of {len(texts)}  (miss {sum(s is None for s in fs)})")

    deg_texts = anchors * n_styles
    degs = degrade_texts(m, tok, deg_texts, batch_size=64, seed=seed)
    rows = [{"text_clean_w": g, "text_clean_l": b}
            for g, (b, _st) in zip(deg_texts, degs) if len(b) >= 15 and b.lower() != g.lower()]
    pl.DataFrame(rows).write_parquet(out_pairs)
    print(f"[write] {out_pairs}: {len(rows)} pairs (few-shot-anchored deopt)")
    print(f"\nNext: train_scorer.py --train-pairs {out_pairs} --out outputs/scorer/qwen3b-bt-v72fs --epochs 1")


if __name__ == "__main__":
    app()
