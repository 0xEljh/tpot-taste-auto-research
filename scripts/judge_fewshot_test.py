"""Can FEW-SHOT human exemplars steer the judge where the rubric can't? (Phase 6e, D37)

Rubric edits barely moved judge↔human alignment (D36: 0.126→0.142) — the base 3B's taste prior is rigid to
abstract rules. This tests the stronger lever: put the HUMAN's labeled examples in the prompt (in-context).
Split the 60: a few clear exemplars (tpot / not) go in the prompt; score the REST and measure Spearman vs the
human — comparing rubric-only vs few-shot on the SAME held-out items (so it's a fair within-test comparison).

  uv run python scripts/judge_fewshot_test.py --n-shot 6
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
    n_shot: int = 6,   # per class (tpot / not) -> 2*n_shot exemplars
    max_new_tokens: int = 80,
) -> None:
    import numpy as np
    import polars as pl
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.scoring.judge import RUBRIC, build_score_prompt, parse_score

    lab = {int(k): _MAP[v] for k, v in json.loads(labels.read_text()).items() if v in _MAP}
    rows = [(r["id"], r["text"]) for r in pl.read_parquet(calib).to_dicts() if r["id"] in lab]
    tpot = [(i, t) for i, t in rows if lab[i] == 1.0]
    nott = [(i, t) for i, t in rows if lab[i] == 0.0]
    shots = tpot[:n_shot] + nott[:n_shot]           # exemplars (clear cases)
    shot_ids = {i for i, _ in shots}
    test = [(i, t) for i, t in rows if i not in shot_ids]   # held-out (incl. borderline)
    human = np.array([lab[i] for i, _ in test])
    print(f"[split] {len(shots)} exemplars ({n_shot}+{n_shot}), {len(test)} held-out")

    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    jm = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16).eval()

    # few-shot preamble: the human's exemplars as scored turns (tpot=9, not=1).
    # shots is ordered [n_shot tpot, then n_shot not], so index < n_shot => tpot.
    shot_msgs = []
    for k, (_, t) in enumerate(shots):
        shot_msgs += build_score_prompt(t)[1:]  # the user turn only
        shot_msgs.append({"role": "assistant", "content": f"SCORE: {9 if k < n_shot else 1}"})

    def score(texts, few_shot: bool) -> np.ndarray:
        out = []
        for i in range(0, len(texts), 8):
            batch = texts[i:i + 8]
            prompts = []
            for t in batch:
                msgs = [{"role": "system", "content": RUBRIC}]
                if few_shot:
                    msgs += shot_msgs
                msgs += build_score_prompt(t)[1:]
                prompts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048).to(jm.device)
            with torch.no_grad():
                g = jm.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tok.pad_token_id)
            for j in range(len(batch)):
                out.append(parse_score(tok.decode(g[j][enc["input_ids"].shape[1]:], skip_special_tokens=True)))
        return np.array([o if o is not None else np.nan for o in out])

    txts = [t for _, t in test]
    base_s = score(txts, False)
    shot_s = score(txts, True)

    def spearman(a, b, m):
        a, b = a[m], b[m]
        ra, rb = a.argsort().argsort().astype(float), b.argsort().argsort().astype(float)
        return float(np.corrcoef(ra, rb)[0, 1])

    mb = ~np.isnan(base_s)
    ms = ~np.isnan(shot_s)
    print(f"\n=== judge↔human on {len(test)} held-out (Spearman) ===")
    print(f"  rubric-only : {spearman(base_s, human, mb):+.3f}  (miss {int((~mb).sum())})")
    print(f"  + few-shot  : {spearman(shot_s, human, ms):+.3f}  (miss {int((~ms).sum())})")
    print("\n  if few-shot >> rubric-only -> in-context human exemplars ARE the lever (re-distill v7.2 with a")
    print("  few-shot judge); if not -> 3B judge is the ceiling -> need more human labels or a bigger judge.")


if __name__ == "__main__":
    app()
