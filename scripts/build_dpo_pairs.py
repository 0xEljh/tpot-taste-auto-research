"""Build best-of-N DPO pairs (Phase 4, D5): sample N candidates per prompt from the SFT Writer,
score with the locked v6 scorer, keep best-vs-worst pairs that clear a margin (drops scorer-noise).

  uv run python scripts/build_dpo_pairs.py
Writes data/splits/dpo_train.jsonl (conversational prompt/chosen/rejected for TRL DPOTrainer).
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

_PRE = re.compile(r'^(sure|here(\'s| is)|okay|ok|certainly)[,:]?\s*', re.I)


def _clean(s: str) -> str:
    s = s.strip().strip('"').strip()
    s = _PRE.sub("", s).strip().strip('"').strip()
    return " ".join(s.split())


@app.command()
def main(
    writer: Path = Path("outputs/writer/qwen3b-sft-v1"),
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    drafts_path: Path = Path("data/splits/deopt_v6_test_pairs.parquet"),
    out: Path = Path("data/splits/dpo_train.jsonl"),
    n_ideate: int = 1200,
    n_improve: int = 600,
    n_cand: int = 4,
    min_margin: float = 1.5,
    length_penalty: float = 0.0,
    bait_penalty: float = 2.0,
    max_new_tokens: int = 64,
    seed: int = 0,
) -> None:
    import gc

    import numpy as np
    import polars as pl
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.writer.dpo_data import form_dpo_pairs
    from tpot_taste.writer.grpo_reward import is_baity
    from tpot_taste.writer.sft_data import IDEATE_PROMPTS, IMPROVE_PROMPTS, SYS_PROMPT

    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    drafts = pl.read_parquet(drafts_path)["text_clean_l"].to_list()
    rng.shuffle(drafts)
    prompts = [("ideate", rng.choice(IDEATE_PROMPTS)) for _ in range(n_ideate)]
    prompts += [("improve", IMPROVE_PROMPTS[0].format(draft=d)) for d in drafts[:n_improve]]
    # each prompt sampled n_cand times
    flat = [(gi, task, user) for gi, (task, user) in enumerate(prompts) for _ in range(n_cand)]
    print(f"[gen] {len(prompts):,} prompts x {n_cand} = {len(flat):,} candidates")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, str(writer)).eval()

    texts: list[str] = []
    for i in range(0, len(flat), 48):
        chunk = flat[i : i + 48]
        rendered = [
            tok.apply_chat_template(
                [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": u}],
                tokenize=False, add_generation_prompt=True,
            )
            for _, _, u in chunk
        ]
        enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=320).to(m.device)
        with torch.no_grad():
            g = m.generate(**enc, max_new_tokens=max_new_tokens, do_sample=True, temperature=0.9,
                           top_p=0.95, pad_token_id=tok.pad_token_id)
        for j in range(len(chunk)):
            texts.append(_clean(tok.decode(g[j][enc["input_ids"].shape[1] :], skip_special_tokens=True)))
        if i % (48 * 20) == 0:
            print(f"  ...{i + len(chunk)}/{len(flat)}")
    del m; gc.collect(); torch.cuda.empty_cache()

    from tpot_taste.engine import resolve_adapter_base
    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    # the SCORER carries its own base (v8 = Qwen3-8B), which differs from the writer `base` (Qwen3-4B).
    scorer_base = resolve_adapter_base(scorer, base)
    sm, stok = load_trained_scorer(str(scorer), scorer_base)
    scores = score_texts(sm, stok, texts, batch_size=48)

    # save raw candidates so pairs can be re-formed with a different length_penalty without regenerating
    raw_out = out.with_name("dpo_candidates.parquet")
    pl.DataFrame({"gi": [gi for gi, _, _ in flat], "task": [t for _, t, _ in flat],
                  "user": [u for _, _, u in flat], "text": texts, "score": scores,
                  "len": [len(t) for t in texts]}).write_parquet(raw_out)

    # length_penalty neutralizes v6's mild length lean (D21); bait_penalty sinks engagement-bait to the
    # REJECTED side — the v8 scorer over-rates 👇-style bait (it gave a generated bait post +0.71), so
    # without this, bait would become the `chosen`. Penalizing it instead teaches the writer bait=bad.
    n_baity = sum(is_baity(t) for t in texts)
    groups: dict[int, dict] = {}
    for (gi, task, user), text, sc in zip(flat, texts, scores):
        adj = sc - length_penalty * len(text) - (bait_penalty if is_baity(text) else 0.0)
        groups.setdefault(gi, {"user": user, "task": task, "candidates": []})["candidates"].append((text, adj))
    print(f"[bait] {n_baity}/{len(texts)} candidates flagged baity (penalty {bait_penalty} -> sunk to rejected)")

    records = form_dpo_pairs(groups.values(), sys_prompt=SYS_PROMPT, min_margin=min_margin)
    ch_len = np.mean([len(r["chosen"][0]["content"]) for r in records]) if records else 0.0
    rj_len = np.mean([len(r["rejected"][0]["content"]) for r in records]) if records else 0.0
    margins = np.array([r["margin"] for r in records]) if records else np.array([0.0])
    by: dict[str, int] = {}
    for r in records:
        by[r["task"]] = by.get(r["task"], 0) + 1

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in records:
            f.write(json.dumps({k: r[k] for k in ("prompt", "chosen", "rejected")}, ensure_ascii=False) + "\n")
    print(f"[write] {out} : {len(records):,} pairs (margin>= {min_margin}) {by}  "
          f"margin mean={margins.mean():.2f} p50={np.median(margins):.2f}")
    print(f"  length balance: chosen {ch_len:.0f} chars  rejected {rj_len:.0f} chars  "
          f"(want ~equal; chosen>>rejected = length leak) [length_penalty={length_penalty}]")
    for r in records[:3]:
        print(f"  --- {r['task']} (margin {r['margin']:.2f})")
        print(f"    CHOSEN  : {r['chosen'][0]['content'][:90]}")
        print(f"    REJECTED: {r['rejected'][0]['content'][:90]}")


if __name__ == "__main__":
    app()
