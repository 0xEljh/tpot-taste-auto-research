"""Eval the Taste Writer (Phase 3): SFT Writer vs base, judged by the LOCKED v6 scorer.

Generate ideate + improve samples from the SFT adapter and from the base model (same weights,
adapter disabled), score both with the locked scorer, report win-rate + means, and print
examples for the qualitative check (the real signal — don't just trust the win-rate).

  uv run python scripts/eval_writer.py
"""
from __future__ import annotations

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
    n_ideate: int = 40,
    n_improve: int = 20,
    seed: int = 0,
    out: Path = Path("outputs/writer/eval_v1.json"),
) -> None:
    import gc
    import json

    import numpy as np
    import polars as pl
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.writer.sft_data import IDEATE_PROMPTS, IMPROVE_PROMPTS, SYS_PROMPT

    rng = random.Random(seed)
    tok = AutoTokenizer.from_pretrained(base)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    drafts = pl.read_parquet(drafts_path)["text_clean_l"].to_list()
    rng.shuffle(drafts)
    tasks = [("ideate", rng.choice(IDEATE_PROMPTS)) for _ in range(n_ideate)]
    tasks += [("improve", IMPROVE_PROMPTS[0].format(draft=d)) for d in drafts[:n_improve]]

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    m = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16)
    m = PeftModel.from_pretrained(m, str(writer)).eval()

    def generate(use_adapter: bool) -> list[str]:
        outs: list[str] = []
        for i in range(0, len(tasks), 16):
            chunk = tasks[i : i + 16]
            prompts = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True,
                )
                for _, u in chunk
            ]
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=320).to(m.device)
            gen_kwargs = dict(max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.95,
                              pad_token_id=tok.pad_token_id)
            if use_adapter:
                with torch.no_grad():
                    g = m.generate(**enc, **gen_kwargs)
            else:
                with torch.no_grad(), m.disable_adapter():
                    g = m.generate(**enc, **gen_kwargs)
            for j in range(len(chunk)):
                outs.append(_clean(tok.decode(g[j][enc["input_ids"].shape[1] :], skip_special_tokens=True)))
        return outs

    print("[gen] writer..."); writer_out = generate(True)
    print("[gen] base...");   base_out = generate(False)
    del m; gc.collect(); torch.cuda.empty_cache()

    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    sm, stok = load_trained_scorer(str(scorer), base)
    sw = np.array(score_texts(sm, stok, writer_out, batch_size=32))
    sb = np.array(score_texts(sm, stok, base_out, batch_size=32))

    kinds = np.array([t for t, _ in tasks])
    res = {"n": len(tasks), "writer_mean": float(sw.mean()), "base_mean": float(sb.mean()),
           "win_rate": float((sw > sb).mean())}
    for k in ["ideate", "improve"]:
        mask = kinds == k
        res[f"{k}_writer_mean"] = float(sw[mask].mean())
        res[f"{k}_base_mean"] = float(sb[mask].mean())
        res[f"{k}_win_rate"] = float((sw[mask] > sb[mask]).mean())

    print(f"\n== Writer SFT v1 vs base (judged by locked v6 scorer) ==")
    print(f"  overall : writer {res['writer_mean']:+.2f}  base {res['base_mean']:+.2f}  win-rate {res['win_rate']:.2f}  (n={res['n']})")
    for k in ["ideate", "improve"]:
        print(f"  {k:8s}: writer {res[f'{k}_writer_mean']:+.2f}  base {res[f'{k}_base_mean']:+.2f}  win-rate {res[f'{k}_win_rate']:.2f}")

    order = np.argsort(-(sw - sb))
    print("\n== examples (writer's biggest wins; score in []) ==")
    for idx in order[:6]:
        print(f"  [{kinds[idx]}] W {sw[idx]:+.2f} | B {sb[idx]:+.2f}")
        print(f"    WRITER: {writer_out[idx][:120]}")
        print(f"    BASE  : {base_out[idx][:120]}")
    print("\n== examples (writer's worst losses) ==")
    for idx in order[-3:]:
        print(f"  [{kinds[idx]}] W {sw[idx]:+.2f} | B {sb[idx]:+.2f}")
        print(f"    WRITER: {writer_out[idx][:120]}")
        print(f"    BASE  : {base_out[idx][:120]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    app()
