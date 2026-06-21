"""Eval Writer adapters head-to-head, judged by the LOCKED v6 scorer (Phase 3/4).

Generate ideate+improve samples from model A and model B on the same prompts, score both with the
locked scorer, report win-rate + means per task, a LENGTH-DRIFT audit (reward-hacking check), and
side-by-side examples (the real signal). B may be another adapter or the literal "base".

  uv run python scripts/eval_writer.py                      # DPO vs SFT (defaults)
  uv run python scripts/eval_writer.py --a outputs/writer/qwen3b-sft-v1 --b base   # SFT vs base
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
    a: Path = Path("outputs/writer/qwen3b-dpo-v1"),
    b: str = "outputs/writer/qwen3b-sft-v1",
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    drafts_path: Path = Path("data/splits/deopt_v6_test_pairs.parquet"),
    n_ideate: int = 40,
    n_improve: int = 20,
    seed: int = 0,
    out: Path = Path("outputs/writer/eval_ab.json"),
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
    m = PeftModel.from_pretrained(m, str(a), adapter_name="A").eval()
    b_is_base = str(b).lower() == "base"
    if not b_is_base:
        m.load_adapter(str(b), adapter_name="B")

    def generate(which: str) -> list[str]:
        outs: list[str] = []
        for i in range(0, len(tasks), 16):
            chunk = tasks[i : i + 16]
            rendered = [
                tok.apply_chat_template(
                    [{"role": "system", "content": SYS_PROMPT}, {"role": "user", "content": u}],
                    tokenize=False, add_generation_prompt=True,
                )
                for _, u in chunk
            ]
            enc = tok(rendered, return_tensors="pt", padding=True, truncation=True, max_length=320).to(m.device)
            kw = dict(max_new_tokens=64, do_sample=True, temperature=0.8, top_p=0.95, pad_token_id=tok.pad_token_id)
            if which == "B" and b_is_base:
                with torch.no_grad(), m.disable_adapter():
                    g = m.generate(**enc, **kw)
            else:
                m.set_adapter(which)
                with torch.no_grad():
                    g = m.generate(**enc, **kw)
            for j in range(len(chunk)):
                outs.append(_clean(tok.decode(g[j][enc["input_ids"].shape[1] :], skip_special_tokens=True)))
        return outs

    label_a, label_b = a.name, ("base" if b_is_base else Path(b).name)
    print(f"[gen] A={label_a}"); out_a = generate("A")
    print(f"[gen] B={label_b}"); out_b = generate("B")
    del m; gc.collect(); torch.cuda.empty_cache()

    from tpot_taste.scoring.model import load_trained_scorer, score_texts

    sm, stok = load_trained_scorer(str(scorer), base)
    sa = np.array(score_texts(sm, stok, out_a, batch_size=32))
    sb = np.array(score_texts(sm, stok, out_b, batch_size=32))
    la = np.array([len(x) for x in out_a]); lb = np.array([len(x) for x in out_b])
    kinds = np.array([t for t, _ in tasks])

    res = {"A": label_a, "B": label_b, "n": len(tasks),
           "A_mean": float(sa.mean()), "B_mean": float(sb.mean()), "A_beats_B": float((sa > sb).mean()),
           "A_len_mean": float(la.mean()), "B_len_mean": float(lb.mean())}
    for k in ["ideate", "improve"]:
        mk = kinds == k
        res[f"{k}_A_beats_B"] = float((sa[mk] > sb[mk]).mean())
        res[f"{k}_A_mean"] = float(sa[mk].mean()); res[f"{k}_B_mean"] = float(sb[mk].mean())

    print(f"\n== A={label_a}  vs  B={label_b}  (judged by locked v6 scorer) ==")
    print(f"  overall : A {res['A_mean']:+.2f}  B {res['B_mean']:+.2f}  A-beats-B {res['A_beats_B']:.2f}  (n={res['n']})")
    for k in ["ideate", "improve"]:
        print(f"  {k:8s}: A {res[f'{k}_A_mean']:+.2f}  B {res[f'{k}_B_mean']:+.2f}  A-beats-B {res[f'{k}_A_beats_B']:.2f}")
    print(f"  LENGTH audit (chars): A {res['A_len_mean']:.0f}  B {res['B_len_mean']:.0f}  "
          f"(big A>B = possible length reward-hacking)")

    order = np.argsort(-(sa - sb))
    print("\n== A's biggest wins ==")
    for idx in order[:5]:
        print(f"  [{kinds[idx]}] A {sa[idx]:+.2f} | B {sb[idx]:+.2f}")
        print(f"    A: {out_a[idx][:120]}")
        print(f"    B: {out_b[idx][:120]}")
    print("\n== A's biggest losses ==")
    for idx in order[-3:]:
        print(f"  [{kinds[idx]}] A {sa[idx]:+.2f} | B {sb[idx]:+.2f}")
        print(f"    A: {out_a[idx][:120]}")
        print(f"    B: {out_b[idx][:120]}")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=2))
    print(f"\n[done] -> {out}")


if __name__ == "__main__":
    app()
