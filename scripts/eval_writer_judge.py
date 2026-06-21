"""Judge-arbitrated Writer A/B (Phase 6d, D34) — the FAIR final eval.

Comparing the new Writer (re-SFT + re-DPO vs v7.1) to the old (DPO-v2 vs v6) with the v7.1 SCORER would give
the new one home advantage (it was trained against v7.1). The independent, validated TASTE JUDGE (D31) is the
fair arbiter: for each prompt, take each Writer's best-of-N candidate, and let the judge pick which is more tpot
(order-robust, agree_winner). Win-rate by the judge = which Writer the taste signal actually prefers.

Memory: engines loaded sequentially (writer A + scorer, then B), then the judge — never >2 models at once.

  uv run python scripts/eval_writer_judge.py --a outputs/writer/qwen3b-dpo-v2sft --b outputs/writer/qwen3b-dpo-v2
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

# Fixed prompt set (incl. the platitude traps). (kind, payload)
PROMPTS = [
    ("ideate", "recursion"), ("ideate", "debugging at 2am"), ("ideate", "the feeling of deleting a lot of code"),
    ("ideate", "discipline"), ("ideate", "reading philosophy you don't fully understand"), ("ideate", "your desk"),
    ("ideate", "what your terminal history says about you"), ("ideate", "the moment a bug becomes interesting"),
    ("ideate", "productivity"), ("ideate", "leaving a job"), ("ideate", "old code you wrote"),
    ("ideate", "why you procrastinate"),
    ("improve", "lit a fake cig to feel something"), ("improve", "work hard and you will succeed"),
    ("improve", "i think therefore i am but for code"), ("improve", "AI is going to change everything"),
    ("improve", "always be learning"), ("improve", "the gym is my therapy"),
]


@app.command()
def main(
    a: Path = Path("outputs/writer/qwen3b-dpo-v2sft"),
    b: Path = Path("outputs/writer/qwen3b-dpo-v2"),
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    best_of: int = 8,
) -> None:
    import gc

    import torch

    from tpot_taste.engine import TasteEngine

    def tops(writer: Path) -> list[str]:
        eng = TasteEngine(writer=writer, scorer=scorer)
        out = []
        for kind, payload in PROMPTS:
            ranked = eng.ideate(payload, k=1, best_of=best_of) if kind == "ideate" \
                else eng.improve(payload, k=1, best_of=best_of)
            out.append(ranked[0][0] if ranked else "")
        del eng
        gc.collect()
        torch.cuda.empty_cache()
        return out

    print(f"[gen] A={a.name} ...")
    a_tops = tops(a)
    print(f"[gen] B={b.name} ...")
    b_tops = tops(b)

    # judge arbitration
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.scoring.judge import judge_pairwise_2x

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    jm = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16).eval()

    a_wins = b_wins = ties = 0
    print(f"\n{'prompt':42s} winner")
    for (kind, payload), ta, tb in zip(PROMPTS, a_tops, b_tops):
        w = judge_pairwise_2x(jm, tok, ta, tb, max_new_tokens=80, temperature=0.0)  # 'a'=A, 'b'=B, None=tie
        if w == "a":
            a_wins += 1
        elif w == "b":
            b_wins += 1
        else:
            ties += 1
        tag = {"a": "A", "b": "B", None: "tie"}[w]
        print(f"  {kind:7s} {payload[:32]:32s} {tag}")
        print(f"      A: {ta[:96]}")
        print(f"      B: {tb[:96]}")
    decided = a_wins + b_wins
    wr = a_wins / decided if decided else float("nan")
    print(f"\n[judge A/B] A({a.name}) wins {a_wins}  B({b.name}) wins {b_wins}  ties {ties}  "
          f"-> A win-rate {wr:.2f} (of decided)")


if __name__ == "__main__":
    app()
