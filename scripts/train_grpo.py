"""GRPO the Taste Writer from DPO v2 (Phase 4b, D5 / design §3).

On-policy: the policy generates N completions per prompt, the multi-objective reward (v6 taste
score − length penalty − bait penalty) scores them, GRPO updates on group-relative advantage with a
KL leash (β) to the reference (adapter-disabled base). No vLLM (12 GB-tight) — transformers gen.

  uv run python scripts/train_grpo.py --smoke   # 5 steps, no W&B
  uv run python scripts/train_grpo.py
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    start_adapter: Path = Path("outputs/writer/qwen3b-writer-LOCKED"),  # DPO v2
    scorer: Path = Path("outputs/scorer/qwen3b-bt-taste-LOCKED"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    drafts_path: Path = Path("data/splits/deopt_v6_test_pairs.parquet"),
    out: Path = Path("outputs/writer/qwen3b-grpo-v1"),
    max_steps: int = 200,
    num_generations: int = 4,
    batch_size: int = 4,
    grad_accum: int = 4,
    lr: float = 1e-6,
    beta: float = 0.04,
    temperature: float = 0.9,
    max_completion: int = 64,
    max_prompt: int = 256,
    n_prompts: int = 600,
    length_target: int = 200,
    length_penalty: float = 0.01,
    bait_penalty: float = 2.0,
    rep_penalty: float = 0.0,
    run_name: str = "writer-grpo-v1",
    smoke: bool = False,
) -> None:
    import random

    import polars as pl
    import torch
    from datasets import Dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.writer._trl_compat import patch_trl_availability

    patch_trl_availability()
    from trl import GRPOConfig, GRPOTrainer

    from tpot_taste.engine import resolve_adapter_base
    from tpot_taste.scoring.model import load_trained_scorer, score_texts
    from tpot_taste.writer.grpo_reward import reward_components
    from tpot_taste.writer.sft_data import IDEATE_PROMPTS, IMPROVE_PROMPTS, SYS_PROMPT

    rng = random.Random(0)
    drafts = pl.read_parquet(drafts_path)["text_clean_l"].to_list()
    rng.shuffle(drafts)
    rows = [[{"role": "system", "content": SYS_PROMPT},
             {"role": "user", "content": rng.choice(IDEATE_PROMPTS)}] for _ in range(2 * n_prompts // 3)]
    rows += [[{"role": "system", "content": SYS_PROMPT},
              {"role": "user", "content": IMPROVE_PROMPTS[0].format(draft=d)}] for d in drafts[: n_prompts // 3]]
    rng.shuffle(rows)
    ds = Dataset.from_dict({"prompt": rows})
    if smoke:
        ds = ds.select(range(32))
    print(f"[data] {ds.num_rows:,} prompts")

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    scorer_base = resolve_adapter_base(scorer, base)
    sm, stok = load_trained_scorer(str(scorer), scorer_base)  # reward model, loaded once

    def taste_reward(completions, **kwargs):
        texts = [c[-1]["content"] if isinstance(c, list) else str(c) for c in completions]
        base_scores = score_texts(sm, stok, texts, batch_size=16)
        return [reward_components(t, s, length_target=length_target, length_penalty=length_penalty,
                                  bait_penalty=bait_penalty, rep_penalty=rep_penalty)
                for t, s in zip(texts, base_scores)]

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}
    model = PeftModel.from_pretrained(model, str(start_adapter), is_trainable=True)  # start from DPO v2
    model.config.use_cache = False

    cfg = GRPOConfig(
        output_dir=str(out),
        max_steps=(5 if smoke else max_steps),
        num_generations=num_generations,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        beta=beta,
        temperature=temperature,
        max_completion_length=max_completion,
        max_prompt_length=max_prompt,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=(1 if smoke else 10),
        save_strategy=("no" if smoke else "steps"),
        save_steps=100,
        report_to=("none" if smoke else "wandb"),
        run_name=run_name,
        use_vllm=False,
    )
    trainer = GRPOTrainer(model=model, reward_funcs=taste_reward, args=cfg, train_dataset=ds, processing_class=tok)
    # kbit-training prep upcasts the FROZEN norms + lm_head to fp32; GRPO *generates* (bf16 hidden states)
    # -> dtype mismatch. Cast frozen fp32 params back to bf16 (trainable LoRA params stay fp32 -> training
    # precision unaffected). Done after trainer setup so it catches the upcast.
    n_cast = sum(1 for p in trainer.model.parameters() if p.dtype == torch.float32 and not p.requires_grad)
    for p in trainer.model.parameters():
        if p.dtype == torch.float32 and not p.requires_grad:
            p.data = p.data.to(torch.bfloat16)
    print(f"[dtype] cast {n_cast} frozen fp32 params -> bf16")
    trainer.train()
    if not smoke:
        trainer.save_model(str(out))
        tok.save_pretrained(str(out))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    app()
