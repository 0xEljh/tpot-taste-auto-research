"""DPO the Taste Writer from the SFT checkpoint (Phase 4, D5/D20).

Policy = base(4-bit) + SFT LoRA adapter (trainable, continues from SFT); reference = adapter
disabled (= base), the memory-efficient TRL QLoRA-DPO path (ref_model=None). Preferences come from
best-of-N pairs judged by the locked v6 scorer (build_dpo_pairs.py). The reward-as-judge can be
gamed, so eval audits length/bait drift; raise --beta (stronger KL leash) if it hacks.

  uv run python scripts/train_dpo.py --smoke   # 5 steps, no W&B
  uv run python scripts/train_dpo.py
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    data: Path = Path("data/splits/dpo_train.jsonl"),
    sft_adapter: Path = Path("outputs/writer/qwen3b-sft-v1"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    out: Path = Path("outputs/writer/qwen3b-dpo-v1"),
    beta: float = 0.1,
    epochs: float = 1.0,
    batch_size: int = 4,
    grad_accum: int = 8,
    lr: float = 5e-6,
    scheduler: str = "linear",  # linear decay to 0 (recent-literature default); cosine = explicit ablation
    max_length: int = 512,
    max_prompt_length: int = 256,
    run_name: str = "writer-dpo-v1",
    smoke: bool = False,
) -> None:
    import torch
    from datasets import load_dataset
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    from tpot_taste.writer._trl_compat import patch_trl_availability

    patch_trl_availability()
    from trl import DPOConfig, DPOTrainer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=str(data), split="train")
    if smoke:
        ds = ds.select(range(min(64, ds.num_rows)))
    print(f"[data] {ds.num_rows:,} preference pairs")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16)
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}  # TRL 0.24 expects it; transformers 5.5 dropped it (cf. D20)
    model = PeftModel.from_pretrained(model, str(sft_adapter), is_trainable=True)  # continue from SFT
    model.config.use_cache = False

    cfg = DPOConfig(
        output_dir=str(out),
        beta=beta,
        num_train_epochs=epochs,
        max_steps=(5 if smoke else -1),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type=scheduler,
        warmup_ratio=0.05,
        max_length=max_length,
        max_prompt_length=max_prompt_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=(1 if smoke else 10),
        save_strategy=("no" if smoke else "epoch"),
        report_to=("none" if smoke else "wandb"),
        run_name=run_name,
    )

    trainer = DPOTrainer(model=model, ref_model=None, args=cfg, train_dataset=ds, processing_class=tok)
    trainer.train()
    if not smoke:
        trainer.save_model(str(out))
        tok.save_pretrained(str(out))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    app()
