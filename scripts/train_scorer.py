"""Train the Taste Scorer: Qwen2.5-3B + scalar reward head, Bradley-Terry on pairs.

  uv run python scripts/train_scorer.py --smoke      # fast GPU sanity (4 steps, no save/wandb)
  uv run python scripts/train_scorer.py              # full run -> outputs/scorer/qwen3b-bt-v1
"""
from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    train_pairs: Path = Path("data/splits/train_pairs.parquet"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    out: Path = Path("outputs/scorer/qwen3b-bt-v1"),
    epochs: float = 1.0,
    batch_size: int = 8,
    grad_accum: int = 2,
    lr: float = 1e-4,
    max_length: int = 256,
    lora_r: int = 16,
    lora_alpha: int = 32,
    run_name: str = "scorer-bt-v1",
    smoke: bool = False,
) -> None:
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from trl import RewardConfig, RewardTrainer

    from tpot_taste.scoring.model import load_reward_model

    df = pl.read_parquet(train_pairs)
    ds = Dataset.from_dict({"chosen": df["text_clean_w"].to_list(), "rejected": df["text_clean_l"].to_list()})
    if smoke:
        ds = ds.select(range(min(64, len(ds))))
    print(f"[data] {len(ds):,} pairs ({'SMOKE' if smoke else 'full'})")

    model, tok = load_reward_model(base, four_bit=True)
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    args = RewardConfig(
        output_dir=str(out),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        num_train_epochs=epochs,
        max_steps=4 if smoke else -1,
        learning_rate=lr,
        max_length=max_length,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        max_grad_norm=1.0,
        center_rewards_coefficient=0.01,
        logging_steps=1 if smoke else 20,
        save_strategy="no" if smoke else "epoch",
        report_to=([] if smoke else ["wandb"]),
        run_name=run_name,
        dataloader_num_workers=2,
    )

    trainer = RewardTrainer(
        model=model, args=args, train_dataset=ds, processing_class=tok, peft_config=peft_config
    )
    trainer.train()

    if not smoke:
        trainer.save_model(str(out))
        tok.save_pretrained(str(out))
        (Path(out) / "train_meta.json").write_text(
            json.dumps(
                {"base": base, "n_pairs": len(ds), "epochs": epochs, "batch_size": batch_size,
                 "grad_accum": grad_accum, "lr": lr, "max_length": max_length,
                 "lora_r": lora_r, "lora_alpha": lora_alpha}, indent=2
            )
        )
        print(f"[done] saved adapter -> {out}")
    else:
        print("[smoke] OK — training step ran without error")


if __name__ == "__main__":
    app()
