"""SFT the Taste Writer (Phase 3, D18): Qwen2.5-3B-Instruct, TRL SFTTrainer + 4-bit QLoRA.

Full-text SFT over chat-formatted ideate+improve records. assistant_only_loss is deferred:
Qwen's chat template lacks the {% generation %} tags TRL needs for it, and the prompts are short,
so full-text loss is fine for v1 (revisit with a custom template if the model wastes capacity).

  uv run python scripts/train_writer.py --smoke     # 5 steps, no W&B, prints a batch sanity check
  uv run python scripts/train_writer.py             # full run
"""
from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)


@app.command()
def main(
    data: Path = Path("data/splits/writer_sft_train.jsonl"),
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    out: Path = Path("outputs/writer/qwen3b-sft-v1"),
    epochs: float = 1.0,
    batch_size: int = 8,
    grad_accum: int = 4,
    lr: float = 2e-4,
    max_length: int = 512,
    run_name: str = "writer-sft-v1",
    smoke: bool = False,
) -> None:
    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    ds = load_dataset("json", data_files=str(data), split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c != "messages"])  # SFTTrainer wants `messages`
    if smoke:
        ds = ds.select(range(64))
    print(f"[data] {ds.num_rows:,} records")

    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, dtype=torch.bfloat16)
    model.config.use_cache = False

    lora = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=str(out),
        num_train_epochs=epochs,
        max_steps=(5 if smoke else -1),
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_length=max_length,
        packing=False,
        assistant_only_loss=False,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        logging_steps=(1 if smoke else 10),
        save_strategy=("no" if smoke else "epoch"),
        report_to=("none" if smoke else "wandb"),
        run_name=run_name,
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok, peft_config=lora)

    if smoke:  # verify formatting + a real batch before committing GPU-hours
        batch = next(iter(trainer.get_train_dataloader()))
        ids = batch["input_ids"]
        n_lab = int((batch["labels"] != -100).sum())
        print(f"[smoke] batch keys={list(batch.keys())} input_ids={tuple(ids.shape)} "
              f"labels!=-100: {n_lab}/{batch['labels'].numel()}")
        print("[smoke] decoded[0]:", tok.decode(ids[0][ids[0] != tok.pad_token_id][:60]))

    trainer.train()
    if not smoke:
        trainer.save_model(str(out))
        tok.save_pretrained(str(out))
    print(f"[done] -> {out}")


if __name__ == "__main__":
    app()
