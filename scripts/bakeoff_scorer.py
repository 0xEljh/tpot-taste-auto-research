"""Scorer bake-off (doc 05 §7): representation × objective, on the SAME human held-out.

Tests the headline research claim on OUR data — that for a ~140-label idiosyncratic taste,
objective+representation beat base size, and the current 3B+Bradley-Terry combo is the weakest.

Arms:
  embed-mlp    frozen sentence-embedding + sklearn head (Ridge / MLP)   [seconds, no train()]
  regress-3b   Qwen2.5-3B + regression head (pointwise MSE, QLoRA)      [minutes, GPU]
  modernbert   ModernBERT-large fine-tuned regressor (LoRA)             [minutes, GPU]
  seqcls       ANY HF base --base <id> as 4-bit seq-cls regressor (LoRA all-linear)   [minutes, GPU]
Baseline (3B + Bradley-Terry, v7.2) is from scripts/eval_scorer_vs_human.py: held-out pw=0.72 / rho=0.39.

  uv run python scripts/bakeoff_scorer.py --arm embed-mlp
  uv run python scripts/bakeoff_scorer.py --arm regress-3b --smoke   # fast GPU sanity
  uv run python scripts/bakeoff_scorer.py --arm regress-3b
  uv run python scripts/bakeoff_scorer.py --arm modernbert
  # the 4B/8B + Gemma + warm-start Skywork-Reward-V2 frontier (doc 05 §9):
  uv run python scripts/bakeoff_scorer.py --arm seqcls --base Skywork/Skywork-Reward-V2-Qwen3-4B
  uv run python scripts/bakeoff_scorer.py --arm seqcls --base Skywork/Skywork-Reward-V2-Qwen3-8B --bs 2
  uv run python scripts/bakeoff_scorer.py --arm seqcls --base Qwen/Qwen3-8B --bs 2
  uv run python scripts/bakeoff_scorer.py --arm seqcls --base google/gemma-3-4b-it
"""
from __future__ import annotations

import json
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, pretty_exceptions_enable=False)

BASELINE = "  baseline 3B+BT (v7.2): held-out pw=0.72  rho=+0.39   (random pw=0.50)"


@app.command()
def main(
    arm: str = "embed-mlp",
    calib: Path = Path("data/splits/calibration_set.parquet"),
    labels: Path = Path("outputs/calibration_labels.json"),
    heldout: Path = Path("outputs/align_heldout_ids.json"),
    embedder: str = "sentence-transformers/all-MiniLM-L6-v2",
    head: str = "both",
    base: str = "Qwen/Qwen2.5-3B-Instruct",
    out: Path = Path("outputs/scorer/bakeoff-tmp"),
    epochs: float = 15.0,
    lr: float = 1e-4,
    bs: int = 4,
    seed: int = 0,
    smoke: bool = False,
    cv: bool = False,
    folds: int = 5,
    fold: int = -1,
    cv_aggregate: bool = False,
    scheduler: str = "linear",
    save_adapter: bool = False,
) -> None:
    import polars as pl

    from tpot_taste.scoring.bakeoff import load_labels, pointwise_rows

    lab = load_labels(labels)
    held_ids = set(json.loads(heldout.read_text())) if heldout.exists() else set()
    rows = [r for r in pl.read_parquet(calib).to_dicts() if r["id"] in lab]
    all_pw = pointwise_rows(rows, lab, {r["id"] for r in rows})
    train_pw = pointwise_rows(rows, lab, {r["id"] for r in rows} - held_ids)
    held_pw = pointwise_rows(rows, lab, held_ids)
    print(f"[data] {len(rows)} labeled | {len(train_pw)} train | {len(held_pw)} held-out | arm={arm}")
    print(BASELINE)

    if arm == "embed-mlp":
        _embed_mlp(train_pw, held_pw, all_pw, embedder, head, seed)
    elif arm == "regress-3b":
        _regress_3b(train_pw, held_pw, base, out, epochs, lr, seed, smoke)
    elif arm == "modernbert":
        _modernbert(train_pw, held_pw, out, epochs, lr, seed, smoke)
    elif arm == "seqcls":
        if cv_aggregate:
            _cv_aggregate(all_pw, base, out, folds)
        elif cv and fold >= 0:
            _seqcls_cv_fold(all_pw, base, out, epochs, lr, seed, smoke, bs, folds, fold, scheduler)
        elif cv:
            _seqcls_cv(all_pw, base, out, epochs, lr, seed, smoke, bs, folds, scheduler)
        else:
            _seqcls(train_pw, held_pw, base, out, epochs, lr, seed, smoke, bs, scheduler, save_adapter)
    else:
        raise SystemExit(f"unknown arm '{arm}'")


def _embed_mlp(train_pw, held_pw, all_pw, embedder: str, head: str, seed: int) -> None:
    import numpy as np
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    from sklearn.neural_network import MLPRegressor

    from tpot_taste.data.embed import embed_texts, load_embedder
    from tpot_taste.scoring.bakeoff import human_eval

    model, tok = load_embedder(embedder)

    def emb(pw):
        return embed_texts(model, tok, [t for _, t, _ in pw])

    Xtr, ytr = emb(train_pw), np.array([y for *_, y in train_pw])
    Xhe, yhe = emb(held_pw), np.array([y for *_, y in held_pw])
    Xall, yall = emb(all_pw), np.array([y for *_, y in all_pw])

    heads = {
        "ridge": Ridge(alpha=10.0),
        "mlp": MLPRegressor(hidden_layer_sizes=(128,), alpha=1e-2, max_iter=2000,
                            early_stopping=True, random_state=seed),
    }
    sel = heads if head == "both" else {head: heads[head]}
    for name, reg in sel.items():
        reg.fit(Xtr, ytr)
        he = human_eval(reg.predict(Xhe), yhe)
        oof = cross_val_predict(reg, Xall, yall, cv=KFold(5, shuffle=True, random_state=seed))
        cv = human_eval(oof, yall)
        print(f"  [{name:5}] held-out: pw={he['pairwise']:.2f} rho={he['spearman']:+.2f} "
              f"prec={he['precision']:.2f} (n={he['n']}, {he['n_tpot']}tpot/{he['n_not']}not)"
              f"   | 5-fold CV({len(yall)}): pw={cv['pairwise']:.2f} rho={cv['spearman']:+.2f}")


def _train_seqcls_regression(model, tok, train_pw, *, epochs, lr, batch_size, out,
                             smoke, grad_ckpt=True, max_length=256, scheduler="linear",
                             needs_token_type_ids=False):
    """Fine-tune any AutoModelForSequenceClassification(num_labels=1) as an MSE regressor on
    pointwise (text, target in {0,0.5,1}) rows. num_labels=1 + float labels => HF uses MSELoss.

    needs_token_type_ids: Gemma3's mask builder raises during training if token_type_ids is None
    (token_type==1 marks image tokens; text-only => all zeros). It's gated on self.training, so
    eval/score_texts is exempt — only the train collator must supply the zeros."""
    import torch
    from datasets import Dataset
    from transformers import DataCollatorWithPadding, Trainer, TrainingArguments

    ds = Dataset.from_dict({"text": [t for _, t, _ in train_pw],
                            "labels": [float(y) for *_, y in train_pw]})
    ds = ds.map(lambda ex: tok(ex["text"], truncation=True, max_length=max_length), batched=True)
    ds = ds.remove_columns(["text"])
    data_collator = DataCollatorWithPadding(tok)
    if needs_token_type_ids:
        _pad = DataCollatorWithPadding(tok)

        def data_collator(features):  # noqa: F811 — Gemma3 text-only: all-zero token_type_ids
            batch = _pad(features)
            batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
            return batch

    args = TrainingArguments(
        output_dir=str(out), per_device_train_batch_size=batch_size, gradient_accumulation_steps=1,
        num_train_epochs=(1 if smoke else epochs), max_steps=(4 if smoke else -1),
        learning_rate=lr, bf16=True, gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        optim="paged_adamw_8bit", warmup_ratio=0.03, lr_scheduler_type=scheduler, max_grad_norm=1.0,
        logging_steps=5, save_strategy="no", report_to=[], dataloader_num_workers=2,
    )
    Trainer(model=model, args=args, train_dataset=ds,
            data_collator=data_collator, processing_class=tok).train()
    model.eval()
    return model


def _eval_seqcls(model, tok, held_pw, label: str) -> None:
    import numpy as np

    from tpot_taste.scoring.bakeoff import human_eval
    from tpot_taste.scoring.model import score_texts

    s = score_texts(model, tok, [t for _, t, _ in held_pw], batch_size=16)
    he = human_eval(s, np.array([y for *_, y in held_pw]))
    print(f"  [{label}] held-out: pw={he['pairwise']:.2f} rho={he['spearman']:+.2f} "
          f"prec={he['precision']:.2f} (n={he['n']}, {he['n_tpot']}tpot/{he['n_not']}not)")


def _regress_3b(train_pw, held_pw, base, out, epochs, lr, seed, smoke) -> None:
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    from tpot_taste.scoring.model import load_reward_model

    model, tok = load_reward_model(base, four_bit=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type=TaskType.SEQ_CLS,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    model.config.pad_token_id = tok.pad_token_id
    _train_seqcls_regression(model, tok, train_pw, epochs=epochs, lr=lr, batch_size=8,
                             out=out, smoke=smoke, grad_ckpt=True)
    _eval_seqcls(model, tok, held_pw, "regress-3b" + ("/smoke" if smoke else ""))


def _seqcls(train_pw, held_pw, base, out, epochs, lr, seed, smoke, batch_size=4, scheduler="linear",
            save_adapter=False) -> None:
    """Generic frontier arm: load ANY HF model as a 4-bit AutoModelForSequenceClassification(num_labels=1)
    and LoRA-fine-tune it as a pointwise taste regressor. Covers the 4B/8B + Gemma bases AND the warm-start
    Skywork-Reward-V2 RMs — those already ship a trained num_labels=1 head, so load_reward_model loads a
    *calibrated* reward head here rather than a random one (warm-start). Same all-linear LoRA recipe as
    --arm modernbert, so across the bake-off only the BASE varies (apples-to-apples; regression objective,
    which the research found >= Bradley-Terry at <=8B). --bs 2 for the 8B bases to stay inside 12GB."""
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    from tpot_taste.scoring.model import load_reward_model

    model, tok = load_reward_model(base, four_bit=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_CLS, target_modules="all-linear"))
    model.config.pad_token_id = tok.pad_token_id
    _train_seqcls_regression(model, tok, train_pw, epochs=epochs, lr=lr, batch_size=batch_size,
                             out=out, smoke=smoke, grad_ckpt=True, scheduler=scheduler,
                             needs_token_type_ids="gemma" in base.lower())
    if save_adapter and not smoke:
        model.save_pretrained(str(out))
        tok.save_pretrained(str(out))
        (Path(out) / "train_meta.json").write_text(json.dumps(
            {"base": base, "objective": "regression-pointwise", "scheduler": scheduler,
             "n_train": len(train_pw), "epochs": epochs, "lr": lr, "batch_size": batch_size,
             "lora_r": 16, "lora_alpha": 32, "target_modules": "all-linear", "max_length": 256},
            indent=2))
        print(f"[saved] adapter -> {out}")
    _eval_seqcls(model, tok, held_pw, f"seqcls:{base.split('/')[-1]}" + ("/smoke" if smoke else ""))


def _train_and_score_fold(all_pw, base, out_fold, tr, te, *, epochs, lr, batch_size,
                          smoke, scheduler, needs_tti):
    """Train a fresh 4-bit LoRA regressor on `tr`, score `te`, free the GPU, return the scores.

    Self-contained (loads + frees its own model) so it works identically whether called in a
    loop (small bases) or as the entire body of a per-fold subprocess (heavy bases that OOM
    an in-process loop). Returns a numpy score array aligned to `te`."""
    import gc

    import torch
    from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training

    from tpot_taste.scoring.model import load_reward_model, score_texts

    texts = [t for _, t, _ in all_pw]
    model, tok = load_reward_model(base, four_bit=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_CLS, target_modules="all-linear"))
    model.config.pad_token_id = tok.pad_token_id
    _train_seqcls_regression(model, tok, [all_pw[i] for i in tr], epochs=epochs, lr=lr,
                             batch_size=batch_size, out=out_fold, smoke=smoke, grad_ckpt=True,
                             scheduler=scheduler, needs_token_type_ids=needs_tti)
    s = score_texts(model, tok, [texts[i] for i in te], batch_size=16)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return s


def _seqcls_cv(all_pw, base, out, epochs, lr, seed, smoke, batch_size, folds, scheduler="linear") -> None:
    """De-noise the single-split frontier numbers: stratified k-fold CV of the seqcls
    regressor over ALL labeled items. A fresh 4-bit LoRA model is trained per fold on the
    K-1 training folds and scores the held fold; out-of-fold predictions are aggregated into
    ONE human_eval over all N (n=46 single-split is ±0.07; CV over 177 is the robust read).
    In-process loop — fine for <=4B; use --fold/--cv-aggregate for 8B (see _seqcls_cv_fold)."""
    import numpy as np

    from tpot_taste.scoring.bakeoff import cv_fold_indices, human_eval

    needs_tti = "gemma" in base.lower()
    y = np.array([v for *_, v in all_pw], dtype=float)
    oof = np.full(len(y), np.nan)
    splits = cv_fold_indices(y, n_splits=folds, seed=seed)
    for k, (tr, te) in enumerate(splits):
        s = _train_and_score_fold(all_pw, base, f"{out}/fold{k}", tr, te, epochs=epochs, lr=lr,
                                  batch_size=batch_size, smoke=smoke, scheduler=scheduler, needs_tti=needs_tti)
        oof[te] = s
        fe = human_eval(s, y[te])
        print(f"  [cv fold {k + 1}/{len(splits)}] n={len(te)} pw={fe['pairwise']:.2f} rho={fe['spearman']:+.2f}")
    cv = human_eval(oof, y)
    print(f"  [seqcls-cv:{base.split('/')[-1]}] {len(splits)}-fold CV(n={len(y)}): "
          f"pw={cv['pairwise']:.2f} rho={cv['spearman']:+.2f} prec={cv['precision']:.2f} "
          f"({cv['n_tpot']}tpot/{cv['n_not']}not)")


def _seqcls_cv_fold(all_pw, base, out, epochs, lr, seed, smoke, batch_size, folds, fold,
                    scheduler="linear") -> None:
    """Run ONE CV fold and dump its out-of-fold predictions to {out}/fold{fold}_oof.json.

    For heavy bases (8B) the in-process 5-reload loop leaks GPU memory and OOMs mid-run; running
    each fold as its own process lets the OS reclaim everything between folds. Splits are
    recomputed from the same (seed, data) so every fold-process sees the identical partition."""
    import json as _json

    import numpy as np

    from tpot_taste.scoring.bakeoff import cv_fold_indices, human_eval

    needs_tti = "gemma" in base.lower()
    y = np.array([v for *_, v in all_pw], dtype=float)
    splits = cv_fold_indices(y, n_splits=folds, seed=seed)
    tr, te = splits[fold]
    s = _train_and_score_fold(all_pw, base, f"{out}/fold{fold}", tr, te, epochs=epochs, lr=lr,
                              batch_size=batch_size, smoke=smoke, scheduler=scheduler, needs_tti=needs_tti)
    fe = human_eval(s, y[te])
    print(f"  [cv fold {fold + 1}/{len(splits)}] n={len(te)} pw={fe['pairwise']:.2f} rho={fe['spearman']:+.2f}")
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / f"fold{fold}_oof.json").write_text(_json.dumps(
        {"idx": [int(i) for i in te], "score": [float(x) for x in s],
         "target": [float(y[i]) for i in te]}))


def _cv_aggregate(all_pw, base, out, folds) -> None:
    """Glue the per-fold *_oof.json dumps into one OOF metric (the subprocess-CV counterpart of
    the in-process summary line). Reports n<N if some folds are missing rather than fabricating."""
    import json as _json

    import numpy as np

    from tpot_taste.scoring.bakeoff import assemble_oof, human_eval

    y = np.array([v for *_, v in all_pw], dtype=float)
    fold_files = sorted(Path(out).glob("fold*_oof.json"))
    fold_data = [_json.loads(f.read_text()) for f in fold_files]
    oof = assemble_oof(fold_data, len(y))
    mask = ~np.isnan(oof)
    cv = human_eval(oof[mask], y[mask])
    print(f"  [seqcls-cv:{base.split('/')[-1]}] {len(fold_data)} folds, CV(n={int(mask.sum())}/{len(y)}): "
          f"pw={cv['pairwise']:.2f} rho={cv['spearman']:+.2f} prec={cv['precision']:.2f} "
          f"({cv['n_tpot']}tpot/{cv['n_not']}not)")


def _modernbert(train_pw, held_pw, out, epochs, lr, seed, smoke,
                model_id="answerdotai/ModernBERT-large") -> None:
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id, num_labels=1, dtype=torch.bfloat16)
    if torch.cuda.is_available():
        model = model.cuda()
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type=TaskType.SEQ_CLS, target_modules="all-linear"))
    _train_seqcls_regression(model, tok, train_pw, epochs=epochs, lr=lr, batch_size=16,
                             out=out, smoke=smoke, grad_ckpt=False)
    _eval_seqcls(model, tok, held_pw, "modernbert" + ("/smoke" if smoke else ""))


if __name__ == "__main__":
    app()
