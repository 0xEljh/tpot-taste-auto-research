#!/usr/bin/env bash
# Exhaustive frontier validation (doc 05 §10), serialized on one GPU.
# Standard recipe = LINEAR decay-to-0, 15 epochs, all-linear LoRA r16, lr 1e-4 (bs2 for 8B).
# Ordering is most-decisive-first so a reboot loses the least; per-phase guards (set +e) so one
# failure never aborts the chain. expandable_segments curbs fragmentation across many model reloads.
set +e
cd /home/elijah/tpot-taste-auto-research || exit 1
export HF_TOKEN="$(grep '^HF_TOKEN=' /home/elijah/dotfiles/scripts/.env | cut -d= -f2-)"
export WANDB_API_KEY="$(grep '^WANDB_API_KEY=' /home/elijah/dotfiles/scripts/.env | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/frontier_cv.log
: > "$LOG"
echo "[frontier-cv start] $(date -Is)" >> "$LOG"

phase () { echo "" >> "$LOG"; echo "=== $* ===" >> "$LOG"; }
cv () {  # $1=label $2=base $3=bs $4=scheduler $5=outdir
  phase "CV $4 — $1 (5-fold, n=177)"
  uv run python scripts/bakeoff_scorer.py --arm seqcls --cv --folds 5 --base "$2" --bs "$3" \
    --scheduler "$4" --out "$5" >> "$LOG" 2>&1 || echo "[CV FAILED: $1/$4]" >> "$LOG"
}
single () {  # $1=label $2=base $3=bs $4=lr $5=outdir
  phase "SINGLE (frozen-46) — $1  lr=$4 linear"
  uv run python scripts/bakeoff_scorer.py --arm seqcls --base "$2" --bs "$3" --lr "$4" \
    --scheduler linear --out "$5" >> "$LOG" 2>&1 || echo "[SINGLE FAILED: $1]" >> "$LOG"
}

# 1) PRIMARY: does Qwen3-4B-2507's 0.82 survive de-noising under the improved (linear) recipe?
cv "Qwen3-4B-Instruct-2507" "Qwen/Qwen3-4B-Instruct-2507" 4 linear outputs/scorer/cv-qwen3-4b

# 2) Gemma single-run (frozen-46) — the now-unblocked non-Qwen arm (also caches it for phase 7).
single "gemma-3-4b-it" "google/gemma-3-4b-it" 4 1e-4 outputs/scorer/bakeoff-gemma3-4b

# 3) Objective disentangle: BT (RewardTrainer) on Qwen3-4B-2507 vs its regression (0.82) & v7.3 BT (0.69).
phase "BT — Qwen3-4B-Instruct-2507 (frozen-46), linear"
uv run python scripts/train_scorer.py --train-pairs data/splits/taste_human_train_pairs.parquet \
    --base Qwen/Qwen3-4B-Instruct-2507 --out outputs/scorer/qwen3-4b-bt --epochs 2 \
    --batch-size 4 --grad-accum 4 --scheduler linear --run-name scorer-bt-qwen3-4b >> "$LOG" 2>&1 \
  && uv run python scripts/eval_scorer_vs_human.py --scorer outputs/scorer/qwen3-4b-bt \
       --base Qwen/Qwen3-4B-Instruct-2507 --heldout outputs/align_heldout_ids.json >> "$LOG" 2>&1 \
  || echo "[BT-4B FAILED]" >> "$LOG"

# 4) Confirm the Qwen2.5-3B floor (0.60 single) de-noised.
cv "Qwen2.5-3B-Instruct" "Qwen/Qwen2.5-3B-Instruct" 4 linear outputs/scorer/cv-qwen25-3b

# 5) Schedule ablation (the user's question): same candidate under COSINE vs the linear in phase 1.
cv "Qwen3-4B-Instruct-2507" "Qwen/Qwen3-4B-Instruct-2507" 4 cosine outputs/scorer/cv-qwen3-4b-cosine

# 6) Warm-start rescue: is Skywork-V2-4B's collapse an LR artifact? lr 2e-5 vs the 1e-4 that gave 0.50.
single "Skywork-V2-4B @lr2e-5" "Skywork/Skywork-Reward-V2-Qwen3-4B" 4 2e-5 outputs/scorer/bakeoff-skywork4b-lr2e5

# 7) Gemma de-noised (diversity).
cv "gemma-3-4b-it" "google/gemma-3-4b-it" 4 linear outputs/scorer/cv-gemma3-4b

# 8) Ceiling, de-noised (slowest — last).
cv "Qwen3-8B" "Qwen/Qwen3-8B" 2 linear outputs/scorer/cv-qwen3-8b

echo "" >> "$LOG"
echo "[frontier-cv done] $(date -Is)" >> "$LOG"
