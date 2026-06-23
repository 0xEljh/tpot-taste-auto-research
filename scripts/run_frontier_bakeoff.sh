#!/usr/bin/env bash
# Frontier bake-off + hi_lo BT diagnostic (doc 05 §9), serialized on one GPU.
# Ordering is reboot-resilient: cheap/high-value first, Gemma (needs download) last.
# One arm failing must NOT abort the chain (set +e + per-arm guard).
set +e
cd /home/elijah/tpot-taste-auto-research || exit 1
export HF_TOKEN="$(grep '^HF_TOKEN=' /home/elijah/dotfiles/scripts/.env | cut -d= -f2-)"
LOG=outputs/bakeoff_frontier2.log
: > "$LOG"
echo "[frontier start] $(date -Is)" >> "$LOG"

# arm 0 (Qwen2.5-3B seqcls) already ran pre-reboot: pw=0.60 rho=+0.15 -> reused, not re-run.
echo "[note] reusing arm Qwen2.5-3B-Instruct from bakeoff_frontier.log: pw=0.60 rho=+0.15" >> "$LOG"

run_arm () {  # $1=base  $2=batch_size
  echo "" >> "$LOG"
  echo "=== ARM: $1 (bs=$2) ===" >> "$LOG"
  local odir="outputs/scorer/bakeoff-$(echo "$1" | tr '/' '-')"
  uv run python scripts/bakeoff_scorer.py --arm seqcls --base "$1" --bs "$2" --out "$odir" >> "$LOG" 2>&1 \
    || echo "[ARM FAILED: $1]" >> "$LOG"
}

# 1) hi_lo-only Bradley-Terry diagnostic FIRST: tests whether dropping the borderline round-3
#    pairs recovers v7.2-like BT performance (the dipstick-1 hypothesis).
echo "" >> "$LOG"
echo "=== HI_LO BT DIAGNOSTIC (tpot>not pairs only, borderline dropped) ===" >> "$LOG"
uv run python scripts/train_scorer.py \
    --train-pairs data/splits/taste_human_hilo_pairs.parquet \
    --out outputs/scorer/qwen3b-bt-v73hilo --epochs 2 --run-name scorer-bt-v73hilo-diag >> "$LOG" 2>&1 \
  && uv run python scripts/eval_scorer_vs_human.py \
       --scorer outputs/scorer/qwen3b-bt-v73hilo --base Qwen/Qwen2.5-3B-Instruct \
       --heldout outputs/align_heldout_ids.json >> "$LOG" 2>&1 \
  || echo "[HI_LO DIAGNOSTIC FAILED]" >> "$LOG"

# 2) frontier seqcls arms: 4B first, 8B at bs=2 (12GB), Gemma last (needs ~8GB download).
run_arm "Skywork/Skywork-Reward-V2-Qwen3-4B" 4
run_arm "Qwen/Qwen3-4B-Instruct-2507" 4
run_arm "Skywork/Skywork-Reward-V2-Qwen3-8B" 2
run_arm "Qwen/Qwen3-8B" 2
run_arm "google/gemma-3-4b-it" 4

echo "" >> "$LOG"
echo "[bakeoff frontier done] $(date -Is)" >> "$LOG"
