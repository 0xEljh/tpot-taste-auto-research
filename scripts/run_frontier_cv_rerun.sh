#!/usr/bin/env bash
# Re-run the 3 frontier arms that died on INFRA bugs (not quality findings) in run_frontier_cv.sh:
#   A) Gemma single-run  — was: ValueError token_type_ids required (Gemma3 mask). Fixed: train
#      collator now injects all-zero token_type_ids (text-only) for gemma bases.
#   B) Gemma CV          — same fix, in-process 5-fold (4B fits 12GB).
#   C) Qwen3-8B CV       — was: CUDA OOM at fold 3 (in-process loop leaks across reloads). Fixed:
#      one fresh process PER FOLD (OS reclaims all GPU mem between folds), then --cv-aggregate.
# Per-phase guards (set +e) so one failure never aborts the chain. Logs -> outputs/frontier_cv_rerun.log
set +e
cd /home/elijah/tpot-taste-auto-research || exit 1
export HF_TOKEN="$(grep '^HF_TOKEN=' /home/elijah/dotfiles/scripts/.env | cut -d= -f2-)"
export WANDB_API_KEY="$(grep '^WANDB_API_KEY=' /home/elijah/dotfiles/scripts/.env | cut -d= -f2-)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=outputs/frontier_cv_rerun.log
: > "$LOG"
echo "[rerun start] $(date -Is)" >> "$LOG"
phase () { echo "" >> "$LOG"; echo "=== $* ===" >> "$LOG"; }

# A) Gemma single-run (frozen-46) — fastest, early signal that the collator fix holds.
phase "SINGLE (frozen-46) — gemma-3-4b-it  lr=1e-4 linear"
uv run python scripts/bakeoff_scorer.py --arm seqcls --base google/gemma-3-4b-it --bs 4 \
  --scheduler linear --out outputs/scorer/bakeoff-gemma3-4b >> "$LOG" 2>&1 \
  || echo "[SINGLE FAILED: gemma-3-4b-it]" >> "$LOG"

# B) Gemma CV (5-fold, in-process — 4B fits).
phase "CV linear — gemma-3-4b-it (5-fold, n=177)"
uv run python scripts/bakeoff_scorer.py --arm seqcls --cv --folds 5 --base google/gemma-3-4b-it \
  --bs 4 --scheduler linear --out outputs/scorer/cv-gemma3-4b >> "$LOG" 2>&1 \
  || echo "[CV FAILED: gemma-3-4b-it/linear]" >> "$LOG"

# C) Qwen3-8B CV — one fresh process per fold (fixes the cross-fold OOM), then aggregate the dumps.
phase "CV linear — Qwen3-8B (5-fold, n=177, per-fold subprocess)"
rm -f outputs/scorer/cv-qwen3-8b/fold*_oof.json
for k in 0 1 2 3 4; do
  echo "  -- launching 8B fold $k (fresh process) --" >> "$LOG"
  uv run python scripts/bakeoff_scorer.py --arm seqcls --cv --fold "$k" --folds 5 \
    --base Qwen/Qwen3-8B --bs 2 --scheduler linear --out outputs/scorer/cv-qwen3-8b >> "$LOG" 2>&1 \
    || echo "[8B fold $k FAILED]" >> "$LOG"
done
uv run python scripts/bakeoff_scorer.py --arm seqcls --cv-aggregate --folds 5 \
  --base Qwen/Qwen3-8B --out outputs/scorer/cv-qwen3-8b >> "$LOG" 2>&1 \
  || echo "[8B aggregate FAILED]" >> "$LOG"

echo "" >> "$LOG"
echo "[rerun done] $(date -Is)" >> "$LOG"
