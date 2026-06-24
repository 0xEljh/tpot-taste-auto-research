#!/usr/bin/env bash
# Phase 7 — Writer re-anchor on Qwen3-4B-Instruct-2507, signal re-anchored to the v8 (human-aligned) scorer.
#   curate goods (v8) -> build SFT data -> SFT (4B) -> build DPO pairs (v8-scored, bait-penalized) -> DPO (4B)
# The whole point: DPO-v2's prefs were v6-scored (platitude-loving). v8 flips platitude/bait to the REJECTED side,
# and the 4B base lifts the generation ceiling the diagnostic exposed (garbled 3B output on abstract topics).
#
#   STAGE=smoke bash scripts/run_writer_reanchor.sh   # trainers run 5 steps, no save — de-risk the 4B base
#   STAGE=full  bash scripts/run_writer_reanchor.sh   # full pipeline
set -euo pipefail
cd "$(dirname "$0")/.."

set -a; source /home/elijah/dotfiles/scripts/.env 2>/dev/null || true; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

STAGE="${STAGE:-full}"
BASE="Qwen/Qwen3-4B-Instruct-2507"
SFT_OUT="outputs/writer/qwen3-4b-sft-v3"
DPO_OUT="outputs/writer/qwen3-4b-dpo-v3"
SFT_DATA="data/splits/writer_sft_v3.jsonl"
DPO_DATA="data/splits/dpo_train_v3.jsonl"
GOODS="data/splits/curated_goods.parquet"
SMOKE_FLAG=""; [ "$STAGE" = "smoke" ] && SMOKE_FLAG="--smoke"

run() { echo; echo "=== $* ==="; uv run python "$@"; }

# 1. curate goods with the LOCKED (=v8) scorer: broad pool, English-filtered, bait-dropped, top-12k by taste
[ -f "$GOODS" ] && [ "${RECURATE:-0}" != "1" ] || run scripts/curate_goods.py --out "$GOODS"

# 2. SFT data: ideate from curated goods + improve from deopt pairs (both bait-filtered in make_sft_records)
run scripts/build_writer_sft.py --goods-path "$GOODS" --out "$SFT_DATA"

# 3. SFT on the 4B base (linear LR by default) — resumable: skip if the adapter exists (RESFT=1 to force)
if [ -f "$SFT_OUT/adapter_model.safetensors" ] && [ "$STAGE" = "full" ] && [ "${RESFT:-0}" != "1" ]; then
  echo "=== SFT exists ($SFT_OUT), skipping (RESFT=1 to force) ==="
else
  run scripts/train_writer.py --base "$BASE" --data "$SFT_DATA" --out "$SFT_OUT" \
      --batch-size 4 --grad-accum 8 --run-name writer-sft-v3-4b $SMOKE_FLAG
fi

# 4. DPO pairs: best-of-N from the 4B SFT, scored by v8 (LOCKED), bait sunk to rejected; v8-scale margin.
#    Resumable: skip if the pairs file exists (REPAIR=1 to force a regenerate).
if [ -f "$DPO_DATA" ] && [ "$STAGE" = "full" ] && [ "${REPAIR:-0}" != "1" ]; then
  echo "=== DPO pairs exist ($DPO_DATA), skipping (REPAIR=1 to force) ==="
else
  run scripts/build_dpo_pairs.py --writer "$SFT_OUT" --base "$BASE" \
      --out "$DPO_DATA" --min-margin 0.2 --bait-penalty 2.0 ${SMOKE_N:+--n-ideate 8 --n-improve 4}
fi

# 5. DPO on the 4B base from the SFT checkpoint (linear LR by default)
run scripts/train_dpo.py --base "$BASE" --sft-adapter "$SFT_OUT" --data "$DPO_DATA" --out "$DPO_OUT" \
    --run-name writer-dpo-v3-4b $SMOKE_FLAG

echo; echo "=== DONE: $DPO_OUT ==="
