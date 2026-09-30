#!/usr/bin/env bash
# Score SyncReward-Bench with a Stage-2 checkpoint and report agreement with human ratings.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA_ROOT="${DATA_ROOT:?set DATA_ROOT}"
CKPT="${CKPT:?set CKPT (Stage-2 checkpoint dir, e.g. outputs/stage2/best)}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/eval}"

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE:-8}" \
  evaluate.py --config configs/eval.yaml \
  --checkpoint "${CKPT}" \
  --data "${DATA_ROOT}/syncreward_bench.jsonl" \
  --output "${OUTPUT_DIR}/predictions.jsonl" \
  --metrics_output "${OUTPUT_DIR}/metrics.json" \
  "$@"

# Evaluate any other model's predictions (jsonl with uid, model, reward) on the same targets:
#   python evaluate.py --predictions other_preds.jsonl --data "${DATA_ROOT}/syncreward_bench.jsonl"
