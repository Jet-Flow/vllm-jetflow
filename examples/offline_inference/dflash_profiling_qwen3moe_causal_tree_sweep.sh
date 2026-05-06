#!/usr/bin/env bash
#
# Multi-dataset DFlash tree-verification sweep on Qwen3-30B-A3B (MoE target)
# + a causal-head DFlash draft.  For each dataset, runs both AR and DFlash
# tree mode and writes per-dataset metrics under $LOG_DIR/<dataset>/.
# Use scripts/summarize_tree_sweep.py (or
# examples/offline_inference/summarize_tree_sweep.py) to compile the table.
#
# Same tree-mode flags as dflash_profiling_qwen3moe_causal_tree_aime24.sh:
#   tree_width=7, max_tree_budget=255, accum_logp + breadth_first
#   (= HF entropy_guided when max_draft_passes=0),
#   --enforce-eager (tree_attn .item() blocks CUDAGraph capture),
#   --no-profile (tree-mode stop_trace hangs on huge buffers).
#
# Override datasets via env: ONLY="aime24:30 humaneval:164"

set -uo pipefail

DRAFT_MODEL="${DRAFT_MODEL:?need DRAFT_MODEL (e.g. specforge causal-head ckpt)}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-30B-A3B}"
TIMESTAMP="${TIMESTAMP:-$(date +%y%m%d-%H%M%S)}"
LOG_DIR="${LOG_DIR:-./logs/qwen3moe_tree_sweep-${TIMESTAMP}}"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "Run dir: $LOG_DIR"
echo "Target:  $TARGET_MODEL"
echo "Draft :  $DRAFT_MODEL"
echo "Time  :  $(date)"
echo ""

# (dataset_name:max_samples) — mirrors HF run_yulun_benchmark_moe_causal_tree.sh
TASKS=(
  "aime24:30"
  "aime25:30"
  "gsm8k:128"
  "math500:128"
  "humaneval:164"
  "mbpp:128"
  "livecodebench:128"
  "swe-bench:128"
  "mt-bench:80"
  "alpaca:128"
)
if [[ -n "${ONLY:-}" ]]; then
    read -ra TASKS <<< "$ONLY"
fi

for task in "${TASKS[@]}"; do
    IFS=':' read -r DATASET MAX_SAMPLES <<< "$task"
    DS_DIR="$LOG_DIR/$DATASET"
    DS_LOG="$DS_DIR/blk16_w7_impl_triton.log"
    PROFILER_DIR="$DS_DIR/profile"
    mkdir -p "$DS_DIR"

    echo ""
    echo "[$(date +%H:%M:%S)] $DATASET ($MAX_SAMPLES samples)"

    # GPU cleanup between runs (each invocation forks a new EngineCore).
    pkill -9 -f "dflash_profiling.py" 2>/dev/null || true
    pkill -9 -f "VLLM::EngineCore" 2>/dev/null || true
    sleep 3
    python -c "import torch; [torch.cuda.empty_cache() for _ in range(torch.cuda.device_count())]" 2>/dev/null || true

    python examples/offline_inference/dflash_profiling.py \
      --prompt-set "$DATASET" \
      --max-samples "$MAX_SAMPLES" \
      --mode both \
      --model "$TARGET_MODEL" \
      --draft-model "$DRAFT_MODEL" \
      --head-type causal \
      --block-size 16 \
      --max-tokens 2048 \
      --temperature 0.0 \
      --attention-backend FLASH_ATTN \
      --tp-sizes 1 \
      --batch-sizes 1 \
      --max-num-batched-tokens 51200 \
      --max-num-seqs 1 \
      --num-runs 1 \
      --num-warmup-runs 1 \
      --tree-width 7 \
      --tree-attn-kernel triton \
      --tree-draft accum_logp \
      --tree-construction breadth_first \
      --max-tree-budget 255 \
      --max-draft-passes 0 \
      --enforce-eager \
      --no-profile \
      --torch-profiler-dir "$PROFILER_DIR" \
      2>&1 | tee "$DS_LOG"

    rc=${PIPESTATUS[0]}
    echo "[$(date +%H:%M:%S)] $DATASET exit=$rc"
done

echo ""
echo "[$(date +%H:%M:%S)] sweep done"
echo "logs: $LOG_DIR"
