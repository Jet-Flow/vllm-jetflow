#!/usr/bin/env bash
#
# Single-dataset DFlash tree-verification eval on Qwen3-30B-A3B (MoE target)
# + a causal-head DFlash draft, on aime24.  Used as a quick alignment check
# against the HF-backend benchmark (inference/eval_scripts/run_yulun_benchmark
# _moe_causal_tree.sh).
#
# Tree config matches HF's build_tree_entropy_guided when
# max_draft_passes=0: cumulative-logprob best-first heap expansion
# (--tree-draft accum_logp --tree-construction breadth_first).
#
# Why --enforce-eager: tree_attn.py:118 calls .item() during prefill_metadata
# construction, which is illegal under CUDAGraph stream capture.
# Why --no-profile: tree-mode runs 255 nodes per step; torch.profiler.stop_trace
# hangs while serializing the resulting buffer.

set -euo pipefail

DRAFT_MODEL="${DRAFT_MODEL:?need DRAFT_MODEL (e.g. specforge causal-head ckpt)}"
TARGET_MODEL="${TARGET_MODEL:-Qwen/Qwen3-30B-A3B}"
PROFILER_DIR="${PROFILER_DIR:-./vllm_qwen3moe_aime24_tree_profile}"
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --draft-model)        DRAFT_MODEL="$2";        shift 2 ;;
    --target-model)       TARGET_MODEL="$2";       shift 2 ;;
    --profiler-dir)       PROFILER_DIR="$2";       shift 2 ;;
    *)                    EXTRA_ARGS+=("$1");      shift   ;;
  esac
done

python examples/offline_inference/dflash_profiling.py \
  --prompt-set aime24 \
  --max-samples 30 \
  --mode both \
  --model "${TARGET_MODEL}" \
  --draft-model "${DRAFT_MODEL}" \
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
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
  --torch-profiler-dir "${PROFILER_DIR}"
