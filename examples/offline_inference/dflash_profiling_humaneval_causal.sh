#!/usr/bin/env bash
set -euo pipefail

# ── Defaults ──────────────────────────────────────────────────────────────
#DRAFT_MODEL="/mnt/specdec-dev/checkpoints/specforge/outputs/nemotron-780k-and-codealpaca20k-greedy-3e-4-causal/epoch_2_step_199926/"
DRAFT_MODEL="/mnt/specdec-dev/checkpoints/specforge/outputs/nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/epoch_6_step_583488"
ATTENTION_BACKEND="FLASH_ATTN"
PROFILER_DIR=""
EXTRA_ARGS=()

# ── Parse named arguments ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --draft-model)        DRAFT_MODEL="$2";        shift 2 ;;
    --attention-backend)  ATTENTION_BACKEND="$2";   shift 2 ;;
    --profiler-dir)       PROFILER_DIR="$2";        shift 2 ;;
    *)                    EXTRA_ARGS+=("$1");       shift   ;;
  esac
done

DRAFT_TAG="$(basename "${DRAFT_MODEL}")"
DATE_TAG="$(date +%m%d)"

if [[ -z "${PROFILER_DIR}" ]]; then
  PROFILER_DIR="/data/midas/vllm_profile_dflash_${DATE_TAG}_humaneval_causal_linear_${DRAFT_TAG}_qwen3_template"
fi

python examples/offline_inference/dflash_profiling.py \
  --prompt-set humaneval \
  --mode both \
  --head-type causal \
  --draft-model "${DRAFT_MODEL}" \
  --max-tokens 2048 \
  --block-size 16 \
  --attention-backend "${ATTENTION_BACKEND}" \
  --tp-sizes 1 \
  --batch-sizes 1 \
  --max-num-batched-tokens 51200 \
  --max-samples 4 \
  --max-num-seqs 1 \
  --num-runs 1 \
  --num-warmup-runs 1 \
  "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
  --torch-profiler-dir "${PROFILER_DIR}"
