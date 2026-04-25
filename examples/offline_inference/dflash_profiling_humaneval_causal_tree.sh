#!/usr/bin/env bash
set -euo pipefail

export HF_DATASETS_CACHE="/data/data/hf_datasets"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1

export CUDA_HOME=/usr/local/cuda-12.8
export CUDA_PATH=/usr/local/cuda-12.8
export CUDACXX=/usr/local/cuda-12.8/bin/nvcc
export PATH=/usr/local/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64:${LD_LIBRARY_PATH:-}

# ── Defaults ──────────────────────────────────────────────────────────────
#DRAFT_MODEL="/data/specforge/outputs/data/specforge/trajectory_cache/nemotron-780k-and-codealpaca20k-greedy-3e-4-causal/epoch_1_step_70000"
#DRAFT_MODEL="/mnt/specdec-dev/checkpoints/specforge/outputs/nemotron-780k-and-codealpaca20k-greedy-3e-4-causal/epoch_2_step_199926/"
DRAFT_MODEL="/mnt/specdec-dev/checkpoints/specforge/outputs/nemotron-780k-and-codealpaca20k-v2-causal-distill-lr1e-4-anchorcnt512/epoch_6_step_583488"
#TREE_ATTN_KERNEL="triton"
TREE_ATTN_KERNEL="optimus"

# For causal tree with width > 1, attention backend is automatically tree_attn.
ATTENTION_BACKEND="FLASH_ATTN"
PROFILER_DIR=""
EXTRA_ARGS=()

# ── Parse named arguments ────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --draft-model)        DRAFT_MODEL="$2";        shift 2 ;;
    --tree-attn-kernel)   TREE_ATTN_KERNEL="$2";   shift 2 ;;
    --attention-backend)  ATTENTION_BACKEND="$2";   shift 2 ;;
    --profiler-dir)       PROFILER_DIR="$2";        shift 2 ;;
    *)                    EXTRA_ARGS+=("$1");       shift   ;;
  esac
done

DRAFT_TAG="$(basename "${DRAFT_MODEL}")"
DATE_TAG="$(date +%m%d)"

TREE_WIDTH=7
TREE_DEPTH=16
MAX_TREE_BUDGET=255
NUM_CUDAGRAPH_TREE_CAPTURES=4

#TREE_DRAFT_MODE="entropy"
TREE_DRAFT_MODE="accum_logp"
#TREE_DRAFT_MODE="hybrid"
#TREE_DRAFT_MODE="opt_prefix"

ADDITIONAL_DRAFT_REFINEMENT_PASSES=1

#TREE_DRAFT_MODE="entropy"
#TREE_DRAFT_MODE="hybrid"
TREE_PRUNE_RATIO=0.25
TREE_CONSTRUCTION="breadth_first"

if [[ -z "${PROFILER_DIR}" ]]; then
  PROFILER_DIR="/data/midas/vllm_qwen3_template_profile_${DRAFT_TAG}_${DATE_TAG}_humaneval_causal_${TREE_DRAFT_MODE}_${TREE_CONSTRUCTION}_tree_d${TREE_DEPTH}_w${TREE_WIDTH}_budget${MAX_TREE_BUDGET}_refinecnt_${ADDITIONAL_DRAFT_REFINEMENT_PASSES}_pruneratio_${TREE_PRUNE_RATIO}_tree_impl_${TREE_ATTN_KERNEL}"
fi

OPTIMUS_SRC="/home/i-hulanxiang/workspace/optimus_jit_local/src"
if [[ "${TREE_ATTN_KERNEL}" == "optimus" && -d "${OPTIMUS_SRC}" ]]; then
  export PYTHONPATH="${OPTIMUS_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

python examples/offline_inference/dflash_profiling.py \
  --prompt-set humaneval \
  --mode both \
  --head-type causal \
  --draft-model "${DRAFT_MODEL}" \
  --max-tokens 2048 \
  --block-size ${TREE_DEPTH} \
  --tree-width ${TREE_WIDTH} \
  --max-tree-budget ${MAX_TREE_BUDGET} \
  --tree-draft ${TREE_DRAFT_MODE} \
  --max-draft-passes ${ADDITIONAL_DRAFT_REFINEMENT_PASSES} \
  --tree-prune-ratio ${TREE_PRUNE_RATIO} \
  --tree-construction "${TREE_CONSTRUCTION}" \
  --tree-attn-kernel "${TREE_ATTN_KERNEL}" \
  --num-cudagraph-tree-captures ${NUM_CUDAGRAPH_TREE_CAPTURES} \
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
