#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)
REPO_ROOT=$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)

export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/root/data/cache}"
export VLLM_ALLOW_INSECURE_SERIALIZATION=1
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

if [[ -z "${CUDA_HOME:-}" || ! -d "${CUDA_HOME}" ]]; then
  if [[ -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
fi
export CUDA_PATH="${CUDA_HOME:-${CUDA_PATH:-}}"
if [[ -n "${CUDA_HOME:-}" && -x "${CUDA_HOME}/bin/nvcc" ]]; then
  export CUDACXX="$CUDA_HOME/bin/nvcc"
  export PATH="$CUDA_HOME/bin:$PATH"
fi

prepend_ld_path() {
  local path="$1"
  if [[ -d "$path" && ":${LD_LIBRARY_PATH:-}:" != *":${path}:"* ]]; then
    export LD_LIBRARY_PATH="${path}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
}
prepend_ld_path "${CUDA_HOME:-}/lib64"
prepend_ld_path "/usr/local/cuda/compat/lib.real"
prepend_ld_path "/usr/local/cuda/compat/lib"
prepend_ld_path "/usr/local/nvidia/lib64"
prepend_ld_path "/usr/local/nvidia/lib"

TARGET_MODEL="${TARGET_MODEL:-/root/models/Qwen3-8B}"
DRAFT_MODEL="${DRAFT_MODEL:-/root/data/outputs/dflash-qwen3-8b-causal-bs16-anc1-forwardkl-lr3e-4-gNone/epoch_6_step_291744_forward_kl}"
TREE_ATTN_KERNEL="${TREE_ATTN_KERNEL:-optimus}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-FLASH_ATTN}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
OPTIMUS_SRC="${OPTIMUS_SRC:-/root/workspace/optimus_jit_local/src}"

BATCH_SIZES="${BATCH_SIZES:-1 2 4 8 16}"
TREE_BUDGETS="${TREE_BUDGETS:-16 32 64 128 256}"
TP_SIZE="${TP_SIZE:-1}"
TREE_KV_LAYOUT="${TREE_KV_LAYOUT:-logical}"
PROFILER_DIR="${PROFILER_DIR:-}"
RUN_AR=1
RUN_DFLASH=1

TREE_WIDTH=7
TREE_DEPTH=16
NUM_CUDAGRAPH_TREE_CAPTURES="${NUM_CUDAGRAPH_TREE_CAPTURES:-4}"
TREE_DRAFT_MODE="${TREE_DRAFT_MODE:-accum_logp}"
ADDITIONAL_DRAFT_REFINEMENT_PASSES="${ADDITIONAL_DRAFT_REFINEMENT_PASSES:-0}"
TREE_PRUNE_RATIO="${TREE_PRUNE_RATIO:-0.25}"
TREE_CONSTRUCTION="${TREE_CONSTRUCTION:-breadth_first}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-51200}"
MAX_TOKENS="${MAX_TOKENS:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
NUM_RUNS="${NUM_RUNS:-1}"
NUM_WARMUP_RUNS="${NUM_WARMUP_RUNS:-1}"
PROFILER="${PROFILER:-none}"
CUDAGRAPH_MODE="${CUDAGRAPH_MODE:-default}"
SUMMARY_METRIC="${SUMMARY_METRIC:-e2e_throughput_tok_s}"
EXTRA_ARGS=()

usage() {
  cat <<EOF
Usage: $(basename "$0") [options] [-- extra dflash_profiling.py args]

Runs Math-500 full-set vLLM DFlash sweeps over batch sizes and tree budgets.

Options:
  --model PATH                 Target model path (default: ${TARGET_MODEL})
  --draft-model PATH           DFlash draft model path (default: ${DRAFT_MODEL})
  --profiler-dir DIR           Output root (default: /root/data/vllm-ptd/...)
  --batch-sizes "1 2 ..."      Batch sizes to sweep (default: "${BATCH_SIZES}")
  --tree-budgets "16 32 ..."   Tree budgets to sweep (default: "${TREE_BUDGETS}")
  --tp-size N                  Tensor parallel size (default: ${TP_SIZE})
  --tree-kv-layout LAYOUT      physical or logical (default: ${TREE_KV_LAYOUT})
  --tree-attn-kernel KERNEL    optimus or triton (default: ${TREE_ATTN_KERNEL})
  --attention-backend NAME     vLLM attention backend (default: ${ATTENTION_BACKEND})
  --max-tokens N               Generation max tokens (default: ${MAX_TOKENS})
  --max-samples N              0 means full Math-500 set (default: ${MAX_SAMPLES})
  --num-runs N                 Timed runs per setting (default: ${NUM_RUNS})
  --num-warmup-runs N          Warmup batches per setting (default: ${NUM_WARMUP_RUNS})
  --profiler NAME              none, torch, or cuda (default: ${PROFILER})
  --cudagraph-mode MODE        vLLM CUDA graph mode (default: ${CUDAGRAPH_MODE})
  --summary-metric KEY         e2e_throughput_tok_s or benchmark_tok_s
  --skip-ar                    Do not rerun AR baselines
  --skip-dflash                Do not rerun DFlash budget cells
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)                 TARGET_MODEL="$2"; shift 2 ;;
    --draft-model)           DRAFT_MODEL="$2"; shift 2 ;;
    --profiler-dir)          PROFILER_DIR="$2"; shift 2 ;;
    --batch-sizes)           BATCH_SIZES="$2"; shift 2 ;;
    --tree-budgets)          TREE_BUDGETS="$2"; shift 2 ;;
    --tp-size)               TP_SIZE="$2"; shift 2 ;;
    --tree-kv-layout)        TREE_KV_LAYOUT="$2"; shift 2 ;;
    --tree-attn-kernel)      TREE_ATTN_KERNEL="$2"; shift 2 ;;
    --attention-backend)     ATTENTION_BACKEND="$2"; shift 2 ;;
    --max-tokens)            MAX_TOKENS="$2"; shift 2 ;;
    --max-samples)           MAX_SAMPLES="$2"; shift 2 ;;
    --num-runs)              NUM_RUNS="$2"; shift 2 ;;
    --num-warmup-runs)       NUM_WARMUP_RUNS="$2"; shift 2 ;;
    --profiler)              PROFILER="$2"; shift 2 ;;
    --cudagraph-mode)        CUDAGRAPH_MODE="$2"; shift 2 ;;
    --summary-metric)        SUMMARY_METRIC="$2"; shift 2 ;;
    --skip-ar)               RUN_AR=0; shift ;;
    --skip-dflash)           RUN_DFLASH=0; shift ;;
    -h|--help)               usage; exit 0 ;;
    --)                      shift; EXTRA_ARGS+=("$@"); break ;;
    *)                       EXTRA_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${TREE_KV_LAYOUT}" != "physical" && "${TREE_KV_LAYOUT}" != "logical" ]]; then
  echo "ERROR: --tree-kv-layout must be physical or logical"
  exit 1
fi

if [[ ! -d "$TARGET_MODEL" ]]; then
  echo "ERROR: TARGET_MODEL path does not exist: $TARGET_MODEL"
  exit 1
fi

if [[ ! -d "$DRAFT_MODEL" ]]; then
  echo "ERROR: DRAFT_MODEL path does not exist: $DRAFT_MODEL"
  exit 1
fi

DRAFT_TAG="$(basename "${DRAFT_MODEL}")"
DATE_TAG="$(date +%m%d)"
if [[ -z "${PROFILER_DIR}" ]]; then
  PROFILER_DIR="/root/data/vllm-ptd/vllm_qwen3_8b_profile_${DRAFT_TAG}_${DATE_TAG}_math500_causal_${TREE_DRAFT_MODE}_${TREE_CONSTRUCTION}_tree_d${TREE_DEPTH}_w${TREE_WIDTH}_budget_sweep_bsz_sweep_${TREE_KV_LAYOUT}_tree_impl_${TREE_ATTN_KERNEL}"
fi

mkdir -p "$PROFILER_DIR"
RUN_LOG="${PROFILER_DIR}/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$RUN_LOG") 2>&1

if [[ "${TREE_ATTN_KERNEL}" == "optimus" && -n "${OPTIMUS_SRC}" && -d "${OPTIMUS_SRC}" ]]; then
  export PYTHONPATH="${OPTIMUS_SRC}${PYTHONPATH:+:$PYTHONPATH}"
fi

read -r -a BATCH_SIZE_LIST <<< "${BATCH_SIZES//,/ }"
read -r -a TREE_BUDGET_LIST <<< "${TREE_BUDGETS//,/ }"

echo "Run log:            $RUN_LOG"
echo "Repo root:          $REPO_ROOT"
echo "HF datasets cache:  $HF_DATASETS_CACHE"
echo "Target model:       $TARGET_MODEL"
echo "Draft model:        $DRAFT_MODEL"
echo "Profiler dir:       $PROFILER_DIR"
echo "Prompt set:         math-500"
echo "Batch sizes:        ${BATCH_SIZES}"
echo "Tree budgets:       ${TREE_BUDGETS}"
echo "Tree KV layout:     ${TREE_KV_LAYOUT}"
echo "Tree attn kernel:   ${TREE_ATTN_KERNEL}"
echo "Max samples:        ${MAX_SAMPLES} (0 means full Math-500)"
echo "Max tokens:         ${MAX_TOKENS}"
echo "Profiler:           ${PROFILER}"
echo "CUDAGraph mode:     ${CUDAGRAPH_MODE}"
echo "CUDA home:          ${CUDA_HOME:-unset}"
echo "Python:             $(command -v python)"

python - <<'PY'
import re
import shutil
import subprocess
import torch

print(f"Torch:              {torch.__version__} (CUDA build {torch.version.cuda})")
nvcc = shutil.which("nvcc")
print(f"NVCC:               {nvcc or 'not found'}")
if nvcc:
    nvcc_output = subprocess.check_output([nvcc, "--version"], text=True)
    match = re.search(r"release\s+([0-9]+(?:\.[0-9]+)?)", nvcc_output)
    print(f"NVCC CUDA release:  {match.group(1) if match else 'unknown'}")
print(f"CUDA available:     {torch.cuda.is_available()}")
print(f"CUDA device count:  {torch.cuda.device_count()}")
if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
    raise SystemExit("ERROR: PyTorch cannot initialize CUDA in this script environment")
print(f"CUDA device 0:      {torch.cuda.get_device_name(0)}")
PY

cd "$REPO_ROOT"

run_profile() {
  local label="$1"
  local mode="$2"
  local batch_size="$3"
  local max_tree_budget="${4:-}"
  local run_dir="${PROFILER_DIR}/${label}"

  local tree_args=()
  if [[ "${mode}" == "dflash" ]]; then
    tree_args=(
      --max-tree-budget "${max_tree_budget}"
      --tree-kv-layout "${TREE_KV_LAYOUT}"
    )
  fi

  echo ""
  echo "Running label=${label} mode=${mode} batch_size=${batch_size} max_tree_budget=${max_tree_budget:-n/a}"
  python examples/offline_inference/dflash_profiling.py \
    --prompt-set math-500 \
    --mode "${mode}" \
    --head-type causal \
    --model "${TARGET_MODEL}" \
    --draft-model "${DRAFT_MODEL}" \
    --max-tokens "${MAX_TOKENS}" \
    --block-size "${TREE_DEPTH}" \
    --tree-width "${TREE_WIDTH}" \
    "${tree_args[@]}" \
    --tree-draft "${TREE_DRAFT_MODE}" \
    --max-draft-passes "${ADDITIONAL_DRAFT_REFINEMENT_PASSES}" \
    --tree-prune-ratio "${TREE_PRUNE_RATIO}" \
    --tree-construction "${TREE_CONSTRUCTION}" \
    --tree-attn-kernel "${TREE_ATTN_KERNEL}" \
    --num-cudagraph-tree-captures "${NUM_CUDAGRAPH_TREE_CAPTURES}" \
    --cudagraph-mode "${CUDAGRAPH_MODE}" \
    --attention-backend "${ATTENTION_BACKEND}" \
    --tp-sizes "${TP_SIZE}" \
    --batch-sizes "${batch_size}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}" \
    --max-samples "${MAX_SAMPLES}" \
    --max-num-seqs "${batch_size}" \
    --num-runs "${NUM_RUNS}" \
    --num-warmup-runs "${NUM_WARMUP_RUNS}" \
    --profiler "${PROFILER}" \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
    --torch-profiler-dir "${run_dir}"
}

if [[ "${RUN_AR}" == "1" ]]; then
  for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    run_profile "ar_bsz${batch_size}" "ar" "${batch_size}"
  done
fi

if [[ "${RUN_DFLASH}" == "1" ]]; then
  for batch_size in "${BATCH_SIZE_LIST[@]}"; do
    for budget in "${TREE_BUDGET_LIST[@]}"; do
      run_profile "budget${budget}_bsz${batch_size}_${TREE_KV_LAYOUT}" "dflash" "${batch_size}" "${budget}"
    done
  done
fi

python examples/offline_inference/dflash_math500_sweep_summary.py \
  --profiler-dir "${PROFILER_DIR}" \
  --batch-sizes "${BATCH_SIZES}" \
  --tree-budgets "${TREE_BUDGETS}" \
  --tp-size "${TP_SIZE}" \
  --tree-kv-layout "${TREE_KV_LAYOUT}" \
  --metric "${SUMMARY_METRIC}"

echo ""
echo "Sweep outputs:"
echo "  CSV:   ${PROFILER_DIR}/run_summary.csv"
echo "  LaTeX: ${PROFILER_DIR}/run_summary_latex.tex"
