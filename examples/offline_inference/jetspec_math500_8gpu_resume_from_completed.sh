#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)

PROFILER_DIR="${PROFILER_DIR:-/root/data/vllm-ptd/vllm_qwen3_8b_profile_epoch_6_step_291744_forward_kl_0616_math500_jetspec_accum_logp_breadth_first_tree_d16_w7_budget_sweep_bsz_sweep_logical_tree_impl_optimus}"
RESUME_COMPLETED="${RESUME_COMPLETED:-1}"

EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --resume-completed) RESUME_COMPLETED=1; shift ;;
    --no-resume-completed) RESUME_COMPLETED=0; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

RESUME_COMPLETED="${RESUME_COMPLETED}" \
TREE_KV_LAYOUT=logical \
TREE_DRAFT_MODE=accum_logp \
"${SCRIPT_DIR}/jetspec_profiling_math500_tree_budget_bsz_sweep_dgx_pod.sh" \
  --profiler-dir "${PROFILER_DIR}" \
  --cuda-devices "0 1 2 3 4 5 6 7" \
  --parallel-workers 8 \
  "${EXTRA_ARGS[@]}"
