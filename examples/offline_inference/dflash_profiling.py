#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from datasets import load_dataset
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.utils.argparse_utils import FlexibleArgumentParser

DEFAULT_TARGET_MODEL = "/data/models/Qwen3-8B"
DEFAULT_DRAFT_MODEL = "/data/models/Qwen3-8B-DFlash-b16"

DEFAULT_PROMPTS = [
    "Write a short explanation of speculative decoding.",
    "Summarize how KV cache is used in transformer inference.",
    "Give three tips for profiling CUDA workloads.",
    "Explain why acceptance rate matters for DFlash.",
]

CODING_PROMPTS = [
    (
        "Implement a Python function `binary_search(arr, target)` for a sorted "
        "list of integers that returns the index of target or -1."
    ),
    (
        "Implement a Python class `LRUCache` with methods `get(key)` and "
        "`put(key, value)` using O(1) average-time operations."
    ),
    (
        "Implement Python function `merge_intervals(intervals)` that merges "
        "overlapping intervals and returns a sorted merged list."
    ),
    (
        "Implement Python function `dijkstra(n, edges, src)` that returns "
        "shortest distances from src in a weighted graph with nonnegative weights."
    ),
]


def load_dataset_prompt_bank(prompt_set: str) -> list[str]:
    if prompt_set == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        prompt_fmt = (
            "{question}\n"
            "Please reason step by step, and put your final answer within \\boxed{{}}."
        )
        return [prompt_fmt.format(**row) for row in dataset]
    if prompt_set == "humaneval":
        dataset = load_dataset("openai/openai_humaneval", split="test")
        prompt_fmt = (
            "Write a solution to the following problem and make sure that it "
            "passes the tests:\n```python\n{prompt}\n```"
        )
        return [prompt_fmt.format(**row) for row in dataset]
    raise ValueError(f"Unknown dataset-backed prompt set: {prompt_set}")


def get_prompt_bank(prompt_set: str) -> list[str]:
    if prompt_set == "mix":
        return DEFAULT_PROMPTS
    if prompt_set == "coding":
        return CODING_PROMPTS
    if prompt_set in {"gsm8k", "humaneval"}:
        return load_dataset_prompt_bank(prompt_set)
    raise ValueError(f"Unknown prompt set: {prompt_set}")


def build_prompt_batches(prompts: list[str], batch_size: int) -> list[list[str]]:
    if batch_size < 1:
        raise ValueError(f"Invalid batch size: {batch_size}. Expected batch size >= 1")
    return [prompts[start : start + batch_size] for start in range(0, len(prompts), batch_size)]


def apply_chat_template(tokenizer, prompts: list[str]) -> list[str]:
    templated_prompts = []
    for prompt in prompts:
        messages = [{"role": "user", "content": prompt}]
        try:
            templated_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            templated_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        templated_prompts.append(templated_prompt)
    return templated_prompts


def get_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["ar", "dflash"]
    return [mode]


def collect_spec_decode_counters(metrics) -> dict[str, object]:
    counters: dict[str, object] = {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
        "num_tree_drafts": 0.0,
        "num_tree_nodes": 0.0,
        "accepted_per_pos": [],
        "tree_nodes_per_depth": [],
    }
    for metric in metrics:
        name = getattr(metric, "name", "")
        if name == "vllm:spec_decode_num_accepted_tokens_per_pos":
            values = getattr(metric, "values", None)
            if values is not None:
                counters["accepted_per_pos"] = [float(v) for v in values]
            continue
        if name == "vllm:spec_decode_num_tree_nodes_per_depth":
            values = getattr(metric, "values", None)
            if values is not None:
                counters["tree_nodes_per_depth"] = [float(v) for v in values]
            continue
        value = getattr(metric, "value", None)
        if value is None:
            continue
        if name == "vllm:spec_decode_num_drafts":
            counters["num_drafts"] = float(counters["num_drafts"]) + float(value)
        elif name == "vllm:spec_decode_num_draft_tokens":
            counters["num_draft_tokens"] = float(counters["num_draft_tokens"]) + float(value)
        elif name == "vllm:spec_decode_num_accepted_tokens":
            counters["num_accepted_tokens"] = float(counters["num_accepted_tokens"]) + float(value)
        elif name == "vllm:spec_decode_num_tree_drafts":
            counters["num_tree_drafts"] = float(counters["num_tree_drafts"]) + float(value)
        elif name == "vllm:spec_decode_num_tree_nodes":
            counters["num_tree_nodes"] = float(counters["num_tree_nodes"]) + float(value)
    return counters


def diff_counters(after: dict[str, object], before: dict[str, object]) -> dict[str, object]:
    result: dict[str, object] = {}
    for k in after:
        a, b = after.get(k, 0.0), before.get(k, 0.0)
        if isinstance(a, list) and isinstance(b, list):
            maxlen = max(len(a), len(b))
            result[k] = [
                (a[i] if i < len(a) else 0.0) - (b[i] if i < len(b) else 0.0)
                for i in range(maxlen)
            ]
        else:
            result[k] = float(a) - float(b)  # type: ignore[arg-type]
    return result


def _duration_to_seconds(value: float, unit: str) -> float:
    if unit == "s":
        return value
    if unit == "ms":
        return value / 1000.0
    if unit == "us":
        return value / 1_000_000.0
    raise ValueError(f"Unsupported duration unit: {unit}")


def collect_execute_context_cuda_seconds(run_output_dir: Path) -> dict[str, float]:
    """Aggregate execute_context self-CUDA time by phase from profiler text files."""
    totals = {"prefill_cuda_s": 0.0, "decode_cuda_s": 0.0, "mixed_cuda_s": 0.0}
    files = sorted(run_output_dir.glob("profiler_out_*.txt"))
    if not files:
        return totals

    name_pattern = re.compile(
        r"execute_context_(\d+)\((\d+)\)_generation_(\d+)\((\d+)\)"
    )
    # Last columns are: self_cuda, self_cuda%, cuda_total, cuda_avg, calls.
    # Capture self_cuda right before the percentage column.
    self_cuda_pattern = re.compile(
        r"\s([0-9.]+)(us|ms|s)\s+[0-9.]+%\s+[0-9.]+(?:us|ms|s)\s+[0-9.]+(?:us|ms|s)\s+\d+\s*$"
    )

    for path in files:
        for line in path.read_text(encoding="utf-8").splitlines():
            name_match = name_pattern.search(line)
            if not name_match:
                continue
            cuda_match = self_cuda_pattern.search(line)
            if not cuda_match:
                continue

            ctx_reqs, ctx_tokens, gen_reqs, gen_tokens = map(int, name_match.groups())
            self_cuda_s = _duration_to_seconds(
                float(cuda_match.group(1)),
                cuda_match.group(2),
            )

            if gen_reqs == 0 and gen_tokens == 0 and (ctx_reqs > 0 or ctx_tokens > 0):
                totals["prefill_cuda_s"] += self_cuda_s
            elif ctx_reqs == 0 and ctx_tokens == 0 and (gen_reqs > 0 or gen_tokens > 0):
                totals["decode_cuda_s"] += self_cuda_s
            else:
                totals["mixed_cuda_s"] += self_cuda_s

    return totals


def resolve_effective_block_size(args) -> int | None:
    if args.block_size is None:
        return None
    if args.block_size <= 1:
        raise ValueError("--block-size must be greater than 1.")
    return args.block_size


def resolve_effective_num_speculative_tokens(args) -> int:
    block_size = resolve_effective_block_size(args)
    if block_size is not None:
        return block_size - 1
    return args.num_speculative_tokens


def mean_or_zero(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def validate_native_only_settings(args) -> None:
    invalid_tp = [tp for tp in args.tp_sizes if tp != 1]
    invalid_bs = [bs for bs in args.batch_sizes if bs != 1]
    if invalid_tp or invalid_bs:
        raise ValueError(
            "This native-only parity benchmark currently supports only "
            "--tp-sizes 1 and --batch-sizes 1."
        )


def run_native_profile(
    *,
    args,
    mode: str,
    prompt_batches: list[list[str]],
    sampling_params: SamplingParams,
    tp_size: int,
    batch_size: int,
    effective_num_speculative_tokens: int,
    effective_max_num_seqs: int,
    profiler_config: dict[str, str],
    run_output_dir: Path,
    post_generation_hook: Callable[[Any], None] | None = None,
    pre_generation_hook: Callable[[Any], None] | None = None,
) -> dict[str, object]:
    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=effective_max_num_seqs,
        max_model_len=args.max_model_len,
        enforce_eager=args.enforce_eager,
        profiler_config=profiler_config,
        disable_log_stats=False,
    )
    if args.attention_backend is not None:
        llm_kwargs["attention_backend"] = args.attention_backend
    if mode == "dflash":
        llm_kwargs["speculative_config"] = {
            "method": "dflash",
            "model": args.draft_model,
            "num_speculative_tokens": effective_num_speculative_tokens,
            "max_model_len": args.max_model_len,
            "head_type": args.head_type,
            "tree_width": args.tree_width,
            "max_tree_budget": args.max_tree_budget,
            "tree_draft": args.tree_draft,
            "tree_hybrid_alpha": args.tree_hybrid_alpha,
            "max_draft_passes": args.max_draft_passes,
            "tree_prune_ratio": args.tree_prune_ratio,
            "tree_construction": args.tree_construction,
            "tree_attn_kernel": args.tree_attn_kernel,
            "num_cudagraph_tree_captures": args.num_cudagraph_tree_captures,
        }
        if args.tree_attn_kernel == "optimus":
            llm_kwargs["block_size"] = 128

    llm = LLM(**llm_kwargs)
    if pre_generation_hook is not None:
        try:
            pre_generation_hook(llm)
        except Exception as hook_exc:
            import traceback

            print(f"WARNING: pre_generation_hook failed: {hook_exc}")
            traceback.print_exc()
    warmup_batches = prompt_batches[: min(args.num_warmup_runs, len(prompt_batches))]
    for batch_prompts in warmup_batches:
        llm.generate(batch_prompts, sampling_params=sampling_params)

    metrics_before = collect_spec_decode_counters(llm.get_metrics())
    llm.start_profile()
    t0 = time.perf_counter()
    total_output_tokens = 0
    total_prompt_tokens = 0
    time_per_output_token_samples: list[float] = []
    sampled_outputs: list[tuple[str, str]] = []
    num_batches = 0
    num_samples = 0
    for run_idx in range(args.num_runs):
        for batch_idx, batch_prompts in enumerate(prompt_batches):
            batch_t0 = time.perf_counter()
            outputs = llm.generate(batch_prompts, sampling_params=sampling_params)
            batch_elapsed = time.perf_counter() - batch_t0
            batch_output_tokens = sum(
                len(output.outputs[0].token_ids) for output in outputs
            )
            batch_prompt_tokens = sum(
                len(output.prompt_token_ids) for output in outputs
                if output.prompt_token_ids is not None
            )
            total_output_tokens += batch_output_tokens
            total_prompt_tokens += batch_prompt_tokens
            num_batches += 1
            num_samples += len(outputs)
            if batch_output_tokens > 0:
                time_per_output_token_samples.append(batch_elapsed / batch_output_tokens)
            if run_idx == args.num_runs - 1 and batch_idx < 3:
                sampled_outputs.extend(
                    (output.prompt, output.outputs[0].text) for output in outputs
                )
    elapsed = time.perf_counter() - t0
    llm.stop_profile()
    metrics_after = collect_spec_decode_counters(llm.get_metrics())
    metrics_delta = diff_counters(metrics_after, metrics_before)

    def _extract_topk_log(worker):
        worker_obj = getattr(worker, "worker", worker)
        model_runner = getattr(worker_obj, "model_runner", None)
        drafter = getattr(model_runner, "drafter", None)
        if drafter is not None and hasattr(drafter, "get_topk_log"):
            return drafter.get_topk_log()
        return []

    try:
        topk_logs = llm.collective_rpc(_extract_topk_log)
        topk_entries = topk_logs[0] if topk_logs else []
    except Exception as e:
        import traceback
        print(f"\n{'='*60}")
        print(f"ERROR extracting topk_log via collective_rpc: {e}")
        traceback.print_exc()
        print(f"{'='*60}\n")
        topk_entries = []
    if not topk_entries:
        try:
            worker = llm.llm_engine.model_executor.driver_worker.worker
            drafter = getattr(worker.model_runner, "drafter", None)
            if drafter is not None and hasattr(drafter, "get_topk_log"):
                topk_entries = drafter.get_topk_log()
        except Exception as e:
            print(f"WARNING: direct topk_log extraction failed: {e}")
    if topk_entries:
        topk_path = run_output_dir / "topk_log.json"
        topk_path.write_text(json.dumps(topk_entries, indent=2))
        print(f"Wrote {len(topk_entries)} topk log entries to {topk_path}")
    else:
        print(f"WARNING: topk_log is empty — no entries collected from drafter")

    if post_generation_hook is not None:
        try:
            post_generation_hook(llm)
        except Exception as hook_exc:
            import traceback

            print(f"WARNING: post_generation_hook failed: {hook_exc}")
            traceback.print_exc()

    del llm
    time.sleep(args.sleep_after_stop)
    phase_cuda = collect_execute_context_cuda_seconds(run_output_dir)

    num_drafts = float(metrics_delta["num_drafts"])
    num_draft_tokens = float(metrics_delta["num_draft_tokens"])
    num_accepted_tokens = float(metrics_delta["num_accepted_tokens"])
    num_tree_drafts = float(metrics_delta["num_tree_drafts"])
    num_tree_nodes = float(metrics_delta["num_tree_nodes"])
    accepted_per_pos = metrics_delta.get("accepted_per_pos", [])
    if not isinstance(accepted_per_pos, list):
        accepted_per_pos = []
    tree_nodes_per_depth_raw = metrics_delta.get("tree_nodes_per_depth", [])
    if not isinstance(tree_nodes_per_depth_raw, list):
        tree_nodes_per_depth_raw = []

    draft_tokens = (
        num_tree_nodes - num_tree_drafts
        if num_tree_drafts > 0
        else num_draft_tokens
    )

    per_pos_acceptance_rate: list[float] = []
    if num_drafts > 0 and accepted_per_pos:
        per_pos_acceptance_rate = [v / num_drafts for v in accepted_per_pos]

    acceptance_length_histogram: list[float] = []
    if num_drafts > 0 and per_pos_acceptance_rate:
        n = len(per_pos_acceptance_rate)
        cumulative = list(per_pos_acceptance_rate)
        hist = [0.0] * (n + 1)
        for d in range(n):
            rate_at_d = cumulative[d]
            rate_at_next = cumulative[d + 1] if d + 1 < n else 0.0
            hist[d + 1] = rate_at_d - rate_at_next
        hist[0] = 1.0 - cumulative[0] if cumulative else 1.0
        acceptance_length_histogram = hist

    prefill_cuda_s = phase_cuda["prefill_cuda_s"]
    decode_cuda_s = phase_cuda["decode_cuda_s"]
    prefill_throughput = (
        total_prompt_tokens / prefill_cuda_s if prefill_cuda_s > 0 else 0.0
    )
    decode_throughput = (
        total_output_tokens / decode_cuda_s if decode_cuda_s > 0 else 0.0
    )
    gpu_active_s = prefill_cuda_s + decode_cuda_s + phase_cuda["mixed_cuda_s"]
    gpu_utilization = gpu_active_s / elapsed if elapsed > 0 else 0.0

    return {
        "elapsed": elapsed,
        "total_output_tokens": total_output_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "throughput": total_output_tokens / elapsed if elapsed > 0 else 0.0,
        "prefill_throughput": prefill_throughput,
        "decode_throughput": decode_throughput,
        "gpu_utilization": gpu_utilization,
        "draft_tokens": draft_tokens,
        "accepted_tokens": num_accepted_tokens,
        "drafts": num_drafts,
        "acceptance_rate": (
            num_accepted_tokens / draft_tokens if draft_tokens > 0 else 0.0
        ),
        "acceptance_length": 1.0 + (
            num_accepted_tokens / num_drafts
            if num_drafts > 0
            else 0.0
        ),
        "per_pos_acceptance_rate": per_pos_acceptance_rate,
        "acceptance_length_histogram": acceptance_length_histogram,
        "avg_tree_nodes_per_depth": (
            [v / num_tree_drafts for v in tree_nodes_per_depth_raw]
            if num_tree_drafts > 0 and tree_nodes_per_depth_raw
            else []
        ),
        "phase_cuda": phase_cuda,
        "outputs": sampled_outputs,
        "engine_label": "vllm_native",
        "mean_time_per_output_token_s": mean_or_zero(time_per_output_token_samples),
        "num_batches": num_batches,
        "num_samples": num_samples,
    }


def parse_args():
    parser = FlexibleArgumentParser(
        description=(
            "Profile vLLM DFlash offline generation.\n\n"
            "Positional `model` can be a local directory or HF model id.\n"
            f"If omitted, defaults to {DEFAULT_TARGET_MODEL}."
        )
    )
    parser.add_argument(
        "--prompt-set",
        type=str,
        default="mix",
        choices=["mix", "coding", "gsm8k", "humaneval"],
        help=(
            "Prompt set to use. "
            "'mix' uses general profiling prompts; 'coding' uses 4 Python "
            "algorithm/data-structure tasks; 'gsm8k' and 'humaneval' match "
            "the dataset prompt formatting used in the dflash repo."
        ),
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=16,
        help=(
            "Maximum number of prompts or dataset entries to benchmark. "
            "Use a non-positive value to keep the full prompt set."
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_TARGET_MODEL,
        help=(
            "Target model path or HF model id. "
            "Can be a local directory; defaults to Qwen/Qwen3-8B."
        ),
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=DEFAULT_DRAFT_MODEL,
        help=(
            "DFlash draft/speculator model path or HF model id. "
            "Can be a local directory; defaults to z-lab/Qwen3-8B-DFlash-b16."
        ),
    )
    parser.add_argument(
        "--profiler",
        type=str,
        default="torch",
        choices=["torch", "cuda"],
        help="Profiler backend.",
    )
    parser.add_argument(
        "--torch-profiler-dir",
        type=str,
        default="./vllm_profile_dflash",
        help="Output directory for torch profiler traces.",
    )
    parser.add_argument(
        "--num-speculative-tokens",
        type=int,
        default=16,
        help="Number of speculative tokens for DFlash.",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=None,
        help=(
            "Reference-style DFlash block size. When set, vLLM uses "
            "num_speculative_tokens=block_size-1 so native runs match the "
            "reference benchmark's block-size semantics."
        ),
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=32768,
        help="Max model length.",
    )
    parser.add_argument(
        "--tp-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Tensor parallel sizes to profile. Native parity mode currently supports only 1.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1],
        help="Batch sizes to profile. Native parity mode currently supports only 1.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["dflash", "ar", "both"],
        help="Profile dflash only, ar only, or both for throughput gains.",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="GPU memory utilization target.",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=51200,
        help=(
            "Scheduler max_num_batched_tokens passed to vLLM. Defaults to 51200 "
            "for tree profiling so full-tree slot reservation does not collapse "
            "the scheduled-token budget."
        ),
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=None,
        help=(
            "Scheduler max_num_seqs passed to vLLM. Defaults to the largest "
            "profiled batch size when unset."
        ),
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs and use eager mode.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Generation max tokens.",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=2,
        help="Number of timed and profiled generate calls.",
    )
    parser.add_argument(
        "--num-warmup-runs",
        type=int,
        default=1,
        help="Number of warmup generate calls before profiling.",
    )
    parser.add_argument(
        "--sleep-after-stop",
        type=int,
        default=10,
        help="Seconds to wait after stop_profile to allow trace flush.",
    )
    parser.add_argument(
        "--trust-remote-code",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable trust_remote_code for model loading.",
    )
    parser.add_argument(
        "--attention-backend",
        type=str,
        default="FLASH_ATTN",
        help=(
            "Optional attention backend override, e.g. FLASH_ATTN, TRITON_ATTN, TREE_ATTN, "
            "or FLEX_ATTENTION."
        ),
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default="auto",
        choices=["auto", "bidirectional", "causal"],
        help=(
            "DFlash draft attention mode. 'auto' follows the draft checkpoint, "
            "'bidirectional' forces non-causal draft attention, and 'causal' "
            "forces causal draft attention."
        ),
    )
    parser.add_argument(
        "--tree-width",
        type=int,
        default=1,
        help=(
            "Requested DFlash draft tree width. Width 1 keeps the current linear "
            "parallel drafting path."
        ),
    )
    parser.add_argument(
        "--max-tree-budget",
        type=int,
        default=None,
        help="Optional cap on total tree nodes for DFlash tree experiments.",
    )
    parser.add_argument(
        "--tree-draft",
        type=str,
        default="accum_logp",
        choices=["accum_logp", "entropy", "hybrid", "opt_prefix"],
        help=(
            "Scoring strategy for tree node expansion. "
            "'accum_logp' prioritises high-probability prefixes. "
            "'entropy' prioritises uncertain positions. "
            "'hybrid' combines cumulative log-prob with entropy. "
            "'opt_prefix' builds the provably optimal tree under factorized "
            "draft marginals (ignores --tree-construction)."
        ),
    )
    parser.add_argument(
        "--tree-hybrid-alpha",
        type=float,
        default=1.0,
        help="Weight for per-depth entropy in 'hybrid' scoring mode.",
    )
    parser.add_argument(
        "--max-draft-passes",
        type=int,
        default=5,
        help="Number of prune/regrow refinement passes after initial tree build.",
    )
    parser.add_argument(
        "--tree-prune-ratio",
        type=float,
        default=0.25,
        help="Fraction of leaves to prune per refinement pass (0.0-1.0).",
    )
    parser.add_argument(
        "--tree-construction",
        type=str,
        default="depth_first",
        choices=["depth_first", "breadth_first"],
        help=(
            "Tree node allocation strategy. 'depth_first' pre-allocates "
            "the greedy spine to full depth before side branches. "
            "'breadth_first' uses best-cumulative-logprob heap in a breath-first manner (legacy)."
        ),
    )
    parser.add_argument(
        "--tree-attn-kernel",
        type=str,
        default="triton",
        choices=["triton", "optimus"],
        help=(
            "Attention kernel for DFlash tree verification. "
            "'triton' uses Triton bias-based path; "
            "'optimus' uses fused SM90 paged tree-mask kernel."
        ),
    )
    parser.add_argument(
        "--num-cudagraph-tree-captures",
        type=int,
        default=0,
        help=(
            "Number of distinct tree sizes to capture as CUDAGraphs. "
            "E.g. 16 produces evenly-spaced sizes up to max-tree-budget. "
            "After pruning the tree is adjusted to the nearest captured "
            "size for a guaranteed CUDAGraph hit. 0 disables."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if args.num_warmup_runs < 0:
        raise ValueError("--num-warmup-runs must be >= 0")
    if args.tree_width < 1:
        raise ValueError("--tree-width must be >= 1")
    validate_native_only_settings(args)

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    prompt_bank = apply_chat_template(tokenizer, get_prompt_bank(args.prompt_set))
    if args.max_samples > 0:
        prompt_bank = prompt_bank[: args.max_samples]
    modes = get_modes(args.mode)
    effective_num_speculative_tokens = resolve_effective_num_speculative_tokens(args)
    effective_block_size = (
        resolve_effective_block_size(args) or (effective_num_speculative_tokens + 1)
    )
    effective_max_num_seqs = args.max_num_seqs or max(args.batch_sizes)
    # key: (tp_size, batch_size, mode) -> throughput tokens/s
    throughputs: dict[tuple[int, int, str], float] = {}
    benchmark_tok_s: dict[tuple[int, int, str], float] = {}
    execute_context_cuda: dict[tuple[int, int, str], dict[str, float]] = {}
    summary_lines = []

    for tp_size in args.tp_sizes:
        if tp_size < 1:
            raise ValueError(f"Invalid tp size: {tp_size}. Expected tp >= 1")
        for batch_size in args.batch_sizes:
            if batch_size < 1:
                raise ValueError(
                    f"Invalid batch size: {batch_size}. Expected batch size >= 1"
                )
            prompt_batches = build_prompt_batches(prompt_bank, batch_size)

            for mode in modes:
                print("=" * 80)
                print(
                    f"Profiling mode={mode}, tensor_parallel_size={tp_size}, "
                    f"batch_size={batch_size}"
                )

                if args.profiler == "torch":
                    run_output_dir = Path(
                        f"{args.torch_profiler_dir}/{mode}/tp{tp_size}/bs{batch_size}"
                    )
                    profiler_config = {
                        "profiler": "torch",
                        "torch_profiler_dir": str(run_output_dir),
                    }
                else:
                    run_output_dir = Path(
                        f"{args.torch_profiler_dir}/{mode}/tp{tp_size}/bs{batch_size}"
                    )
                    profiler_config = {"profiler": "cuda"}
                run_output_dir.mkdir(parents=True, exist_ok=True)
                result = run_native_profile(
                    args=args,
                    mode=mode,
                    prompt_batches=prompt_batches,
                    sampling_params=sampling_params,
                    tp_size=tp_size,
                    batch_size=batch_size,
                    effective_num_speculative_tokens=effective_num_speculative_tokens,
                    effective_max_num_seqs=effective_max_num_seqs,
                    profiler_config=profiler_config,
                    run_output_dir=run_output_dir,
                )

                elapsed = float(result["elapsed"])
                total_output_tokens = int(result["total_output_tokens"])
                total_prompt_tokens = int(result["total_prompt_tokens"])
                throughput = float(result["throughput"])
                prefill_throughput = float(result["prefill_throughput"])
                decode_throughput = float(result["decode_throughput"])
                gpu_utilization = float(result["gpu_utilization"])
                throughputs[(tp_size, batch_size, mode)] = throughput
                draft_tokens = float(result["draft_tokens"])
                accepted_tokens = float(result["accepted_tokens"])
                drafts = float(result["drafts"])
                acceptance_rate = float(result["acceptance_rate"])
                acceptance_length = float(result["acceptance_length"])
                phase_cuda = result["phase_cuda"]
                mean_time_per_output_token_s = float(result["mean_time_per_output_token_s"])
                benchmark_speed_tok_s = (
                    1.0 / mean_time_per_output_token_s
                    if mean_time_per_output_token_s > 0
                    else 0.0
                )
                benchmark_tok_s[(tp_size, batch_size, mode)] = benchmark_speed_tok_s
                print(
                    f"[RESULT] mode={mode} tp={tp_size} bs={batch_size} "
                    f"prompt_tokens={total_prompt_tokens} "
                    f"output_tokens={total_output_tokens} elapsed_s={elapsed:.3f} "
                    f"throughput_tok_s={throughput:.2f}"
                )
                print(
                    f"[THROUGHPUT] mode={mode} tp={tp_size} bs={batch_size} "
                    f"prefill_tok_s={prefill_throughput:.2f} "
                    f"decode_tok_s={decode_throughput:.2f} "
                    f"e2e_tok_s={throughput:.2f} "
                    f"gpu_utilization={gpu_utilization:.2%}"
                )
                print(
                    f"[BENCHMARK] mode={mode} tp={tp_size} bs={batch_size} "
                    f"num_samples={result['num_samples']} num_batches={result['num_batches']} "
                    f"mean_time_per_output_token_s={mean_time_per_output_token_s:.6f} "
                    f"benchmark_tok_s={benchmark_speed_tok_s:.2f}"
                )
                print(
                    f"[SPEC_METRICS] mode={mode} tp={tp_size} bs={batch_size} "
                    f"num_drafts={drafts:.0f} num_draft_tokens={draft_tokens:.0f} "
                    f"num_accepted_tokens={accepted_tokens:.0f} "
                    f"acceptance_rate={acceptance_rate:.4f} "
                    f"acceptance_length={acceptance_length:.4f}"
                )
                per_pos_rate = result.get("per_pos_acceptance_rate", [])
                acc_len_hist = result.get("acceptance_length_histogram", [])
                if per_pos_rate:
                    pos_str = " ".join(
                        f"d{i}={r:.4f}" for i, r in enumerate(per_pos_rate)
                    )
                    print(
                        f"[PER_DEPTH_ACCEPTANCE] mode={mode} tp={tp_size} "
                        f"bs={batch_size} {pos_str}"
                    )
                if acc_len_hist:
                    hist_str = " ".join(
                        f"len{i}={p:.4f}" for i, p in enumerate(acc_len_hist)
                    )
                    print(
                        f"[ACCEPTANCE_LEN_HIST] mode={mode} tp={tp_size} "
                        f"bs={batch_size} {hist_str}"
                    )
                avg_tree_shape = result.get("avg_tree_nodes_per_depth", [])
                if avg_tree_shape:
                    shape_str = " ".join(
                        f"d{i}={v:.1f}" for i, v in enumerate(avg_tree_shape)
                    )
                    print(
                        f"[TREE_SHAPE] mode={mode} tp={tp_size} "
                        f"bs={batch_size} {shape_str}"
                    )
                for prompt, generated_text in result["outputs"]:
                    print("-" * 80)
                    print(f"Prompt: {prompt}")
                    print(f"Generated: {generated_text}")

                execute_context_cuda[(tp_size, batch_size, mode)] = phase_cuda

                report_lines = [
                    f"mode={mode}",
                    f"engine={result['engine_label']}",
                    f"prompt_set={args.prompt_set}",
                    "prompt_format=chat_template",
                    f"attention_backend={args.attention_backend}",
                    f"head_type={args.head_type}",
                    f"block_size={effective_block_size}",
                    f"tree_width={args.tree_width}",
                    f"max_tree_budget={args.max_tree_budget}",
                    f"tree_draft={args.tree_draft}",
                    f"max_draft_passes={args.max_draft_passes}",
                    f"max_num_batched_tokens={args.max_num_batched_tokens}",
                    f"max_num_seqs={effective_max_num_seqs}",
                    f"num_samples={result['num_samples']}",
                    f"num_batches={result['num_batches']}",
                    f"tp_size={tp_size}",
                    f"batch_size={batch_size}",
                    f"prompt_tokens={total_prompt_tokens}",
                    f"output_tokens={total_output_tokens}",
                    f"elapsed_s={elapsed:.6f}",
                    f"e2e_throughput_tok_s={throughput:.6f}",
                    f"prefill_throughput_tok_s={prefill_throughput:.6f}",
                    f"decode_throughput_tok_s={decode_throughput:.6f}",
                    f"gpu_utilization={gpu_utilization:.6f}",
                    f"mean_time_per_output_token_s={mean_time_per_output_token_s:.6f}",
                    f"benchmark_tok_s={benchmark_speed_tok_s:.6f}",
                    f"num_drafts={drafts:.0f}",
                    f"num_draft_tokens={draft_tokens:.0f}",
                    f"num_accepted_tokens={accepted_tokens:.0f}",
                    f"acceptance_rate={acceptance_rate:.6f}",
                    f"acceptance_length={acceptance_length:.6f}",
                    f"prefill_execute_context_cuda_s={phase_cuda['prefill_cuda_s']:.6f}",
                    f"decode_execute_context_cuda_s={phase_cuda['decode_cuda_s']:.6f}",
                    f"mixed_execute_context_cuda_s={phase_cuda['mixed_cuda_s']:.6f}",
                ]
                if per_pos_rate:
                    report_lines.append(
                        "per_depth_acceptance_rate="
                        + ",".join(f"{r:.6f}" for r in per_pos_rate)
                    )
                if acc_len_hist:
                    report_lines.append(
                        "acceptance_length_histogram="
                        + ",".join(f"{p:.6f}" for p in acc_len_hist)
                    )
                if avg_tree_shape:
                    report_lines.append(
                        "avg_tree_nodes_per_depth="
                        + ",".join(f"{v:.2f}" for v in avg_tree_shape)
                    )
                (run_output_dir / "metrics_report.txt").write_text(
                    "\n".join(report_lines) + "\n", encoding="utf-8"
                )
                summary_lines.append(
                    " ".join(
                        [
                            f"mode={mode}",
                            f"engine={result['engine_label']}",
                            f"prompt_set={args.prompt_set}",
                            "prompt_format=chat_template",
                            f"attention_backend={args.attention_backend}",
                            f"head_type={args.head_type}",
                            f"block_size={effective_block_size}",
                            f"tree_width={args.tree_width}",
                            f"max_tree_budget={args.max_tree_budget}",
                            f"tree_draft={args.tree_draft}",
                            f"max_draft_passes={args.max_draft_passes}",
                            f"max_num_batched_tokens={args.max_num_batched_tokens}",
                            f"max_num_seqs={effective_max_num_seqs}",
                            f"num_samples={result['num_samples']}",
                            f"num_batches={result['num_batches']}",
                            f"tp={tp_size}",
                            f"bs={batch_size}",
                            f"prompt_tokens={total_prompt_tokens}",
                            f"output_tokens={total_output_tokens}",
                            f"e2e_throughput_tok_s={throughput:.6f}",
                            f"prefill_throughput_tok_s={prefill_throughput:.6f}",
                            f"decode_throughput_tok_s={decode_throughput:.6f}",
                            f"gpu_utilization={gpu_utilization:.6f}",
                            f"mean_time_per_output_token_s={mean_time_per_output_token_s:.6f}",
                            f"benchmark_tok_s={benchmark_speed_tok_s:.6f}",
                            f"num_drafts={drafts:.0f}",
                            f"num_draft_tokens={draft_tokens:.0f}",
                            f"num_accepted_tokens={accepted_tokens:.0f}",
                            f"acceptance_rate={acceptance_rate:.6f}",
                            f"acceptance_length={acceptance_length:.6f}",
                            f"prefill_execute_context_cuda_s={phase_cuda['prefill_cuda_s']:.6f}",
                            f"decode_execute_context_cuda_s={phase_cuda['decode_cuda_s']:.6f}",
                            f"mixed_execute_context_cuda_s={phase_cuda['mixed_cuda_s']:.6f}",
                        ] + (
                            [
                                "per_depth_acceptance_rate="
                                + ",".join(f"{r:.6f}" for r in per_pos_rate)
                            ] if per_pos_rate else []
                        ) + (
                            [
                                "acceptance_length_histogram="
                                + ",".join(f"{p:.6f}" for p in acc_len_hist)
                            ] if acc_len_hist else []
                        ) + (
                            [
                                "avg_tree_nodes_per_depth="
                                + ",".join(f"{v:.2f}" for v in avg_tree_shape)
                            ] if avg_tree_shape else []
                        )
                    )
                )

    if "dflash" in modes and "ar" in modes:
        print("=" * 80)
        print("Throughput gains report (DFlash vs AR)")
        gains = []
        gain_lines = []
        for tp_size in args.tp_sizes:
            for batch_size in args.batch_sizes:
                ar_tps = throughputs.get((tp_size, batch_size, "ar"), 0.0)
                dflash_tps = throughputs.get((tp_size, batch_size, "dflash"), 0.0)
                ar_bench_tps = benchmark_tok_s.get((tp_size, batch_size, "ar"), 0.0)
                dflash_bench_tps = benchmark_tok_s.get((tp_size, batch_size, "dflash"), 0.0)
                if ar_tps <= 0:
                    print(f"tp={tp_size} bs={batch_size}: AR throughput unavailable")
                    continue
                gain = dflash_tps / ar_tps
                gains.append(gain)
                gain_pct = (gain - 1.0) * 100.0
                benchmark_gain = (
                    dflash_bench_tps / ar_bench_tps if ar_bench_tps > 0 else 0.0
                )
                ar_cuda = execute_context_cuda.get(
                    (tp_size, batch_size, "ar"),
                    {"prefill_cuda_s": 0.0, "decode_cuda_s": 0.0, "mixed_cuda_s": 0.0},
                )
                dflash_cuda = execute_context_cuda.get(
                    (tp_size, batch_size, "dflash"),
                    {"prefill_cuda_s": 0.0, "decode_cuda_s": 0.0, "mixed_cuda_s": 0.0},
                )
                decode_speedup = (
                    ar_cuda["decode_cuda_s"] / dflash_cuda["decode_cuda_s"]
                    if dflash_cuda["decode_cuda_s"] > 0
                    else 0.0
                )
                prefill_speedup = (
                    ar_cuda["prefill_cuda_s"] / dflash_cuda["prefill_cuda_s"]
                    if dflash_cuda["prefill_cuda_s"] > 0
                    else 0.0
                )
                print(
                    f"tp={tp_size} bs={batch_size}: "
                    f"AR={ar_tps:.2f} tok/s, DFlash={dflash_tps:.2f} tok/s, "
                    f"gain={gain:.3f}x ({gain_pct:+.2f}%), "
                    f"benchmark_gain={benchmark_gain:.3f}x, "
                    f"decode_cuda_speedup={decode_speedup:.3f}x, "
                    f"prefill_cuda_speedup={prefill_speedup:.3f}x"
                )
                gain_lines.append(
                    " ".join(
                        [
                            f"tp={tp_size}",
                            f"bs={batch_size}",
                            f"throughput_gain={gain:.6f}",
                            f"throughput_gain_pct={gain_pct:+.2f}",
                            f"benchmark_gain={benchmark_gain:.6f}",
                            f"ar_benchmark_tok_s={ar_bench_tps:.6f}",
                            f"dflash_benchmark_tok_s={dflash_bench_tps:.6f}",
                            f"decode_cuda_speedup={decode_speedup:.6f}",
                            f"prefill_cuda_speedup={prefill_speedup:.6f}",
                            f"ar_decode_cuda_s={ar_cuda['decode_cuda_s']:.6f}",
                            f"dflash_decode_cuda_s={dflash_cuda['decode_cuda_s']:.6f}",
                            f"ar_prefill_cuda_s={ar_cuda['prefill_cuda_s']:.6f}",
                            f"dflash_prefill_cuda_s={dflash_cuda['prefill_cuda_s']:.6f}",
                        ]
                    )
                )
        if gains:
            avg_gain = sum(gains) / len(gains)
            print(f"Average gain across settings: {avg_gain:.3f}x")
        gains_path = Path(args.torch_profiler_dir) / "gains_report.txt"
        gains_path.parent.mkdir(parents=True, exist_ok=True)
        gains_path.write_text("\n".join(gain_lines) + "\n", encoding="utf-8")
        print(f"Wrote gains report to: {gains_path}")
    summary_path = Path(args.torch_profiler_dir) / "metrics_summary.txt"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print(f"Wrote metrics summary to: {summary_path}")


if __name__ == "__main__":
    main()
