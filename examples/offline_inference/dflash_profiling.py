#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import re
import time
from pathlib import Path

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


def get_prompt_bank(prompt_set: str) -> list[str]:
    if prompt_set == "mix":
        return DEFAULT_PROMPTS
    if prompt_set == "coding":
        return CODING_PROMPTS
    raise ValueError(f"Unknown prompt set: {prompt_set}")


def build_prompts(batch_size: int, prompt_bank: list[str]) -> list[str]:
    if batch_size < 1:
        raise ValueError(f"Invalid batch size: {batch_size}. Expected batch size >= 1")
    repeat = (batch_size + len(prompt_bank) - 1) // len(prompt_bank)
    return (prompt_bank * repeat)[:batch_size]


def get_modes(mode: str) -> list[str]:
    if mode == "both":
        return ["ar", "dflash"]
    return [mode]


def collect_spec_decode_counters(metrics) -> dict[str, float]:
    counters = {
        "num_drafts": 0.0,
        "num_draft_tokens": 0.0,
        "num_accepted_tokens": 0.0,
    }
    for metric in metrics:
        name = getattr(metric, "name", "")
        value = getattr(metric, "value", None)
        if value is None:
            continue
        if name == "vllm:spec_decode_num_drafts":
            counters["num_drafts"] += float(value)
        elif name == "vllm:spec_decode_num_draft_tokens":
            counters["num_draft_tokens"] += float(value)
        elif name == "vllm:spec_decode_num_accepted_tokens":
            counters["num_accepted_tokens"] += float(value)
    return counters


def diff_counters(after: dict[str, float], before: dict[str, float]) -> dict[str, float]:
    return {k: after.get(k, 0.0) - before.get(k, 0.0) for k in after}


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
        choices=["mix", "coding"],
        help=(
            "Prompt set to use. "
            "'mix' uses general profiling prompts; 'coding' uses 4 Python "
            "algorithm/data-structure tasks."
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
        "--max-model-len",
        type=int,
        default=32768,
        help="Max model length.",
    )
    parser.add_argument(
        "--tp-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Tensor parallel sizes to profile (e.g. --tp-sizes 1 2 4 8).",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16],
        help="Batch sizes to profile (e.g. --batch-sizes 1 4 8 16).",
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
        default=128,
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
            "Optional attention backend override, e.g. FLASH_ATTN, TRITON_ATTN, "
            "or FLEX_ATTENTION."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.num_runs < 1:
        raise ValueError("--num-runs must be >= 1")
    if args.num_warmup_runs < 0:
        raise ValueError("--num-warmup-runs must be >= 0")

    sampling_params = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
    )
    prompt_bank = get_prompt_bank(args.prompt_set)
    modes = get_modes(args.mode)
    # key: (tp_size, batch_size, mode) -> throughput tokens/s
    throughputs: dict[tuple[int, int, str], float] = {}
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
            prompts = build_prompts(batch_size, prompt_bank)

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

                llm_kwargs = dict(
                    model=args.model,
                    trust_remote_code=args.trust_remote_code,
                    tensor_parallel_size=tp_size,
                    gpu_memory_utilization=args.gpu_memory_utilization,
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
                        "num_speculative_tokens": args.num_speculative_tokens,
                        "max_model_len": args.max_model_len,
                    }

                llm = LLM(**llm_kwargs)

                for _ in range(args.num_warmup_runs):
                    llm.generate(prompts, sampling_params=sampling_params)

                metrics_before = collect_spec_decode_counters(llm.get_metrics())
                llm.start_profile()
                t0 = time.perf_counter()
                total_output_tokens = 0
                for _ in range(args.num_runs):
                    outputs = llm.generate(prompts, sampling_params=sampling_params)
                    total_output_tokens += sum(
                        len(output.outputs[0].token_ids) for output in outputs
                    )
                elapsed = time.perf_counter() - t0
                llm.stop_profile()
                metrics_after = collect_spec_decode_counters(llm.get_metrics())
                metrics_delta = diff_counters(metrics_after, metrics_before)

                throughput = total_output_tokens / elapsed if elapsed > 0 else 0.0
                throughputs[(tp_size, batch_size, mode)] = throughput
                draft_tokens = metrics_delta["num_draft_tokens"]
                accepted_tokens = metrics_delta["num_accepted_tokens"]
                drafts = metrics_delta["num_drafts"]
                acceptance_rate = (
                    accepted_tokens / draft_tokens if draft_tokens > 0 else 0.0
                )
                acceptance_length = 1.0 + (accepted_tokens / drafts if drafts > 0 else 0.0)
                print(
                    f"[RESULT] mode={mode} tp={tp_size} bs={batch_size} "
                    f"output_tokens={total_output_tokens} elapsed_s={elapsed:.3f} "
                    f"throughput_tok_s={throughput:.2f}"
                )
                print(
                    f"[SPEC_METRICS] mode={mode} tp={tp_size} bs={batch_size} "
                    f"num_drafts={drafts:.0f} num_draft_tokens={draft_tokens:.0f} "
                    f"num_accepted_tokens={accepted_tokens:.0f} "
                    f"acceptance_rate={acceptance_rate:.4f} "
                    f"acceptance_length={acceptance_length:.4f}"
                )
                for output in outputs:
                    print("-" * 80)
                    print(f"Prompt: {output.prompt}")
                    print(f"Generated: {output.outputs[0].text}")

                del llm
                # Add a buffer for background workers to flush profile data.
                time.sleep(args.sleep_after_stop)
                phase_cuda = collect_execute_context_cuda_seconds(run_output_dir)
                execute_context_cuda[(tp_size, batch_size, mode)] = phase_cuda

                report_lines = [
                    f"mode={mode}",
                    f"prompt_set={args.prompt_set}",
                    f"attention_backend={args.attention_backend}",
                    f"tp_size={tp_size}",
                    f"batch_size={batch_size}",
                    f"output_tokens={total_output_tokens}",
                    f"elapsed_s={elapsed:.6f}",
                    f"throughput_tok_s={throughput:.6f}",
                    f"num_drafts={drafts:.0f}",
                    f"num_draft_tokens={draft_tokens:.0f}",
                    f"num_accepted_tokens={accepted_tokens:.0f}",
                    f"acceptance_rate={acceptance_rate:.6f}",
                    f"acceptance_length={acceptance_length:.6f}",
                    f"prefill_execute_context_cuda_s={phase_cuda['prefill_cuda_s']:.6f}",
                    f"decode_execute_context_cuda_s={phase_cuda['decode_cuda_s']:.6f}",
                    f"mixed_execute_context_cuda_s={phase_cuda['mixed_cuda_s']:.6f}",
                ]
                (run_output_dir / "metrics_report.txt").write_text(
                    "\n".join(report_lines) + "\n", encoding="utf-8"
                )
                summary_lines.append(
                    " ".join(
                        [
                            f"mode={mode}",
                            f"prompt_set={args.prompt_set}",
                            f"attention_backend={args.attention_backend}",
                            f"tp={tp_size}",
                            f"bs={batch_size}",
                            f"throughput_tok_s={throughput:.6f}",
                            f"num_drafts={drafts:.0f}",
                            f"num_draft_tokens={draft_tokens:.0f}",
                            f"num_accepted_tokens={accepted_tokens:.0f}",
                            f"acceptance_rate={acceptance_rate:.6f}",
                            f"acceptance_length={acceptance_length:.6f}",
                            f"prefill_execute_context_cuda_s={phase_cuda['prefill_cuda_s']:.6f}",
                            f"decode_execute_context_cuda_s={phase_cuda['decode_cuda_s']:.6f}",
                            f"mixed_execute_context_cuda_s={phase_cuda['mixed_cuda_s']:.6f}",
                        ]
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
                if ar_tps <= 0:
                    print(f"tp={tp_size} bs={batch_size}: AR throughput unavailable")
                    continue
                gain = dflash_tps / ar_tps
                gains.append(gain)
                gain_pct = (gain - 1.0) * 100.0
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
