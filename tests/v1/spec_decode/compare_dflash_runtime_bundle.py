from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

DEFAULT_REFERENCE_NAME = "reference_step0_runtime_bundle.pt"
DEFAULT_VLLM_NAME = "vllm_step0_runtime_bundle.pt"


def _resolve_bundle(path_or_dir: str, default_name: str) -> Path:
    path = Path(path_or_dir)
    if path.is_dir():
        return path / default_name
    return path


def _compare_tensor(name: str, reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, Any]:
    result: dict[str, Any] = {
        "name": name,
        "reference_shape": list(reference.shape),
        "candidate_shape": list(candidate.shape),
        "reference_dtype": str(reference.dtype),
        "candidate_dtype": str(candidate.dtype),
    }
    if reference.shape != candidate.shape:
        result["shape_match"] = False
        return result

    result["shape_match"] = True
    if reference.numel() == 0:
        result["empty"] = True
        return result

    if reference.dtype.is_floating_point or candidate.dtype.is_floating_point:
        ref32 = reference.float()
        cand32 = candidate.float()
        diff = (ref32 - cand32).abs()
        result["max_abs_diff"] = float(diff.max().item())
        result["mean_abs_diff"] = float(diff.mean().item())
        result["reference_norm"] = float(ref32.norm().item())
        result["candidate_norm"] = float(cand32.norm().item())
        flat_ref = ref32.reshape(-1)
        flat_cand = cand32.reshape(-1)
        cosine = torch.nn.functional.cosine_similarity(
            flat_ref.unsqueeze(0),
            flat_cand.unsqueeze(0),
        )
        result["cosine_similarity"] = float(cosine.item())
        return result

    unequal = (reference != candidate).sum().item()
    result["exact_match"] = bool(unequal == 0)
    result["num_unequal"] = int(unequal)
    return result


def compare_runtime_bundles(
    *,
    reference_bundle: str,
    vllm_bundle: str,
    output_json: str | None = None,
) -> dict[str, Any]:
    ref_path = _resolve_bundle(reference_bundle, DEFAULT_REFERENCE_NAME)
    vllm_path = _resolve_bundle(vllm_bundle, DEFAULT_VLLM_NAME)
    if not ref_path.exists():
        raise FileNotFoundError(f"Reference bundle not found: {ref_path}")
    if not vllm_path.exists():
        raise FileNotFoundError(f"vLLM bundle not found: {vllm_path}")

    ref = torch.load(ref_path, map_location="cpu")
    cand = torch.load(vllm_path, map_location="cpu")

    tensor_keys = [
        "prompt_token_ids",
        "target_token_ids",
        "target_positions",
        "next_token_ids",
        "raw_target_hidden_states",
        "combined_target_hidden_states",
        "context_positions",
        "query_input_ids",
        "query_positions",
        "token_indices_to_sample",
        "seq_lens",
        "sample_hidden_states_req0",
        "draft_logits_req0",
        "draft_logprobs_req0",
        "topk_tok_0",
        "topk_lp_0",
        "builder_topk_tok",
        "builder_topk_lp",
        "builder_per_depth_entropy",
        "builder_tree_node_token_ids",
        "builder_tree_parent_indices",
        "builder_tree_depths",
        "builder_tree_node_child_ranks",
        "builder_tree_node_cum_logprobs",
        "builder_tree_node_expand_scores",
        "builder_tree_node_added_order",
        "tree_node_token_ids",
        "tree_parent_indices",
        "tree_depths",
        "verify_greedy_tokens",
        "accepted_path",
        "correction_token",
        "accepted_tokens",
        "emitted_tokens",
        "compact_src_slots",
        "compact_dst_slots",
        "post_compact_hidden_states",
        "post_compact_target_hidden_states",
        "post_verify_target_hidden_states",
    ]

    summary: dict[str, Any] = {
        "reference_bundle": str(ref_path),
        "vllm_bundle": str(vllm_path),
        "scalar_fields": {},
        "tensor_comparisons": {},
    }

    scalar_keys = [
        "source",
        "prompt_text",
        "model",
        "draft_model",
        "step",
        "dflash_is_causal",
        "parallel_drafting_token_id",
        "block_size",
        "tree_width",
        "tree_budget",
        "tree_num_nodes",
        "builder_tree_budget",
        "builder_tree_num_nodes",
        "builder_tree_construction",
        "builder_score_mode",
        "builder_hybrid_alpha",
        "accepted_len",
        "seed",
    ]
    for key in scalar_keys:
        summary["scalar_fields"][key] = {
            "reference": ref.get(key),
            "vllm": cand.get(key),
            "match": ref.get(key) == cand.get(key),
        }

    for key in tensor_keys:
        ref_value = ref.get(key)
        cand_value = cand.get(key)
        if not isinstance(ref_value, torch.Tensor) or not isinstance(cand_value, torch.Tensor):
            summary["tensor_comparisons"][key] = {
                "missing": True,
                "reference_type": type(ref_value).__name__,
                "candidate_type": type(cand_value).__name__,
            }
            continue
        summary["tensor_comparisons"][key] = _compare_tensor(key, ref_value, cand_value)

    if isinstance(ref.get("topk_tok_0"), torch.Tensor) and isinstance(cand.get("topk_tok_0"), torch.Tensor):
        ref_topk = ref["topk_tok_0"].tolist()
        cand_topk = cand["topk_tok_0"].tolist()
        summary["topk_overlap"] = {
            "reference": ref_topk,
            "vllm": cand_topk,
            "intersection": sorted(set(ref_topk).intersection(cand_topk)),
        }

    if output_json is not None:
        output_path = Path(output_json)
        output_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    return summary


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare reference and vLLM DFlash runtime bundles.")
    parser.add_argument("--reference-bundle", required=True)
    parser.add_argument("--vllm-bundle", required=True)
    parser.add_argument("--output-json")
    return parser


def main() -> None:
    args = _build_arg_parser().parse_args()
    summary = compare_runtime_bundles(
        reference_bundle=args.reference_bundle,
        vllm_bundle=args.vllm_bundle,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
