# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import copy
from dataclasses import replace
from typing import Any, cast

import torch
from typing_extensions import override

from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.triton_utils import triton
from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.dflash_tree import (
    DraftTree,
    adjust_tree_to_size,
    build_tree_from_topk,
    compute_per_depth_entropy,
    compute_tree_budget,
    find_closest_capture_size,
    prune_and_regrow,
    sample_topk_from_logits,
    tree_signature,
)
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.metadata import DFlashRequestTreeSpec, SpecDecodeMetadata
from vllm.v1.spec_decode.utils import copy_and_expand_dflash_inputs_kernel
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

logger = init_logger(__name__)


class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner=None,
    ):
        assert vllm_config.speculative_config is not None
        assert vllm_config.speculative_config.method == "dflash"
        super().__init__(
            vllm_config=vllm_config,
            device=device,
            pass_hidden_states_to_model=True,
            runner=runner,
        )

        # Only next_token_ids and mask tokens are query tokens, all other context is K/V
        self.max_query_tokens = self.max_batch_size * (1 + self.num_speculative_tokens)
        # Positions covers both context states + query states
        self.max_positions = self.max_num_tokens + self.max_query_tokens

        # Separate context buffers to keep query buffer addresses stable for CUDA graphs
        self._context_slot_mapping_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._slot_mapping_buffer = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )
        self._context_positions_buffer = torch.zeros(
            self.max_num_tokens,
            dtype=torch.int64,
            device=device,
        )
        self.positions = torch.zeros(
            self.max_query_tokens,
            dtype=torch.int64,
            device=device,
        )

        self.arange = torch.arange(
            self.max_positions + 1, device=device, dtype=torch.int32
        )

        # For DFlash we use the input embeddings to embed the mask token
        self.parallel_drafting_hidden_state_tensor = None
        self.dflash_is_causal = self._resolve_causal_head()
        self._last_tree_specs: list[DFlashRequestTreeSpec] | None = None
        self._tree_propose_step = 0
        self._cg_hit_count = 0
        self._cg_miss_count = 0
        self._logged_capture_sizes = False
        self._topk_log: list[dict] = []
        self._pending_topk_log_indices: list[int] = []
        self._runtime_bundles: list[dict[str, Any]] = []
        self._runtime_capture_steps: set[int] = {0}
        self._latest_first_pass_debug: dict[str, Any] | None = None
        self._latest_prepare_next_debug: dict[str, Any] | None = None
        self._latest_prepare_inputs_debug: dict[str, Any] | None = None

    @override
    def prepare_next_token_ids_padded(
        self,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        next_token_ids, valid_sampled_tokens_count = super().prepare_next_token_ids_padded(
            sampled_token_ids=sampled_token_ids,
            requests=requests,
            gpu_input_batch=gpu_input_batch,
            discard_request_mask=discard_request_mask,
        )
        batch_size = sampled_token_ids.shape[0]
        backup_gpu = getattr(self.backup_next_token_ids, "gpu", None)
        self._latest_prepare_next_debug = {
            "prepare_next_sampled_token_ids": sampled_token_ids.detach().cpu(),
            "prepare_next_valid_sampled_tokens_count": (
                valid_sampled_tokens_count.detach().cpu()
            ),
            "prepare_next_next_token_ids": next_token_ids.detach().cpu(),
            "prepare_next_discard_request_mask": discard_request_mask.detach().cpu(),
            "prepare_next_backup_token_ids": (
                backup_gpu[:batch_size].detach().cpu()
                if backup_gpu is not None
                else None
            ),
        }
        return next_token_ids, valid_sampled_tokens_count

    @override
    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        ) = super().prepare_inputs_padded(
            common_attn_metadata=common_attn_metadata,
            spec_decode_metadata=spec_decode_metadata,
            valid_sampled_tokens_count=valid_sampled_tokens_count,
        )
        self._latest_prepare_inputs_debug = {
            "prepare_inputs_query_start_loc": (
                common_attn_metadata.query_start_loc.detach().cpu()
            ),
            "prepare_inputs_seq_lens": common_attn_metadata.seq_lens.detach().cpu(),
            "prepare_inputs_valid_sampled_tokens_count": (
                valid_sampled_tokens_count.detach().cpu()
            ),
            "prepare_inputs_cu_num_draft_tokens": (
                spec_decode_metadata.cu_num_draft_tokens.detach().cpu()
            ),
            "prepare_inputs_token_indices_to_sample": (
                token_indices_to_sample.detach().cpu()
            ),
            "prepare_inputs_num_rejected_tokens": (
                num_rejected_tokens_gpu.detach().cpu()
            ),
            "prepare_inputs_output_query_start_loc": (
                spec_common_attn_metadata.query_start_loc.detach().cpu()
            ),
            "prepare_inputs_output_seq_lens": (
                spec_common_attn_metadata.seq_lens.detach().cpu()
            ),
        }
        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def _resolve_causal_head(self) -> bool:
        head_type = self.speculative_config.head_type
        if head_type == "bidirectional":
            return False
        if head_type == "causal":
            return True
        dflash_config = getattr(self.draft_model_config.hf_config, "dflash_config", {})
        if isinstance(dflash_config, dict):
            return bool(dflash_config.get("causal_head", False))
        return False

    def uses_tree_drafting(self) -> bool:
        return self.speculative_config.tree_width > 1

    def consume_tree_specs(self) -> list[DFlashRequestTreeSpec] | None:
        tree_specs = self._last_tree_specs
        self._last_tree_specs = None
        return tree_specs

    def get_topk_log(self) -> list[dict]:
        """Return accumulated per-step topk diagnostic entries."""
        return self._topk_log

    def record_topk_verify_outcome(
        self,
        *,
        verify_greedy_tokens: torch.Tensor | list[int],
        accepted_len: int,
        correction_token: torch.Tensor | int | None = None,
        tree_num_nodes: int | None = None,
    ) -> None:
        if not self._pending_topk_log_indices:
            return

        entry = self._topk_log[self._pending_topk_log_indices.pop(0)]
        if isinstance(verify_greedy_tokens, torch.Tensor):
            greedy_tokens = [
                int(tok) for tok in verify_greedy_tokens.detach().cpu().tolist()
            ]
        else:
            greedy_tokens = [int(tok) for tok in verify_greedy_tokens]

        topk_tok_0 = [int(tok) for tok in entry.get("topk_tok_0", [])]
        target_next_token = greedy_tokens[0] if greedy_tokens else None
        draft_top1_token = topk_tok_0[0] if topk_tok_0 else None

        if isinstance(correction_token, torch.Tensor):
            correction_value = int(correction_token.item())
        elif correction_token is None:
            correction_value = None
        else:
            correction_value = int(correction_token)

        entry.update(
            {
                "verify_greedy_tokens": greedy_tokens,
                "target_next_token": target_next_token,
                "draft_top1_token": draft_top1_token,
                "draft_top1_match": (
                    draft_top1_token == target_next_token
                    if draft_top1_token is not None and target_next_token is not None
                    else None
                ),
                "target_in_topk": (
                    target_next_token in topk_tok_0
                    if target_next_token is not None
                    else None
                ),
                "accepted_len": int(accepted_len),
                "accepted_d0": bool(accepted_len >= 1),
                "correction_token": correction_value,
                "tree_num_nodes": int(tree_num_nodes)
                if tree_num_nodes is not None
                else None,
            }
        )

    def get_runtime_bundles(self) -> list[dict[str, Any]]:
        """Return captured runtime diagnostics for real-model comparisons."""
        return self._runtime_bundles

    def clear_runtime_bundles(self) -> None:
        self._runtime_bundles.clear()

    def set_runtime_capture_steps(self, steps: list[int] | tuple[int, ...] | None) -> None:
        if steps is None:
            self._runtime_capture_steps = {0}
            return
        parsed = {int(s) for s in steps if int(s) >= 0}
        self._runtime_capture_steps = parsed or {0}

    def get_runtime_capture_steps(self) -> list[int]:
        return sorted(int(s) for s in self._runtime_capture_steps)

    def _capture_runtime_bundle(
        self,
        *,
        raw_target_hidden_states: torch.Tensor,
        combined_target_hidden_states: torch.Tensor,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor,
        common_attn_metadata: CommonAttentionMetadata,
        model_input_ids: torch.Tensor,
        query_positions: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        draft_logits: torch.Tensor,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
    ) -> None:
        current_step = int(self._tree_propose_step)
        if current_step not in self._runtime_capture_steps:
            return
        if any(int(b.get("step", -1)) == current_step for b in self._runtime_bundles):
            return

        num_query_per_req = 1 + self.num_speculative_tokens
        num_spec = self.num_speculative_tokens
        logprobs_req0 = torch.log_softmax(draft_logits[0].float(), dim=-1)
        topk_lp_0, topk_tok_0 = logprobs_req0[0].topk(
            self.speculative_config.tree_width
        )
        self._runtime_bundles.append(
            {
                "step": current_step,
                "dflash_is_causal": self.dflash_is_causal,
                "parallel_drafting_token_id": self.parallel_drafting_token_id,
                "target_token_ids": target_token_ids.detach().cpu(),
                "target_positions": target_positions.detach().cpu(),
                "next_token_ids": next_token_ids.detach().cpu(),
                "raw_target_hidden_states": raw_target_hidden_states.detach().cpu(),
                "combined_target_hidden_states": (
                    combined_target_hidden_states.detach().cpu()
                ),
                "context_positions": self._context_positions_buffer[
                    : target_token_ids.shape[0]
                ].detach().cpu(),
                "context_slot_mapping": self._context_slot_mapping_buffer[
                    : target_token_ids.shape[0]
                ].detach().cpu(),
                "query_input_ids": model_input_ids[:num_query_per_req].detach().cpu(),
                "query_positions": query_positions[:num_query_per_req].detach().cpu(),
                "token_indices_to_sample": (
                    token_indices_to_sample[:num_spec].detach().cpu()
                ),
                "seq_lens": common_attn_metadata.seq_lens.detach().cpu(),
                "num_query_per_req": int(num_query_per_req),
                "num_context": int(target_token_ids.shape[0]),
                "num_rejected_tokens": (
                    num_rejected_tokens_gpu.detach().cpu()
                    if num_rejected_tokens_gpu is not None
                    else None
                ),
                "sample_hidden_states_req0": (
                    sample_hidden_states[:num_spec].detach().cpu()
                ),
                "draft_logits_req0": draft_logits[0].detach().cpu(),
                "draft_logprobs_req0": logprobs_req0.detach().cpu(),
                "topk_tok_0": topk_tok_0.detach().cpu(),
                "topk_lp_0": topk_lp_0.detach().cpu(),
                **(
                    dict(self._latest_prepare_next_debug)
                    if isinstance(self._latest_prepare_next_debug, dict)
                    else {}
                ),
                **(
                    dict(self._latest_prepare_inputs_debug)
                    if isinstance(self._latest_prepare_inputs_debug, dict)
                    else {}
                ),
                **(
                    dict(self._latest_first_pass_debug)
                    if isinstance(self._latest_first_pass_debug, dict)
                    else {}
                ),
            }
        )

    def _capture_tree_builder_runtime_bundle(
        self,
        *,
        tree_budget: int,
        depth_first: bool,
        score_mode: str,
        hybrid_alpha: float,
        per_depth_entropy: list[float] | None,
        topk_tok: torch.Tensor,
        topk_lp: torch.Tensor,
        tree: DraftTree,
    ) -> None:
        current_step = int(self._tree_propose_step)
        if current_step not in self._runtime_capture_steps or not self._runtime_bundles:
            return
        bundle = next(
            (
                b
                for b in reversed(self._runtime_bundles)
                if int(b.get("step", -1)) == current_step
            ),
            None,
        )
        if bundle is None:
            return
        if "builder_tree_node_token_ids" in bundle:
            return

        entropy_tensor = (
            torch.tensor(per_depth_entropy, dtype=torch.float32)
            if per_depth_entropy is not None
            else torch.empty(0, dtype=torch.float32)
        )
        bundle.update(
            {
                "builder_tree_budget": tree_budget,
                "builder_tree_num_nodes": tree.num_nodes,
                "builder_tree_construction": (
                    "depth_first" if depth_first else "breadth_first"
                ),
                "builder_score_mode": score_mode,
                "builder_hybrid_alpha": hybrid_alpha,
                "builder_topk_tok": topk_tok.detach().cpu(),
                "builder_topk_lp": topk_lp.detach().cpu(),
                "builder_per_depth_entropy": entropy_tensor,
                "builder_tree_node_token_ids": tree.token_ids.detach().cpu(),
                "builder_tree_parent_indices": tree.parent_indices.detach().cpu(),
                "builder_tree_depths": tree.depth.detach().cpu(),
            }
        )

    @override
    def _create_draft_vllm_config(self) -> VllmConfig:
        draft_vllm_config = super()._create_draft_vllm_config()
        if self.uses_tree_drafting():
            draft_speculative_config = copy.deepcopy(self.speculative_config)
            draft_speculative_config.tree_width = 1
            return replace(
                draft_vllm_config,
                speculative_config=draft_speculative_config,
            )
        return draft_vllm_config

    @override
    def _raise_if_multimodal(self):
        # Override to allow multimodal inputs since DFlash supports Qwen3.5 models
        # Support for multimodal inputs has not been tested.
        pass

    @override
    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
        num_rejected_tokens_gpu: torch.Tensor | None,
    ) -> tuple[int, torch.Tensor, CommonAttentionMetadata]:
        # DFlash cross-attention: context K/V from target hidden states,
        # Q from query embeddings (bonus + mask tokens).
        batch_size = cad.batch_size()
        num_context = target_token_ids.shape[0]
        num_query_per_req = 1 + self.num_speculative_tokens
        num_query_total = batch_size * num_query_per_req

        # Store for build_model_inputs_first_pass to use
        self._dflash_num_context = num_context

        # We don't need to copy into a buffer here since the context preprocessing
        # does not run in a CUDA graph
        self._dflash_hidden_states = target_hidden_states

        token_indices_to_sample = torch.empty(
            batch_size * self.num_speculative_tokens,
            dtype=torch.int32,
            device=self.device,
        )

        # Launch fused triton kernel for input_ids, positions, slot_mapping,
        # and token_indices_to_sample
        max_ctx_per_req = cad.max_query_len
        max_tokens_per_req = max_ctx_per_req + num_query_per_req
        BLOCK_SIZE = min(256, triton.next_power_of_2(max_tokens_per_req))
        num_blocks = triton.cdiv(max_tokens_per_req, BLOCK_SIZE)
        grid = (batch_size, num_blocks)

        has_num_rejected = num_rejected_tokens_gpu is not None
        copy_and_expand_dflash_inputs_kernel[grid](
            # Inputs
            next_token_ids_ptr=next_token_ids,
            target_positions_ptr=target_positions,
            # Outputs
            out_input_ids_ptr=self.input_ids,
            out_context_positions_ptr=self._context_positions_buffer,
            out_query_positions_ptr=self.positions,
            out_context_slot_mapping_ptr=self._context_slot_mapping_buffer,
            out_query_slot_mapping_ptr=self._slot_mapping_buffer,
            out_token_indices_ptr=token_indices_to_sample,
            # Block table
            block_table_ptr=cad.block_table_tensor,
            block_table_stride=cad.block_table_tensor.stride(0),
            # Metadata
            query_start_loc_ptr=cad.query_start_loc,
            num_rejected_tokens_ptr=(
                num_rejected_tokens_gpu if has_num_rejected else 0
            ),
            # Scalars
            parallel_drafting_token_id=self.parallel_drafting_token_id,
            block_size=self.block_size,
            num_query_per_req=num_query_per_req,
            num_speculative_tokens=self.num_speculative_tokens,
            total_input_tokens=num_context,
            BLOCK_SIZE=BLOCK_SIZE,
            HAS_NUM_REJECTED=has_num_rejected,
        )

        query_slot_mapping = self._slot_mapping_buffer[:num_query_total]
        new_query_start_loc = self.arange[: batch_size + 1] * num_query_per_req
        input_context_lens = cad.query_start_loc[1:] - cad.query_start_loc[:-1]

        # In padded mode, cad.seq_lens includes rejected tokens. Subtract
        # them so attention only sees the valid prefix of context states.
        effective_seq_lens = cad.seq_lens
        compacted_context_lens = input_context_lens
        if has_num_rejected:
            effective_seq_lens = effective_seq_lens - num_rejected_tokens_gpu
            compacted_context_lens = compacted_context_lens - num_rejected_tokens_gpu
        visible_context_lens = effective_seq_lens
        output_seq_lens = effective_seq_lens + num_query_per_req
        output_max_seq_len = cad.max_seq_len + num_query_per_req

        self._latest_first_pass_debug = {
            "input_cad_query_start_loc": cad.query_start_loc.detach().cpu(),
            "input_cad_seq_lens": cad.seq_lens.detach().cpu(),
            "input_context_lens_from_query_start_loc": (
                input_context_lens.detach().cpu()
            ),
            "input_seq_lens_minus_rejected": effective_seq_lens.detach().cpu(),
            "input_compacted_context_lens": compacted_context_lens.detach().cpu(),
            "output_visible_context_lens": visible_context_lens.detach().cpu(),
            "output_query_start_loc": new_query_start_loc.detach().cpu(),
            "output_seq_lens": output_seq_lens.detach().cpu(),
            "output_max_seq_len": output_max_seq_len,
            "output_query_slot_mapping_head": query_slot_mapping[
                : min(16, query_slot_mapping.shape[0])
            ].detach().cpu(),
            "input_cad_max_query_len": int(cad.max_query_len),
        }

        new_cad = CommonAttentionMetadata(
            query_start_loc=new_query_start_loc,
            seq_lens=output_seq_lens,
            query_start_loc_cpu=(
                torch.from_numpy(self.token_arange_np[: batch_size + 1]).clone()
                * num_query_per_req
            ),
            _seq_lens_cpu=None,
            _num_computed_tokens_cpu=None,
            num_reqs=cad.num_reqs,
            num_actual_tokens=num_query_total,
            max_query_len=num_query_per_req,
            max_seq_len=output_max_seq_len,
            block_table_tensor=cad.block_table_tensor,
            slot_mapping=query_slot_mapping,
            causal=self.dflash_is_causal,
        )

        return num_query_total, token_indices_to_sample, new_cad

    @override
    @torch.inference_mode()
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """
        Key differences to default dummy_run:
        - Only one forward pass due to parallel drafting
        - DFlash uses context states as unpadded metadata, so hidden_states will
        use the unpadded num_tokens instead of num_input_tokens
        - max_query_tokens is quite small, DFlash only sees spec tokens as queries
        - Multimodal inputs are not currently supported
        """
        num_query_tokens = min(num_tokens, self.max_query_tokens)
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(
                num_query_tokens, use_cudagraphs=use_cudagraphs
            )
        )

        # Slot mapping sized to num_input_tokens (query only), matching
        # the K/V tensor size from the model forward.  Context KVs are
        # pre-inserted separately and don't flow through the model.
        if (
            self._draft_attn_layer_names
            and slot_mappings is not None
            and next(iter(self._draft_attn_layer_names)) in slot_mappings
        ):
            slot_mapping_dict = self._get_slot_mapping(num_input_tokens)
        else:
            slot_mapping_dict = slot_mappings or {}

        # Context and query positions use separate buffers; no copy needed.
        context_positions = self._context_positions_buffer[:num_tokens]
        # Context states will be passed directly to the precomputation without
        # going through the buffer, since no CUDA graph is used for the precomputation.
        # For the dummy run, we use the dummy buffer.
        context_states = self.hidden_states[:num_tokens]

        # Precompute context KVs for memory profiling / warmup.  Skip during
        # CUDAGraph capture since this runs outside the captured graph.
        if not is_graph_capturing:
            self.model.precompute_and_store_context_kv(
                context_states, context_positions,
            )
        with set_forward_context(
            None,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mapping_dict,
        ):
            self.model(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            )

    @override
    def build_model_inputs_first_pass(
        self,
        num_tokens: int,
        num_input_tokens: int,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None,
    ) -> tuple[dict[str, Any], int]:
        # Context and query positions/slots were written to separate
        # buffers by the kernel — no copy needed.
        num_context = self._dflash_num_context

        # Pre-insert context KVs directly into cache
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # Shape is already [num_context, hidden_size]
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
        return (
            dict(
                input_ids=self.input_ids[:num_input_tokens],
                positions=self._get_positions(num_input_tokens),
                inputs_embeds=None,
            ),
            num_input_tokens,
        )

    @override
    def build_per_group_and_layer_attn_metadata(
        self, cad: CommonAttentionMetadata, draft_index: int = 0
    ) -> tuple[list[object], dict[str, object]]:
        per_group, per_layer = super().build_per_group_and_layer_attn_metadata(
            cad, draft_index
        )
        for layer_name, attn_metadata in per_layer.items():
            assert getattr(attn_metadata, "causal", None) is self.dflash_is_causal, (
                f"Attention metadata for layer {layer_name} does not match the "
                f"resolved DFlash causal mode ({self.dflash_is_causal})."
            )
        return per_group, per_layer

    @override
    def _get_eagle3_use_aux_hidden_state_from_config(self):
        use_aux_hidden_state = True
        dflash_config = getattr(
            self.draft_model_config.hf_config, "dflash_config", None
        )
        if dflash_config is not None:
            use_aux_hidden_state = dflash_config.get("use_aux_hidden_state", True)
        return use_aux_hidden_state

    @override
    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor | list[list[int]]:
        if not self.uses_tree_drafting():
            self._last_tree_specs = None
            return super().propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                sampling_metadata=sampling_metadata,
                mm_embed_inputs=mm_embed_inputs,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
                slot_mappings=slot_mappings,
            )

        if not self.dflash_is_causal:
            raise NotImplementedError(
                "Native DFlash tree drafting only supports causal heads."
            )

        batch_size = common_attn_metadata.batch_size()
        _diag = self._tree_propose_step < 3
        if _diag:
            _raw_hs = target_hidden_states
            logger.info(
                "[DIAG step=%d] raw_hidden shape=%s norm=%.6f "
                "first5=%s last5=%s",
                self._tree_propose_step, list(_raw_hs.shape),
                _raw_hs.float().norm().item(),
                _raw_hs[0, :5].tolist(), _raw_hs[0, -5:].tolist(),
            )
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)
        assert target_hidden_states.shape[-1] == self.hidden_size
        if _diag:
            logger.info(
                "[DIAG step=%d] after_fc shape=%s norm=%.6f "
                "first5=%s last5=%s",
                self._tree_propose_step, list(target_hidden_states.shape),
                target_hidden_states.float().norm().item(),
                target_hidden_states[0, :5].tolist(),
                target_hidden_states[0, -5:].tolist(),
            )
            logger.info(
                "[DIAG step=%d] num_ctx=%d next_token_ids=%s "
                "target_positions[:5]=%s target_positions[-5:]=%s "
                "seq_lens=%s",
                self._tree_propose_step,
                target_token_ids.shape[0],
                next_token_ids.tolist(),
                target_positions[:5].tolist(),
                target_positions[-5:].tolist(),
                common_attn_metadata.seq_lens.tolist(),
            )

        num_tokens, token_indices_to_sample, common_attn_metadata = (
            self.set_inputs_first_pass(
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                token_indices_to_sample=token_indices_to_sample,
                cad=common_attn_metadata,
                num_rejected_tokens_gpu=num_rejected_tokens_gpu,
            )
        )

        per_group_attn_metadata, per_layer_attn_metadata = (
            self.build_per_group_and_layer_attn_metadata(common_attn_metadata)
        )
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = (
            self._determine_batch_execution_and_padding(num_tokens)
        )
        model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(
            num_tokens, num_input_tokens, mm_embed_inputs
        )

        if _diag:
            _ids = model_kwargs.get("input_ids")
            logger.info(
                "[DIAG step=%d] query_input_ids=%s "
                "query_positions=%s num_tokens=%d "
                "num_input_tokens=%d",
                self._tree_propose_step,
                _ids[:num_tokens].tolist() if _ids is not None else "N/A",
                self.positions[:num_tokens].tolist(),
                num_tokens, num_input_tokens,
            )

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=self._get_slot_mapping(
                slot_mapping_size, common_attn_metadata.slot_mapping
            ),
        ):
            ret_hidden_states = self.model(**model_kwargs)
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
            else:
                last_hidden_states, _ = ret_hidden_states

        sample_hidden_states = last_hidden_states[token_indices_to_sample]
        draft_logits = self.model.compute_logits(sample_hidden_states).view(
            batch_size, self.num_speculative_tokens, -1
        )
        self._capture_runtime_bundle(
            raw_target_hidden_states=_raw_hs if _diag else target_hidden_states,
            combined_target_hidden_states=target_hidden_states,
            target_token_ids=target_token_ids,
            target_positions=target_positions,
            next_token_ids=next_token_ids,
            token_indices_to_sample=token_indices_to_sample,
            common_attn_metadata=common_attn_metadata,
            model_input_ids=cast(torch.Tensor, model_kwargs["input_ids"]),
            query_positions=self.positions,
            sample_hidden_states=sample_hidden_states,
            draft_logits=draft_logits,
            num_rejected_tokens_gpu=num_rejected_tokens_gpu,
        )
        if _diag:
            _lp0 = torch.log_softmax(draft_logits[0, 0], dim=-1)
            _top5_lp, _top5_tok = _lp0.topk(5)
            logger.info(
                "[DIAG step=%d] draft_logits_d0 top5_tok=%s "
                "top5_lp=%s argmax=%d",
                self._tree_propose_step,
                _top5_tok.tolist(), _top5_lp.tolist(),
                _lp0.argmax().item(),
            )

        tree_budget = compute_tree_budget(
            self.num_speculative_tokens + 1,
            self.speculative_config.tree_width,
            self.speculative_config.max_tree_budget,
        )
        tree_width = self.speculative_config.tree_width
        block_size = self.num_speculative_tokens + 1
        query_len = 1 + self.num_speculative_tokens
        base_query_input_ids = self.input_ids[:num_tokens].view(
            batch_size, query_len
        ).clone()

        if not self._logged_capture_sizes:
            self._logged_capture_sizes = True
            cs = self.speculative_config.cudagraph_tree_capture_sizes
            if cs is not None:
                logger.info(
                    "DFlash tree CUDAGraph capture sizes (%d): %s",
                    len(cs), cs,
                )
            else:
                logger.info("DFlash tree CUDAGraph multi-size capture: disabled")

        device = draft_logits.device

        depth_first = (
            self.speculative_config.tree_construction == "depth_first"
        )
        score_mode = self.speculative_config.tree_draft
        hybrid_alpha = self.speculative_config.tree_hybrid_alpha

        per_depth_entropies: list[list[float] | None]
        if score_mode in ("entropy", "hybrid"):
            per_depth_entropies = [
                compute_per_depth_entropy(draft_logits[req_idx])
                for req_idx in range(batch_size)
            ]
        else:
            per_depth_entropies = [None] * batch_size

        trees: list[DraftTree] = []
        for req_idx in range(batch_size):
            topk_lp, topk_tok = sample_topk_from_logits(
                draft_logits[req_idx], tree_width,
            )
            root_token = next_token_ids[req_idx].item()
            topk_tok_0 = topk_tok[0].tolist()
            topk_lp_0 = topk_lp[0].tolist()
            self._pending_topk_log_indices.append(len(self._topk_log))
            self._topk_log.append({
                "step": self._tree_propose_step,
                "req": req_idx,
                "root_token": root_token,
                "topk_tok_0": topk_tok_0,
                "topk_lp_0": topk_lp_0,
            })
            if self._tree_propose_step < 8 or self._tree_propose_step % 50 == 0:
                logger.info(
                    "[tree-propose step=%d req=%d] root_token=%d  "
                    "topk_tok[0]=%s  topk_lp[0]=%s",
                    self._tree_propose_step, req_idx, root_token,
                    topk_tok_0, topk_lp_0,
                )
            tree = build_tree_from_topk(
                root_token, topk_tok, topk_lp, tree_budget, device,
                depth_first=depth_first,
                score_mode=score_mode,
                per_depth_entropy=per_depth_entropies[req_idx],
                hybrid_alpha=hybrid_alpha,
            )
            trees.append(tree)
            if req_idx == 0:
                self._capture_tree_builder_runtime_bundle(
                    tree_budget=tree_budget,
                    depth_first=depth_first,
                    score_mode=score_mode,
                    hybrid_alpha=hybrid_alpha,
                    per_depth_entropy=per_depth_entropies[req_idx],
                    topk_tok=topk_tok,
                    topk_lp=topk_lp,
                    tree=tree,
                )

        last_cond_logits: torch.Tensor | None = None

        num_passes = max(
            0, getattr(self.speculative_config, "max_draft_passes", 0),
        )
        if num_passes > 0:
            for _ in range(num_passes):
                conditioned_input_ids = base_query_input_ids.clone()
                for req_idx, tree in enumerate(trees):
                    best_path = tree.longest_path()
                    if len(best_path) <= 1:
                        continue
                    tok_ids = tree.token_ids[best_path[1:]]
                    conditioned_input_ids[req_idx, 1:1 + len(tok_ids)] = tok_ids

                cond_model_kwargs = dict(model_kwargs)
                cond_input_ids = cast(
                    torch.Tensor, model_kwargs["input_ids"],
                ).clone()
                cond_input_ids[:num_tokens] = conditioned_input_ids.reshape(-1)
                cond_model_kwargs["input_ids"] = cond_input_ids
                with set_forward_context(
                    per_layer_attn_metadata,
                    self.vllm_config,
                    num_tokens=num_input_tokens,
                    num_tokens_across_dp=num_tokens_across_dp,
                    cudagraph_runtime_mode=cudagraph_runtime_mode,
                    slot_mapping=self._get_slot_mapping(
                        slot_mapping_size, common_attn_metadata.slot_mapping
                    ),
                ):
                    cond_ret = self.model(**cond_model_kwargs)
                    if not self.model_returns_tuple():
                        cond_last_hidden_states = cond_ret
                    else:
                        cond_last_hidden_states, _ = cond_ret
                cond_hidden_states = (
                    cond_last_hidden_states[token_indices_to_sample]
                )
                cond_logits = self.model.compute_logits(
                    cond_hidden_states,
                ).view(batch_size, self.num_speculative_tokens, -1)

                new_trees: list[DraftTree] = []
                prune_ratio = self.speculative_config.tree_prune_ratio
                for req_idx, tree in enumerate(trees):
                    new_trees.append(prune_and_regrow(
                        tree=tree,
                        cond_logits=cond_logits[req_idx],
                        block_size=block_size,
                        tree_width=tree_width,
                        budget=tree_budget,
                        device=device,
                        prune_ratio=prune_ratio,
                    ))
                trees = new_trees
                last_cond_logits = cond_logits

        capture_sizes = self.speculative_config.cudagraph_tree_capture_sizes
        if capture_sizes is not None:
            for req_idx, tree in enumerate(trees):
                orig_size = tree.num_nodes
                target = find_closest_capture_size(
                    orig_size, capture_sizes,
                )
                is_hit = (orig_size == target)
                if is_hit:
                    self._cg_hit_count += 1
                else:
                    self._cg_miss_count += 1
                    trees[req_idx] = adjust_tree_to_size(
                        tree,
                        target,
                        cond_logits=(
                            last_cond_logits[req_idx]
                            if last_cond_logits is not None else None
                        ),
                        block_size=block_size,
                        tree_width=tree_width,
                        device=device,
                    )

                log_detail = (self._tree_propose_step < 5
                              or self._tree_propose_step % 50 == 0)
                if log_detail:
                    sig = tree_signature(trees[req_idx])
                    tag = "CG-HIT " if is_hit else "CG-MISS"
                    logger.info(
                        "[step=%d req=%d] %s orig=%d target=%d  %s",
                        self._tree_propose_step, req_idx, tag,
                        orig_size, target, sig,
                    )

        self._tree_propose_step += 1
        if (self._tree_propose_step % 100 == 0
                and (self._cg_hit_count + self._cg_miss_count) > 0):
            total = self._cg_hit_count + self._cg_miss_count
            logger.info(
                "CUDAGraph tree stats after %d steps: hits=%d (%.1f%%) "
                "misses=%d (%.1f%%)",
                self._tree_propose_step,
                self._cg_hit_count, 100.0 * self._cg_hit_count / total,
                self._cg_miss_count, 100.0 * self._cg_miss_count / total,
            )

        self._last_tree_specs = []
        draft_token_ids: list[list[int]] = []
        for tree in trees:
            tids = tree.token_ids[1:].tolist()
            pids = tree.parent_indices[1:].tolist()
            deps = tree.depth[1:].tolist()
            draft_token_ids.append(tids)
            self._last_tree_specs.append(
                DFlashRequestTreeSpec(
                    parent_indices=pids,
                    depths=deps,
                )
            )
        return draft_token_ids
