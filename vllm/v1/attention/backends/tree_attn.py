# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Attention layer with TreeAttention."""

import ast
from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm import _custom_ops as ops
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)
from vllm.v1.attention.backends.utils import (
    split_decodes_and_prefills,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)


class TreeAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "float16",
        "bfloat16",
    ]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "TREE_ATTN"

    @staticmethod
    def get_impl_cls() -> type["TreeAttentionImpl"]:
        return TreeAttentionImpl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_builder_cls() -> type["TreeAttentionMetadataBuilder"]:
        return TreeAttentionMetadataBuilder

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


@dataclass
class TreeAttentionMetadata:
    num_actual_tokens: int  # Number of tokens excluding padding.
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    num_prefills: int = 0
    num_decodes: int = 0

    tree_attn_bias: torch.Tensor | None = None
    ancestor_masks: torch.Tensor | None = None

    # Cached Prefill/decode metadata.
    _cached_prefill_metadata: "TreeAttentionMetadata | None" = None
    _cached_decode_metadata: "TreeAttentionMetadata | None" = None

    @property
    def prefill_metadata(self) -> "TreeAttentionMetadata | None":
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            # Recover cached prefill-phase attention
            # metadata structure
            return self._cached_prefill_metadata

        q_start_loc = self.query_start_loc[self.num_decodes :]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[self.num_decodes :]
        # Construct & cache prefill-phase attention metadata structure
        self._cached_prefill_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_prefill_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc - q_start_loc[0],
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[self.num_decodes :],
            slot_mapping=self.slot_mapping[self.num_decode_tokens :],
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> "TreeAttentionMetadata | None":
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            # Recover cached decode-phase attention
            # metadata structure
            return self._cached_decode_metadata

        q_start_loc = self.query_start_loc[: self.num_decodes + 1]
        q_seqlens = torch.diff(q_start_loc)
        kv_seqlens = self.seq_lens[: self.num_decodes]
        # Construct & cache decode-phase attention metadata structure
        self._cached_decode_metadata = TreeAttentionMetadata(
            num_actual_tokens=self.num_decode_tokens,
            max_query_len=int(q_seqlens.max().item()),
            query_start_loc=q_start_loc,
            max_seq_len=int(kv_seqlens.max().item()),
            seq_lens=kv_seqlens,
            block_table=self.block_table[: self.num_decodes],
            slot_mapping=self.slot_mapping[: self.num_decode_tokens],
            tree_attn_bias=self.tree_attn_bias,
            ancestor_masks=self.ancestor_masks,
        )
        return self._cached_decode_metadata


class TreeAttentionMetadataBuilder(AttentionMetadataBuilder[TreeAttentionMetadata]):
    _cudagraph_support: ClassVar[AttentionCGSupport] = (
        AttentionCGSupport.UNIFORM_BATCH
    )

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size

        spec_config = vllm_config.speculative_config
        spec_token_tree: str | None = None
        if spec := spec_config:
            spec_token_tree = spec.speculative_token_tree
        tree_choices: list[tuple[int, ...]] = (
            ast.literal_eval(spec_token_tree) if spec_token_tree is not None else [(0,)]
        )
        # Construct the tree attention bias.
        depth_counts = _get_depth_counts(tree_choices)
        self.tree_attn_bias = _prepare_tree_attn_bias(
            tree_choices,
            depth_counts,
            dtype=torch.float32,
            device=device,
        )

        self.reorder_batch_threshold = self.tree_attn_bias.shape[0]
        self._cudagraph_tree_attn_bias: torch.Tensor | None = None
        self._cudagraph_ancestor_masks: torch.Tensor | None = None
        self._max_cudagraph_tree_query_len = self._init_max_cudagraph_tree_query_len()
        self._max_cudagraph_batch_size = vllm_config.scheduler_config.max_num_seqs
        self._dflash_tree_debug_records: list[dict[str, object]] = []

    def get_dflash_tree_debug_records(self) -> list[dict[str, object]]:
        return list(self._dflash_tree_debug_records)

    def clear_dflash_tree_debug_records(self) -> None:
        self._dflash_tree_debug_records.clear()

    def _append_dflash_tree_debug_record(
        self,
        *,
        build_method: str,
        common_attn_metadata: CommonAttentionMetadata,
        output_metadata: TreeAttentionMetadata,
        tree_attn_bias: torch.Tensor | None,
        ancestor_masks: torch.Tensor | None,
        debug_context: dict[str, object],
    ) -> None:
        decode_meta = output_metadata.decode_metadata
        self._dflash_tree_debug_records.append(
            {
                "build_method": build_method,
                "caller_role": str(debug_context.get("caller_role", "unknown")),
                "builder_owner": str(debug_context.get("builder_owner", "unknown")),
                "tree_propose_step": int(debug_context.get("tree_propose_step", -1)),
                "draft_index": (
                    int(debug_context.get("draft_index", -1))
                    if debug_context.get("draft_index") is not None
                    else None
                ),
                "for_cudagraph_capture": bool(
                    debug_context.get("for_cudagraph_capture", False)
                ),
                "input_num_actual_tokens": int(common_attn_metadata.num_actual_tokens),
                "input_num_reqs": int(common_attn_metadata.num_reqs),
                "input_max_query_len": int(common_attn_metadata.max_query_len),
                "input_max_seq_len": int(common_attn_metadata.max_seq_len),
                "input_query_start_loc": common_attn_metadata.query_start_loc.detach()
                .cpu(),
                "input_seq_lens": common_attn_metadata.seq_lens.detach().cpu(),
                "input_slot_mapping": common_attn_metadata.slot_mapping.detach().cpu(),
                "tree_attn_bias_shape": (
                    list(tree_attn_bias.shape) if tree_attn_bias is not None else None
                ),
                "ancestor_masks_shape": (
                    list(ancestor_masks.shape)
                    if ancestor_masks is not None
                    else None
                ),
                "output_max_query_len": int(output_metadata.max_query_len),
                "output_max_seq_len": int(output_metadata.max_seq_len),
                "output_query_start_loc": output_metadata.query_start_loc.detach().cpu(),
                "output_seq_lens": output_metadata.seq_lens.detach().cpu(),
                "decode_max_query_len": (
                    int(decode_meta.max_query_len) if decode_meta is not None else None
                ),
                "decode_max_seq_len": (
                    int(decode_meta.max_seq_len) if decode_meta is not None else None
                ),
                "decode_query_start_loc": (
                    decode_meta.query_start_loc.detach().cpu()
                    if decode_meta is not None
                    else None
                ),
                "decode_seq_lens": (
                    decode_meta.seq_lens.detach().cpu()
                    if decode_meta is not None
                    else None
                ),
            }
        )

    def _init_max_cudagraph_tree_query_len(self) -> int:
        capture_hints = self.vllm_config.compilation_config.cudagraph_capture_sizes
        if capture_hints:
            return max(capture_hints)
        return self.tree_attn_bias.shape[0]

    def _copy_tree_attn_bias_for_cudagraph(
        self, tree_attn_bias: torch.Tensor
    ) -> torch.Tensor:
        query_len = tree_attn_bias.shape[0]
        if query_len > self._max_cudagraph_tree_query_len:
            raise ValueError(
                "Tree attention bias exceeds cudagraph capture capacity: "
                f"{query_len} > {self._max_cudagraph_tree_query_len}"
            )
        if self._cudagraph_tree_attn_bias is None:
            self._cudagraph_tree_attn_bias = torch.empty(
                (
                    self._max_cudagraph_tree_query_len,
                    self._max_cudagraph_tree_query_len,
                ),
                dtype=tree_attn_bias.dtype,
                device=self.device,
            )
        self._cudagraph_tree_attn_bias[:query_len, :query_len].copy_(tree_attn_bias)
        return self._cudagraph_tree_attn_bias[:query_len, :query_len]

    def _copy_ancestor_masks_for_cudagraph(
        self, ancestor_masks: torch.Tensor,
    ) -> torch.Tensor:
        B, N, _ = ancestor_masks.shape
        max_B = self._max_cudagraph_batch_size
        max_N = self._max_cudagraph_tree_query_len
        if self._cudagraph_ancestor_masks is None:
            self._cudagraph_ancestor_masks = torch.zeros(
                (max_B, max_N, max_N), dtype=torch.int32, device=self.device
            )
        self._cudagraph_ancestor_masks[:B, :N, :N].copy_(ancestor_masks)
        return self._cudagraph_ancestor_masks[:B, :N, :N]

    def build_for_dflash_tree(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        tree_attn_bias: torch.Tensor | None,
        *,
        for_cudagraph_capture: bool = False,
        ancestor_masks: torch.Tensor | None = None,
    ) -> TreeAttentionMetadata:
        debug_context = getattr(self, "_dflash_tree_debug_context", {}) or {}
        if tree_attn_bias is not None and for_cudagraph_capture:
            tree_attn_bias = self._copy_tree_attn_bias_for_cudagraph(tree_attn_bias)
        if ancestor_masks is not None and for_cudagraph_capture:
            ancestor_masks = self._copy_ancestor_masks_for_cudagraph(
                ancestor_masks
            )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        num_reqs = common_attn_metadata.num_reqs

        decode_meta = TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            tree_attn_bias=tree_attn_bias,
            ancestor_masks=ancestor_masks,
        )

        meta = TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            num_prefill_tokens=0,
            num_decode_tokens=num_actual_tokens,
            num_prefills=0,
            num_decodes=num_reqs,
            tree_attn_bias=tree_attn_bias,
            ancestor_masks=ancestor_masks,
            _cached_decode_metadata=decode_meta,
        )
        self._append_dflash_tree_debug_record(
            build_method="build_for_dflash_tree",
            common_attn_metadata=common_attn_metadata,
            output_metadata=meta,
            tree_attn_bias=tree_attn_bias,
            ancestor_masks=ancestor_masks,
            debug_context={
                **debug_context,
                "for_cudagraph_capture": bool(
                    debug_context.get("for_cudagraph_capture", for_cudagraph_capture)
                ),
            },
        )
        try:
            delattr(self, "_dflash_tree_debug_context")
        except Exception:
            pass
        return meta

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> TreeAttentionMetadata:
        decode_threshold = self.tree_attn_bias.shape[0]
        num_decodes, num_prefills, num_decode_tokens, num_prefill_tokens = (
            split_decodes_and_prefills(
                common_attn_metadata, decode_threshold=decode_threshold
            )
        )

        num_actual_tokens = common_attn_metadata.num_actual_tokens
        q_start_loc = common_attn_metadata.query_start_loc
        max_query_len = common_attn_metadata.max_query_len
        kv_seqlens = common_attn_metadata.seq_lens
        max_seq_len = common_attn_metadata.max_seq_len
        block_table = common_attn_metadata.block_table_tensor
        slot_mapping = common_attn_metadata.slot_mapping

        return TreeAttentionMetadata(
            num_actual_tokens=num_actual_tokens,
            num_prefill_tokens=num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            num_prefills=num_prefills,
            num_decodes=num_decodes,
            max_query_len=max_query_len,
            query_start_loc=q_start_loc,
            max_seq_len=max_seq_len,
            seq_lens=kv_seqlens,
            block_table=block_table,
            slot_mapping=slot_mapping,
            tree_attn_bias=self.tree_attn_bias,
        )

    def build_for_drafting(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        draft_index: int,
    ) -> TreeAttentionMetadata:
        debug_context = getattr(self, "_dflash_tree_debug_context", {}) or {}
        # Cache the original tree attention bias.
        orig_tree_attn_bias = self.tree_attn_bias

        if draft_index == 0:
            # Use prefill for drafting at the root level.
            self.tree_attn_bias = torch.empty(0)
        else:
            # Slice the tree attention bias for drafting. Exclude
            # the root level.
            start, end = 1, 1 + common_attn_metadata.max_query_len
            self.tree_attn_bias = self.tree_attn_bias[start:end, start:end].contiguous()

        # Build attention bias.
        attn_metadata = self.build(0, common_attn_metadata, fast_build=True)

        # Reset the tree attention bias to the original value.
        self.tree_attn_bias = orig_tree_attn_bias
        self._append_dflash_tree_debug_record(
            build_method="build_for_drafting",
            common_attn_metadata=common_attn_metadata,
            output_metadata=attn_metadata,
            tree_attn_bias=self.tree_attn_bias,
            ancestor_masks=None,
            debug_context={
                **debug_context,
                "draft_index": int(draft_index),
            },
        )
        try:
            delattr(self, "_dflash_tree_debug_context")
        except Exception:
            pass
        return attn_metadata


def _get_depth_counts(sorted_tree_choices: list[tuple[int, ...]]) -> list[int]:
    # Count the number of choices at each depth of the tree.
    depth_counts = []
    prev_depth = 0
    for path in sorted_tree_choices:
        depth = len(path)
        if depth != prev_depth:
            depth_counts.append(0)
        depth_counts[depth - 1] += 1
        prev_depth = depth
    return depth_counts


def _prepare_tree_attn_bias(
    sorted_tree_choices: list[tuple[int, ...]],
    depth_counts: list[int],
    dtype: torch.dtype | None,
    device: torch.device | None,
) -> torch.Tensor:
    # +1 comes from the additional root node.
    tree_len = len(sorted_tree_choices) + 1
    tree_attn_mask = torch.full(
        (tree_len, tree_len), -torch.inf, device=device, dtype=dtype
    )

    # Set diagonal to all zeros. Each token should
    # attend to itself.
    mask_val = 0
    for i in range(tree_len):
        tree_attn_mask[i, i] = mask_val

    # Set root to all zeros. All tokens attend to it.
    tree_attn_mask[:, 0] = mask_val

    # Set all ancestors to zeros.
    start = 0
    for i in range(len(depth_counts)):
        for j in range(depth_counts[i]):
            cur_tree_choice = sorted_tree_choices[start + j]
            # Retrieve ancestor position.
            if len(cur_tree_choice) == 1:
                continue
            ancestor_idx = []
            for c in range(len(cur_tree_choice) - 1):
                ancestor_idx.append(
                    sorted_tree_choices.index(cur_tree_choice[: c + 1]) + 1
                )
            tree_attn_mask[j + start + 1, ancestor_idx] = mask_val
        start += depth_counts[i]
    return tree_attn_mask


class TreeAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if logits_soft_cap is None:
            # Setting logits_soft_cap to 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "TreeAttentionImpl."
            )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        key_cache, value_cache = kv_cache.unbind(0)

        # Reshape the input keys and values and store them in the cache.
        # NOTE(woosuk): Here, key and value are padded while slot_mapping is
        # not padded. However, we don't need to do key[:num_actual_tokens]
        # and value[:num_actual_tokens] because the reshape_and_cache_flash
        # op uses the slot_mapping's shape to determine the number of
        # actual tokens.
        ops.reshape_and_cache_flash(
            key,
            value,
            key_cache,
            value_cache,
            slot_mapping,
            self.kv_cache_dtype,
            layer._k_scale,
            layer._v_scale,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TreeAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with TreeAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            kv_cache: shape =
                [2, num_blocks, block_size, num_kv_heads, head_size]
            attn_metadata: Metadata for attention.
        Returns:
            shape = [num_tokens, num_heads * head_size]
        """
        assert output is not None, "Output tensor must be provided."

        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization is not yet supported for TreeAttentionImpl"
            )

        if attn_metadata is None:
            # Profiling run.
            return output.fill_(0)

        key_cache, value_cache = kv_cache.unbind(0)

        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, key.shape[1])
        if prefill_meta := attn_metadata.prefill_metadata:
            unified_attention(
                q=query[num_decode_tokens:num_actual_tokens],
                k=key_cache,
                v=value_cache,
                out=output[num_decode_tokens:num_actual_tokens],
                cu_seqlens_q=prefill_meta.query_start_loc,
                max_seqlen_q=prefill_meta.max_query_len,
                seqused_k=prefill_meta.seq_lens,
                max_seqlen_k=prefill_meta.max_seq_len,
                softmax_scale=self.scale,
                causal=True,
                alibi_slopes=self.alibi_slopes,
                window_size=self.sliding_window,
                block_table=prefill_meta.block_table,
                softcap=self.logits_soft_cap,
                q_descale=None,  # Not supported
                k_descale=layer._k_scale.expand(descale_shape),
                v_descale=layer._v_scale.expand(descale_shape),
            )

        if decode_meta := attn_metadata.decode_metadata:
            if decode_meta.ancestor_masks is not None:
                try:
                    from optimus_cutedsl.flash_attn import (
                        flash_attn_varlen_tree_paged_sm90,
                    )
                except ModuleNotFoundError as exc:
                    raise ModuleNotFoundError(
                        "tree_attn_kernel='optimus' requires the optimus_cutedsl "
                        "package. Set PYTHONPATH to include the optimus src dir, "
                        "e.g. PYTHONPATH=/home/i-hulanxiang/workspace/"
                        "optimus_jit_local/src:$PYTHONPATH"
                    ) from exc
                flash_attn_varlen_tree_paged_sm90(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    tree_mask=decode_meta.ancestor_masks,
                    cu_seqlens_q=decode_meta.query_start_loc.to(torch.int32),
                    seqused_k=decode_meta.seq_lens.to(torch.int32),
                    page_table=decode_meta.block_table.to(torch.int32),
                    softmax_scale=self.scale,
                    out=output[:num_decode_tokens],
                )
            else:
                unified_attention(
                    q=query[:num_decode_tokens],
                    k=key_cache,
                    v=value_cache,
                    out=output[:num_decode_tokens],
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_query_len,
                    seqused_k=decode_meta.seq_lens,
                    max_seqlen_k=decode_meta.max_seq_len,
                    softmax_scale=self.scale,
                    causal=True,
                    alibi_slopes=self.alibi_slopes,
                    qq_bias=decode_meta.tree_attn_bias,
                    window_size=self.sliding_window,
                    block_table=decode_meta.block_table,
                    softcap=self.logits_soft_cap,
                    q_descale=None,  # Not supported
                    k_descale=layer._k_scale.expand(descale_shape),
                    v_descale=layer._v_scale.expand(descale_shape),
                )
        return output
