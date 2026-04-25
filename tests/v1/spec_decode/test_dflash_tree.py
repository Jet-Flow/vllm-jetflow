# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import numpy as np
import torch

from vllm.config import CUDAGraphMode
from vllm.config.speculative import SpeculativeConfig
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionImpl,
    _get_sliding_window_configs,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum
from vllm.v1.attention.selector import get_attn_backend
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.spec_decode.dflash import DFlashProposer
from vllm.v1.spec_decode.dflash_tree import (
    build_tree_from_topk,
    tree_accept,
    tree_accept_greedy,
)
from vllm.v1.spec_decode.metrics import SpecDecodingStats
from vllm.v1.spec_decode.eagle import SpecDecodeBaseProposer
from vllm.v1.spec_decode.metadata import (
    DFlashRequestTreeSpec,
    DFlashTreeSpecDecodeMetadata,
)
from vllm.v1.worker.gpu_model_runner import GPUModelRunner


def test_dflash_tree_max_num_new_slots_uses_tree_budget():
    config = object.__new__(SpeculativeConfig)
    config.method = "dflash"
    config.tree_width = 7
    config.max_tree_budget = 255
    config.num_speculative_tokens = 15
    config.parallel_drafting = True

    assert config.max_num_new_slots_for_drafting == 254


def test_scheduler_marks_full_dflash_tree_requests_as_non_truncatable():
    request = SimpleNamespace(
        spec_tree_metadata=DFlashRequestTreeSpec(parent_indices=[0, 0], depths=[1, 1]),
        spec_token_ids=[11, 12],
    )

    assert Scheduler._is_full_dflash_tree_request(request)

    request.spec_token_ids = [11]

    assert not Scheduler._is_full_dflash_tree_request(request)


def test_build_tree_from_topk_respects_budget_depth_first():
    topk_tokens = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    topk_logprobs = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=torch.float32)

    tree = build_tree_from_topk(
        root_token=5,
        topk_tokens=topk_tokens,
        topk_logprobs=topk_logprobs,
        budget=5,
        device=torch.device("cpu"),
        depth_first=True,
    )
    assert tree.num_nodes == 5
    assert tree.token_ids[0].item() == 5
    assert tree.parent_indices[0].item() == -1
    # Spine: root(5) -> 7 -> 9 must exist as a full-depth chain
    assert tree.depth.max().item() == 2
    # All 4 draft tokens present
    tok_set = set(tree.token_ids.tolist())
    assert {7, 8, 9, 10}.issubset(tok_set)


def test_build_tree_from_topk_respects_budget_breadth_first():
    topk_tokens = torch.tensor([[7, 8], [9, 10]], dtype=torch.long)
    topk_logprobs = torch.tensor([[0.0, -1.0], [0.0, -1.0]], dtype=torch.float32)

    tree = build_tree_from_topk(
        root_token=5,
        topk_tokens=topk_tokens,
        topk_logprobs=topk_logprobs,
        budget=5,
        device=torch.device("cpu"),
        depth_first=False,
    )
    assert tree.token_ids.tolist() == [5, 7, 8, 9, 10]
    assert tree.parent_indices.tolist() == [-1, 0, 0, 1, 1]
    assert tree.depth.tolist() == [0, 1, 1, 2, 2]


def test_tree_accept_greedy_picks_longest_matching_branch():
    tree_tokens = torch.tensor([5, 7, 8, 9, 10], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 0, 1, 1], dtype=torch.int64)
    target_tokens = torch.tensor([7, 9, 0, 0, 0], dtype=torch.int64)

    accepted_path, accepted_len = tree_accept_greedy(
        tree_tokens,
        parent_indices,
        target_tokens,
    )

    assert accepted_path == [0, 1, 3]
    assert accepted_len == 2


def test_tree_accept_matches_reference_correction_token_semantics():
    tree_tokens = torch.tensor([5, 7, 8, 9, 10], dtype=torch.int64)
    parent_indices = torch.tensor([-1, 0, 0, 1, 1], dtype=torch.int64)
    logits = torch.full((5, 32), -1000.0, dtype=torch.float32)
    logits[0, 7] = 1.0
    logits[1, 9] = 1.0
    logits[3, 13] = 1.0

    accepted_path, accepted_len, correction_token = tree_accept(
        tree_tokens,
        parent_indices,
        logits,
    )

    assert accepted_path == [0, 1, 3]
    assert accepted_len == 2
    assert correction_token == 13


def test_spec_decoding_stats_tracks_tree_nodes():
    stats = SpecDecodingStats.new(15)

    stats.observe_draft(num_draft_tokens=254, num_accepted_tokens=7, tree_size=255)

    assert stats.num_tree_drafts == 1
    assert stats.num_tree_nodes == 255


def test_calc_dflash_tree_spec_decode_metadata_builds_bias():
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner._arange_scratch = np.empty(8, dtype=np.int32)
    runner.arange_np = np.arange(8, dtype=np.int32)
    runner.speculative_config = None
    runner.input_batch = SimpleNamespace(
        req_ids=["req-1", "req-2"],
        req_id_to_index={"req-1": 0, "req-2": 1},
    )
    runner.input_ids = SimpleNamespace(gpu=torch.tensor([5, 7, 8, 11], dtype=torch.int32))

    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tree_metadata={
            "req-1": DFlashRequestTreeSpec(parent_indices=[0, 0], depths=[1, 1]),
        }
    )

    metadata = runner._calc_dflash_tree_spec_decode_metadata(
        num_draft_tokens=np.array([2, 0], dtype=np.int32),
        cu_num_scheduled_tokens=np.array([3, 4], dtype=np.int32),
        scheduler_output=scheduler_output,
    )

    assert isinstance(metadata, DFlashTreeSpecDecodeMetadata)
    assert metadata.query_lens == [3, 1]
    assert metadata.parent_indices.tolist() == [-1, 0, 0, -1]
    assert metadata.depths.tolist() == [0, 1, 1, 0]
    assert metadata.tree_attn_bias.shape == (4, 4)
    assert metadata.is_tree_req == [True, False]


def test_dflash_proposer_tree_mode_returns_tree_specs(monkeypatch):
    monkeypatch.setattr(
        "vllm.v1.spec_decode.dflash.set_forward_context",
        lambda *args, **kwargs: nullcontext(),
    )

    proposer = object.__new__(DFlashProposer)
    proposer.speculative_config = SimpleNamespace(
        tree_width=2,
        max_tree_budget=5,
        tree_draft="accum_logp",
        tree_hybrid_alpha=1.0,
        max_draft_passes=0,
        tree_construction="depth_first",
        cudagraph_tree_capture_sizes=None,
    )
    proposer.num_speculative_tokens = 2
    proposer.dflash_is_causal = True
    proposer.device = torch.device("cpu")
    proposer.hidden_size = 4
    proposer.vllm_config = SimpleNamespace()
    proposer.input_ids = torch.tensor([5, 0, 0], dtype=torch.int32)
    proposer._last_tree_specs = None
    proposer._tree_propose_step = 0
    proposer._cg_hit_count = 0
    proposer._cg_miss_count = 0
    proposer._logged_capture_sizes = False
    proposer.model_returns_tuple = lambda: False
    proposer._get_slot_mapping = lambda *args, **kwargs: {}
    proposer._determine_batch_execution_and_padding = (
        lambda num_tokens: (CUDAGraphMode.NONE, num_tokens, num_tokens)
    )
    proposer.build_per_group_and_layer_attn_metadata = lambda cad: ({}, {})
    proposer.build_model_inputs_first_pass = (
        lambda num_tokens, num_input_tokens, mm: (
            {"input_ids": proposer.input_ids[:num_input_tokens].clone()},
            0,
        )
    )
    proposer.set_inputs_first_pass = (
        lambda **kwargs: (
            3,
            torch.tensor([0, 1], dtype=torch.int64),
            SimpleNamespace(
                batch_size=lambda: 1,
                slot_mapping=torch.empty(0, dtype=torch.int64),
            ),
        )
    )

    class DummyModel:
        def combine_hidden_states(self, hidden_states):
            return hidden_states

        def __call__(self, **kwargs):
            return torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [1.0, 1.0, 1.0, 1.0],
                ],
                dtype=torch.float32,
            )

        def compute_logits(self, sample_hidden_states):
            del sample_hidden_states
            return torch.tensor(
                [
                    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 6.0, 5.0, -10.0, -10.0],
                    [-10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, -10.0, 6.0, 5.0],
                ],
                dtype=torch.float32,
            )

    proposer.model = DummyModel()

    draft_token_ids = proposer.propose(
        target_token_ids=torch.tensor([5], dtype=torch.int32),
        target_positions=torch.tensor([0], dtype=torch.int64),
        target_hidden_states=torch.zeros((1, 4), dtype=torch.float32),
        next_token_ids=torch.tensor([5], dtype=torch.int32),
        token_indices_to_sample=None,
        common_attn_metadata=SimpleNamespace(batch_size=lambda: 1),
        sampling_metadata=SimpleNamespace(),
    )

    # depth_first: spine [7, 9] first, then side branches [8, 10]
    assert draft_token_ids == [[7, 9, 8, 10]]
    assert proposer.consume_tree_specs() == [
        DFlashRequestTreeSpec(parent_indices=[0, 1, 0, 1], depths=[1, 2, 1, 2])
    ]


def test_dflash_proposer_records_verify_outcome_in_topk_log():
    proposer = object.__new__(DFlashProposer)
    proposer._topk_log = [
        {
            "step": 0,
            "req": 0,
            "root_token": 5,
            "topk_tok_0": [7, 8, 9],
            "topk_lp_0": [0.0, -1.0, -2.0],
        }
    ]
    proposer._pending_topk_log_indices = [0]

    proposer.record_topk_verify_outcome(
        verify_greedy_tokens=torch.tensor([7, 10, 11], dtype=torch.int64),
        accepted_len=2,
        correction_token=torch.tensor(13, dtype=torch.int64),
        tree_num_nodes=5,
    )

    entry = proposer.get_topk_log()[0]
    assert entry["verify_greedy_tokens"] == [7, 10, 11]
    assert entry["target_next_token"] == 7
    assert entry["draft_top1_token"] == 7
    assert entry["draft_top1_match"] is True
    assert entry["target_in_topk"] is True
    assert entry["accepted_len"] == 2
    assert entry["accepted_d0"] is True
    assert entry["correction_token"] == 13
    assert entry["tree_num_nodes"] == 5
    assert proposer._pending_topk_log_indices == []


def test_selector_forces_tree_attention_for_dflash_tree(monkeypatch):
    captured = {}

    def fake_cached_get_attn_backend(*, backend, attn_selector_config, num_heads=None):
        del attn_selector_config, num_heads
        captured["backend"] = backend
        return backend

    monkeypatch.setattr(
        "vllm.v1.attention.selector._cached_get_attn_backend",
        fake_cached_get_attn_backend,
    )
    monkeypatch.setattr(
        "vllm.config.get_current_vllm_config",
        lambda: SimpleNamespace(
            cache_config=SimpleNamespace(
                user_specified_block_size=True,
                block_size=16,
            ),
            speculative_config=SimpleNamespace(method="dflash", tree_width=2),
            attention_config=SimpleNamespace(backend=None),
        ),
    )

    get_attn_backend(
        head_size=128,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
    )

    assert captured["backend"] == AttentionBackendEnum.TREE_ATTN


def test_flash_attn_sliding_window_scan_skips_non_flash_draft_layers(monkeypatch):
    flash_impl = object.__new__(FlashAttentionImpl)
    flash_impl.sliding_window = (128, 0)

    monkeypatch.setattr(
        "vllm.v1.attention.backends.flash_attn.get_layers_from_vllm_config",
        lambda vllm_config, layer_type: {
            "target_attn": SimpleNamespace(impl=flash_impl),
            "draft_attn": SimpleNamespace(impl=SimpleNamespace()),
        },
    )

    sliding_window_configs = _get_sliding_window_configs(SimpleNamespace())

    assert sliding_window_configs == {(128, 0)}


def test_dflash_tree_draft_config_uses_deepcopy(monkeypatch):
    base_vllm_config = SimpleNamespace(speculative_config=None)

    monkeypatch.setattr(
        SpecDecodeBaseProposer,
        "_create_draft_vllm_config",
        lambda self: base_vllm_config,
    )
    monkeypatch.setattr(
        "vllm.v1.spec_decode.dflash.replace",
        lambda cfg, speculative_config: SimpleNamespace(
            parent=cfg, speculative_config=speculative_config
        ),
    )

    proposer = object.__new__(DFlashProposer)
    proposer.speculative_config = SimpleNamespace(tree_width=7)

    draft_cfg = proposer._create_draft_vllm_config()

    assert proposer.speculative_config.tree_width == 7
    assert draft_cfg.speculative_config.tree_width == 1
    assert draft_cfg.speculative_config is not proposer.speculative_config


def test_compact_dflash_tree_kv_cache_moves_accepted_path_slots():
    runner = object.__new__(GPUModelRunner)
    runner._dflash_tree_accept_paths = [[0, 2, 4]]
    kv_cache = torch.zeros((2, 2, 4, 1, 1), dtype=torch.float32)
    key_cache, value_cache = kv_cache.unbind(0)
    for slot in range(5):
        block = slot // 4
        offset = slot % 4
        key_cache[block, offset, 0, 0] = float(slot)
        value_cache[block, offset, 0, 0] = float(slot + 10)
    runner.kv_caches = [kv_cache]

    runner._compact_dflash_tree_kv_cache(
        SimpleNamespace(is_tree_req=[True]),
        SimpleNamespace(
            slot_mapping=torch.tensor([0, 1, 2, 3, 4], dtype=torch.int64),
            query_start_loc=torch.tensor([0, 5], dtype=torch.int32),
        ),
    )

    assert key_cache[0, 0, 0, 0].item() == 0.0
    assert key_cache[0, 1, 0, 0].item() == 2.0
    assert key_cache[0, 2, 0, 0].item() == 4.0
    assert value_cache[0, 0, 0, 0].item() == 10.0
    assert value_cache[0, 1, 0, 0].item() == 12.0
    assert value_cache[0, 2, 0, 0].item() == 14.0


def test_compact_dflash_tree_hidden_states_keeps_metadata_aligned():
    runner = object.__new__(GPUModelRunner)
    runner._dflash_tree_accept_paths_gpu = [
        torch.tensor([0, 2, 4], dtype=torch.int64)
    ]
    runner.uses_mrope = False
    runner.uses_xdrope_dim = 0
    runner.input_ids = SimpleNamespace(
        gpu=torch.tensor([10, 11, 12, 13, 14], dtype=torch.int32)
    )
    runner.positions = torch.tensor([100, 101, 102, 103, 104], dtype=torch.int64)

    hidden_states = torch.tensor(
        [[0.0, 0.5], [1.0, 1.5], [2.0, 2.5], [3.0, 3.5], [4.0, 4.5]],
        dtype=torch.float32,
    )
    aux_hidden_states = [
        torch.tensor(
            [[10.0], [11.0], [12.0], [13.0], [14.0]], dtype=torch.float32
        )
    ]

    runner._compact_dflash_tree_hidden_states(
        SimpleNamespace(query_lens=[5], is_tree_req=[True]),
        hidden_states,
        aux_hidden_states,
    )

    assert runner.input_ids.gpu[:3].tolist() == [10, 12, 14]
    assert runner.positions[:3].tolist() == [100, 102, 104]
    assert hidden_states[:3].tolist() == [[0.0, 0.5], [2.0, 2.5], [4.0, 4.5]]
    assert aux_hidden_states[0][:3, 0].tolist() == [10.0, 12.0, 14.0]


def test_prepare_input_ids_accepts_variable_length_tree_drafts():
    runner = object.__new__(GPUModelRunner)
    runner.device = torch.device("cpu")
    runner.pin_memory = False
    runner.enable_prompt_embeds = False
    runner.num_spec_tokens = 4
    runner.prev_positions = SimpleNamespace(np=np.array([0, 1], dtype=np.int32))
    runner._draft_token_ids = [[11, 12], [21]]
    runner.input_batch = SimpleNamespace(
        req_ids=["req-1", "req-2"],
        prev_sampled_token_ids=torch.tensor([[10], [20]], dtype=torch.int32),
    )
    runner.input_ids = SimpleNamespace(
        gpu=torch.zeros(5, dtype=torch.int32),
        copy_to_gpu=lambda *args, **kwargs: None,
    )

    scheduler_output = SimpleNamespace(
        scheduled_spec_decode_tokens={
            "req-1": [11, 12],
            "req-2": [21],
        }
    )

    runner._prepare_input_ids(
        scheduler_output=scheduler_output,
        num_reqs=2,
        total_num_scheduled_tokens=5,
        cu_num_tokens=np.array([3, 5], dtype=np.int32),
    )

    assert runner.input_ids.gpu.tolist() == [10, 11, 12, 20, 21]
