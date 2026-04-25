# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import heapq
from dataclasses import dataclass

import numpy as np
import torch


def compute_per_depth_entropy(logits: torch.Tensor) -> list[float]:
    """Compute entropy of each depth's distribution.

    *logits* has shape ``(depth, vocab)`` (or ``(batch, depth, vocab)``
    for batched usage — only the last two dims are used here).
    Returns a Python list of length ``depth``.
    """
    probs = torch.softmax(logits, dim=-1)
    ent = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)
    return ent.tolist()


def _node_expand_score(
    cum_lp: float,
    depth: int,
    score_mode: str,
    per_depth_entropy: list[float] | None,
    hybrid_alpha: float,
    parent_expansion_entropy: float | None = None,
) -> float:
    """Score a node for heap-based expansion.

    Higher returned value = higher expansion priority (the heap stores
    negated scores so that ``heapq`` pops the best first).

    For entropy/hybrid modes the relevant entropy is the distribution that
    *produced* this node (its parent's expansion), not the distribution at
    the node's own depth.  When ``parent_expansion_entropy`` is provided it
    is used directly; otherwise we fall back to ``per_depth_entropy[depth]``
    which is correct for root / spine nodes that were not created by heap
    expansion.
    """
    if score_mode == "entropy":
        if parent_expansion_entropy is not None:
            return parent_expansion_entropy
        if per_depth_entropy is not None and depth < len(per_depth_entropy):
            return per_depth_entropy[depth]
        return 0.0
    if score_mode == "hybrid":
        if parent_expansion_entropy is not None:
            ent = parent_expansion_entropy
        elif per_depth_entropy is not None and depth < len(per_depth_entropy):
            ent = per_depth_entropy[depth]
        else:
            ent = 0.0
        return cum_lp + hybrid_alpha * ent
    # "accum_logp" (default)
    return cum_lp


@dataclass
class DraftTree:
    """GPU-resident tree representation (legacy interface).

    Prefer ``DraftTreeCPU`` for the proposer hot-path to avoid
    GPU<->CPU round-trips.  This class is still used by the verifier
    and tests that pass GPU tensors.
    """
    # Root-inclusive BFS layout.
    token_ids: torch.Tensor
    parent_indices: torch.Tensor
    depth: torch.Tensor
    num_nodes: int

    def paths(self) -> list[list[int]]:
        parents = self.parent_indices.tolist()
        children: list[list[int]] = [[] for _ in range(self.num_nodes)]
        for idx in range(1, self.num_nodes):
            children[parents[idx]].append(idx)
        leaves = [idx for idx in range(self.num_nodes) if not children[idx]]
        paths: list[list[int]] = []
        for leaf in leaves:
            path: list[int] = []
            node = leaf
            while node >= 0:
                path.append(node)
                node = parents[node]
            paths.append(path[::-1])
        return paths

    def longest_path(self) -> list[int]:
        if self.num_nodes == 0:
            return []
        if self.num_nodes == 1:
            return [0]

        parents = self.parent_indices.tolist()
        depths = self.depth.tolist()

        child_count = [0] * self.num_nodes
        for idx in range(1, self.num_nodes):
            child_count[parents[idx]] += 1

        best_leaf = -1
        best_depth = -1
        for idx in range(self.num_nodes):
            if child_count[idx] == 0 and depths[idx] > best_depth:
                best_depth = depths[idx]
                best_leaf = idx

        if best_leaf < 0:
            return [0]

        path: list[int] = []
        node = best_leaf
        while node >= 0:
            path.append(node)
            node = parents[node]
        return path[::-1]


@dataclass
class DraftTreeCPU:
    """CPU-only tree used during the proposer hot-path.

    All fields are plain Python lists — no GPU tensors are created until
    the tree is finalised via :meth:`to_gpu`.
    """
    token_ids: list[int]
    parent_indices: list[int]
    depths: list[int]
    num_nodes: int

    def longest_path(self) -> list[int]:
        if self.num_nodes == 0:
            return []
        if self.num_nodes == 1:
            return [0]

        parents = self.parent_indices
        depths = self.depths

        child_count = [0] * self.num_nodes
        for idx in range(1, self.num_nodes):
            child_count[parents[idx]] += 1

        best_leaf = -1
        best_depth = -1
        for idx in range(self.num_nodes):
            if child_count[idx] == 0 and depths[idx] > best_depth:
                best_depth = depths[idx]
                best_leaf = idx

        if best_leaf < 0:
            return [0]

        path: list[int] = []
        node = best_leaf
        while node >= 0:
            path.append(node)
            node = parents[node]
        return path[::-1]

    def to_gpu(self, device: torch.device) -> DraftTree:
        return DraftTree(
            token_ids=torch.tensor(self.token_ids, dtype=torch.long,
                                   device=device),
            parent_indices=torch.tensor(self.parent_indices, dtype=torch.long,
                                        device=device),
            depth=torch.tensor(self.depths, dtype=torch.long, device=device),
            num_nodes=self.num_nodes,
        )


def compute_tree_budget(
    block_size: int,
    tree_width: int,
    max_budget: int | None = None,
) -> int:
    if tree_width <= 1:
        return block_size
    full_tree = (tree_width**block_size - 1) // (tree_width - 1)
    if max_budget is not None and max_budget > 0:
        return min(full_tree, max_budget)
    return full_tree


def sample_topk_from_logits(
    logits: torch.Tensor,
    k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs = torch.log_softmax(logits, dim=-1)
    return torch.topk(log_probs, k, dim=-1)


def batch_topk_to_cpu(
    logits: torch.Tensor,
    k: int,
) -> tuple[list, list, list]:
    """Batched log_softmax + topk on GPU, then one `.tolist()` each.

    *logits* may be 2-D ``(depth, vocab)`` or 3-D ``(batch, depth, vocab)``.

    Returns ``(full_lp_cpu, topk_lp_cpu, topk_tok_cpu)`` where each is a
    nested Python list (outer dim = batch when 3-D).
    """
    lp = torch.log_softmax(logits, dim=-1)
    topk_lp, topk_tok = torch.topk(lp, k, dim=-1)
    return lp.tolist(), topk_lp.tolist(), topk_tok.tolist()


def _build_tree_breadth_first(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    score_mode: str = "accum_logp",
    per_depth_entropy: list[float] | None = None,
    hybrid_alpha: float = 1.0,
) -> DraftTreeCPU:
    """Heap expansion (breadth-biased) using the chosen scoring strategy.

    With large width and limited budget the tree may not reach full depth.
    """
    depth_count = len(topk_tokens_cpu)
    width = len(topk_tokens_cpu[0]) if depth_count else 0

    tokens_list = [root_token]
    parents_list = [-1]
    depths_list = [0]
    num_nodes = 1

    root_score = _node_expand_score(
        0.0, 0, score_mode, per_depth_entropy, hybrid_alpha,
    )
    counter = 0
    heap: list[tuple[float, int, int]] = [(-root_score, counter, 0)]
    cum_lp_at: list[float] = [0.0]
    while heap and num_nodes < budget:
        _, _, node_idx = heapq.heappop(heap)
        depth = depths_list[node_idx]
        if depth >= depth_count:
            continue
        children_to_add = min(width, budget - num_nodes)
        row_tokens = topk_tokens_cpu[depth]
        row_logprobs = topk_logprobs_cpu[depth]
        expansion_ent = (
            per_depth_entropy[depth]
            if per_depth_entropy is not None and depth < len(per_depth_entropy)
            else None
        )
        for child_idx in range(children_to_add):
            tokens_list.append(row_tokens[child_idx])
            child_cum_lp = cum_lp_at[node_idx] + row_logprobs[child_idx]
            parents_list.append(node_idx)
            child_depth = depth + 1
            depths_list.append(child_depth)
            cum_lp_at.append(child_cum_lp)
            score = _node_expand_score(
                child_cum_lp, child_depth, score_mode,
                per_depth_entropy, hybrid_alpha,
                parent_expansion_entropy=expansion_ent,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, num_nodes))
            num_nodes += 1

    return DraftTreeCPU(
        token_ids=tokens_list,
        parent_indices=parents_list,
        depths=depths_list,
        num_nodes=num_nodes,
    )


def _build_tree_depth_first(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    score_mode: str = "accum_logp",
    per_depth_entropy: list[float] | None = None,
    hybrid_alpha: float = 1.0,
) -> DraftTreeCPU:
    """Depth-first tree construction that guarantees the greedy spine.

    Phase 1: pre-allocate the top-1 (greedy) chain from root to full
    depth, ensuring tree acceptance >= linear-chain acceptance.

    Phase 2: spend remaining budget on side branches via heap expansion
    using the chosen *score_mode*, skipping the top-1 child for spine
    nodes (already present).
    """
    depth_count = len(topk_tokens_cpu)
    width = len(topk_tokens_cpu[0]) if depth_count else 0

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    num_nodes = 1

    spine_set: set[int] = {0}
    spine_cum_lp = 0.0
    prev_idx = 0
    for d in range(depth_count):
        if num_nodes >= budget:
            break
        tokens_list.append(topk_tokens_cpu[d][0])
        spine_cum_lp += topk_logprobs_cpu[d][0]
        parents_list.append(prev_idx)
        depths_list.append(d + 1)
        spine_set.add(num_nodes)
        prev_idx = num_nodes
        num_nodes += 1

    counter = 0
    heap: list[tuple[float, int, int]] = []
    cum_lp_at: list[float] = [0.0] * num_nodes
    for idx in range(num_nodes):
        d = depths_list[idx]
        if d > 0:
            cum_lp_at[idx] = (
                cum_lp_at[parents_list[idx]] + topk_logprobs_cpu[d - 1][0]
            )
        if d < depth_count:
            score = _node_expand_score(
                cum_lp_at[idx], d, score_mode,
                per_depth_entropy, hybrid_alpha,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, idx))

    while heap and num_nodes < budget:
        _, _, node_idx = heapq.heappop(heap)
        depth = depths_list[node_idx]
        if depth >= depth_count:
            continue
        start_child = 1 if node_idx in spine_set else 0
        children_to_add = min(width - start_child, budget - num_nodes)
        if children_to_add <= 0:
            continue
        row_tokens = topk_tokens_cpu[depth]
        row_logprobs = topk_logprobs_cpu[depth]
        expansion_ent = (
            per_depth_entropy[depth]
            if per_depth_entropy is not None and depth < len(per_depth_entropy)
            else None
        )
        for child_idx in range(start_child, start_child + children_to_add):
            tokens_list.append(row_tokens[child_idx])
            child_cum_lp = cum_lp_at[node_idx] + row_logprobs[child_idx]
            parents_list.append(node_idx)
            child_depth = depth + 1
            depths_list.append(child_depth)
            cum_lp_at.append(child_cum_lp)
            score = _node_expand_score(
                child_cum_lp, child_depth, score_mode,
                per_depth_entropy, hybrid_alpha,
                parent_expansion_entropy=expansion_ent,
            )
            counter += 1
            heapq.heappush(heap, (-score, counter, num_nodes))
            num_nodes += 1

    return DraftTreeCPU(
        token_ids=tokens_list,
        parent_indices=parents_list,
        depths=depths_list,
        num_nodes=num_nodes,
    )


def _build_tree_opt_prefix(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
) -> DraftTreeCPU:
    """Provably optimal tree under factorized draft marginals (DDTree / OPT-Tree).

    Each heap entry is a *rank tuple* representing a root-to-node path.
    Popping always yields the globally highest prefix-probability node.
    Two successors are pushed per pop:
      - next sibling  (same depth, next-best token)
      - first child   (one depth deeper, best token)
    This adds exactly one node per pop and produces the top-B prefixes
    without enumerating the exponential prefix space.
    """
    depth_count = len(topk_tokens_cpu)
    K = min(budget, len(topk_tokens_cpu[0])) if depth_count else 0
    if K == 0 or budget <= 1:
        return DraftTreeCPU(
            token_ids=[root_token],
            parent_indices=[-1],
            depths=[0],
            num_nodes=1,
        )

    tokens_list: list[int] = [root_token]
    parents_list: list[int] = [-1]
    depths_list: list[int] = [0]
    num_nodes = 1

    # Map rank-tuples to their node index so we can look up parent indices.
    # Key: tuple of ranks (length = depth), Value: node index.
    rank_to_idx: dict[tuple[int, ...], int] = {(): 0}

    counter = 0
    # Heap entries: (-score, counter, rank_tuple)
    # Start with rank (0,) = best token at depth 0
    init_score = topk_logprobs_cpu[0][0]
    heap: list[tuple[float, int, tuple[int, ...]]] = [
        (-init_score, counter, (0,))
    ]

    while heap and num_nodes < budget:
        neg_score, _, ranks = heapq.heappop(heap)
        score = -neg_score
        d = len(ranks)  # 1-based depth of this node

        parent_ranks = ranks[:-1]
        parent_idx = rank_to_idx[parent_ranks]
        rank_at_d = ranks[-1]

        tokens_list.append(topk_tokens_cpu[d - 1][rank_at_d])
        parents_list.append(parent_idx)
        depths_list.append(d)
        node_idx = num_nodes
        rank_to_idx[ranks] = node_idx
        num_nodes += 1

        # Push next sibling: same parent, next-ranked token at this depth
        next_rank = rank_at_d + 1
        if next_rank < K:
            sib_score = (score
                         - topk_logprobs_cpu[d - 1][rank_at_d]
                         + topk_logprobs_cpu[d - 1][next_rank])
            sib_ranks = parent_ranks + (next_rank,)
            counter += 1
            heapq.heappush(heap, (-sib_score, counter, sib_ranks))

        # Push first child: extend path with best token at next depth
        if d < depth_count:
            child_score = score + topk_logprobs_cpu[d][0]
            child_ranks = ranks + (0,)
            counter += 1
            heapq.heappush(heap, (-child_score, counter, child_ranks))

    return DraftTreeCPU(
        token_ids=tokens_list,
        parent_indices=parents_list,
        depths=depths_list,
        num_nodes=num_nodes,
    )


def _build_tree_cpu(
    root_token: int,
    topk_tokens_cpu: list[list[int]],
    topk_logprobs_cpu: list[list[float]],
    budget: int,
    depth_first: bool = True,
    score_mode: str = "accum_logp",
    per_depth_entropy: list[float] | None = None,
    hybrid_alpha: float = 1.0,
) -> DraftTreeCPU:
    if score_mode == "opt_prefix":
        return _build_tree_opt_prefix(
            root_token, topk_tokens_cpu, topk_logprobs_cpu, budget,
        )
    builder = _build_tree_depth_first if depth_first else _build_tree_breadth_first
    return builder(
        root_token, topk_tokens_cpu, topk_logprobs_cpu, budget,
        score_mode=score_mode,
        per_depth_entropy=per_depth_entropy,
        hybrid_alpha=hybrid_alpha,
    )


def build_tree_from_topk(
    root_token: int,
    topk_tokens: torch.Tensor,
    topk_logprobs: torch.Tensor,
    budget: int,
    device: torch.device,
    depth_first: bool = True,
    score_mode: str = "accum_logp",
    per_depth_entropy: list[float] | None = None,
    hybrid_alpha: float = 1.0,
) -> DraftTree:
    """GPU-tensor interface — wraps :func:`_build_tree_cpu`."""
    tree_cpu = _build_tree_cpu(
        root_token,
        topk_tokens.tolist(),
        topk_logprobs.tolist(),
        budget,
        depth_first=depth_first,
        score_mode=score_mode,
        per_depth_entropy=per_depth_entropy,
        hybrid_alpha=hybrid_alpha,
    )
    return tree_cpu.to_gpu(device)


def _prune_and_regrow_cpu(
    tree: DraftTreeCPU,
    node_logprobs: list[float],
    cond_topk_lp_cpu: list[list[float]],
    cond_topk_tok_cpu: list[list[int]],
    block_size: int,
    tree_width: int,
    budget: int,
    prune_ratio: float = 0.25,
) -> DraftTreeCPU:
    """Prune lowest-scoring leaves and regrow — pure CPU, zero GPU ops.

    *node_logprobs* contains the per-node conditional log-prob
    (``lp[depth-1][token_id]``) already gathered on GPU.
    ``node_logprobs[0]`` must be 0.0 (root).
    *prune_ratio* controls the fraction of leaves to prune (0.0–1.0).
    """
    max_depth = block_size - 1

    tokens = tree.token_ids
    parents = tree.parent_indices
    depths = tree.depths
    num_nodes = tree.num_nodes

    cum_lp: list[float] = [0.0] * num_nodes
    for i in range(1, num_nodes):
        cum_lp[i] = cum_lp[parents[i]] + node_logprobs[i]

    children_of: list[list[int]] = [[] for _ in range(num_nodes)]
    for i in range(1, num_nodes):
        children_of[parents[i]].append(i)

    leaves_with_score: list[tuple[float, int]] = []
    for i in range(num_nodes):
        if not children_of[i]:
            leaves_with_score.append((cum_lp[i], i))
    if not leaves_with_score:
        return tree
    leaves_with_score.sort()

    n_prune = max(1, int(len(leaves_with_score) * prune_ratio))
    pruned_set: set[int] = set()
    for _, leaf_idx in leaves_with_score[:n_prune]:
        node = leaf_idx
        if node <= 0:
            continue
        parent = parents[node]
        siblings_alive = sum(
            1 for c in children_of[parent] if c not in pruned_set
        )
        if siblings_alive > 1:
            pruned_set.add(node)

    kept = [i for i in range(num_nodes) if i not in pruned_set]
    if not kept:
        kept = [0]
    old_to_new = [-1] * num_nodes
    for new_i, old_i in enumerate(kept):
        old_to_new[old_i] = new_i

    new_tokens = [tokens[i] for i in kept]
    new_parents = [old_to_new[parents[i]] if i > 0 else -1 for i in kept]
    new_depths = [depths[i] for i in kept]
    new_cum_lp = [cum_lp[i] for i in kept]
    num_nodes = len(kept)
    freed = budget - num_nodes

    if freed > 0:
        new_children_of: list[list[int]] = [[] for _ in range(num_nodes)]
        for i in range(1, num_nodes):
            new_children_of[new_parents[i]].append(i)

        counter = 0
        regrow_heap: list[tuple[float, int, int]] = []
        for i in range(num_nodes):
            if not new_children_of[i] and new_depths[i] < max_depth:
                counter += 1
                heapq.heappush(regrow_heap, (-new_cum_lp[i], counter, i))

        while regrow_heap and freed > 0:
            _, _, node_idx = heapq.heappop(regrow_heap)
            depth_idx = new_depths[node_idx]
            if depth_idx >= max_depth:
                continue
            children_to_add = min(tree_width, freed)
            row_tokens = cond_topk_tok_cpu[depth_idx]
            row_logprobs = cond_topk_lp_cpu[depth_idx]
            for child_idx in range(children_to_add):
                child_cum = new_cum_lp[node_idx] + row_logprobs[child_idx]
                new_tokens.append(row_tokens[child_idx])
                new_parents.append(node_idx)
                new_depths.append(depth_idx + 1)
                new_cum_lp.append(child_cum)
                counter += 1
                heapq.heappush(regrow_heap, (-child_cum, counter, num_nodes))
                num_nodes += 1
                freed -= 1

    return DraftTreeCPU(
        token_ids=new_tokens,
        parent_indices=new_parents,
        depths=new_depths,
        num_nodes=num_nodes,
    )


def _gather_node_logprobs(
    lp: torch.Tensor,
    tree: DraftTree,
) -> list[float]:
    """Gather per-node conditional log-probs on GPU, return as CPU list.

    ``lp`` has shape ``(depth, vocab)``.  For each non-root node *i* in
    *tree*, the result contains ``lp[depth[i]-1, token_id[i]]``.
    Index 0 (root) is always 0.0.
    """
    n = tree.num_nodes
    if n <= 1:
        return [0.0] * n
    depths = tree.depth[1:]
    tokens = tree.token_ids[1:]
    gathered = lp[depths - 1, tokens]
    return [0.0] + gathered.tolist()


def prune_and_regrow(
    tree: DraftTree,
    cond_logits: torch.Tensor,
    block_size: int,
    tree_width: int,
    budget: int,
    device: torch.device | None = None,
    prune_ratio: float = 0.25,
) -> DraftTree:
    """Prune lowest-scoring leaves and regrow using conditioned logits."""
    if device is None:
        device = tree.token_ids.device
    lp = torch.log_softmax(cond_logits, dim=-1)
    topk_lp, topk_tok = torch.topk(lp, tree_width, dim=-1)

    node_logprobs = _gather_node_logprobs(lp, tree)
    topk_lp_cpu = topk_lp.tolist()
    topk_tok_cpu = topk_tok.tolist()
    tree_cpu = DraftTreeCPU(
        token_ids=tree.token_ids.tolist(),
        parent_indices=tree.parent_indices.tolist(),
        depths=tree.depth.tolist(),
        num_nodes=tree.num_nodes,
    )
    result = _prune_and_regrow_cpu(
        tree_cpu, node_logprobs, topk_lp_cpu, topk_tok_cpu,
        block_size, tree_width, budget, prune_ratio=prune_ratio,
    )
    return result.to_gpu(device)


def _adjust_tree_to_size_cpu(
    tree: DraftTreeCPU,
    target_size: int,
    node_logprobs: list[float] | None,
    cond_topk_lp_cpu: list[list[float]] | None,
    cond_topk_tok_cpu: list[list[int]] | None,
    block_size: int,
    tree_width: int,
) -> DraftTreeCPU:
    """Grow or shrink *tree* to ``target_size`` — pure CPU, zero GPU ops.

    *node_logprobs* contains the per-node conditional log-prob already
    gathered on GPU.  ``node_logprobs[0]`` must be 0.0 (root).
    """
    if tree.num_nodes == target_size or target_size <= 0:
        return tree

    max_depth = block_size - 1
    tokens = list(tree.token_ids)
    parents = list(tree.parent_indices)
    depths = list(tree.depths)
    num_nodes = tree.num_nodes

    cum_lp: list[float] = [0.0] * num_nodes
    if node_logprobs is not None:
        for i in range(1, num_nodes):
            cum_lp[i] = cum_lp[parents[i]] + node_logprobs[i]

    if num_nodes > target_size:
        children_of: list[list[int]] = [[] for _ in range(num_nodes)]
        for i in range(1, num_nodes):
            children_of[parents[i]].append(i)

        alive = [True] * num_nodes
        leaf_heap: list[tuple[float, int]] = []
        for i in range(num_nodes):
            if not children_of[i]:
                heapq.heappush(leaf_heap, (cum_lp[i], i))

        while num_nodes > target_size and leaf_heap:
            _, idx = heapq.heappop(leaf_heap)
            if not alive[idx]:
                continue
            if idx == 0:
                break
            alive[idx] = False
            num_nodes -= 1
            p = parents[idx]
            children_of[p] = [c for c in children_of[p] if alive[c]]
            if not children_of[p]:
                heapq.heappush(leaf_heap, (cum_lp[p], p))

        kept = [i for i in range(len(alive)) if alive[i]]
        old_to_new = [-1] * len(alive)
        for new_i, old_i in enumerate(kept):
            old_to_new[old_i] = new_i
        tokens = [tokens[i] for i in kept]
        parents = [old_to_new[parents[i]] if i > 0 else -1 for i in kept]
        depths = [depths[i] for i in kept]
        cum_lp = [cum_lp[i] for i in kept]
        num_nodes = len(kept)

    if num_nodes < target_size and cond_topk_tok_cpu is not None:
        children_of_g: list[list[int]] = [[] for _ in range(num_nodes)]
        for i in range(1, num_nodes):
            children_of_g[parents[i]].append(i)

        counter = 0
        grow_heap: list[tuple[float, int, int]] = []
        for i in range(num_nodes):
            if not children_of_g[i] and depths[i] < max_depth:
                counter += 1
                heapq.heappush(grow_heap, (-cum_lp[i], counter, i))

        freed = target_size - num_nodes
        while grow_heap and freed > 0:
            _, _, node_idx = heapq.heappop(grow_heap)
            depth_idx = depths[node_idx]
            if depth_idx >= max_depth:
                continue
            children_to_add = min(tree_width, freed)
            assert cond_topk_lp_cpu is not None
            row_tokens = cond_topk_tok_cpu[depth_idx]
            row_logprobs = cond_topk_lp_cpu[depth_idx]
            for child_idx in range(children_to_add):
                child_cum = cum_lp[node_idx] + row_logprobs[child_idx]
                tokens.append(row_tokens[child_idx])
                parents.append(node_idx)
                depths.append(depth_idx + 1)
                cum_lp.append(child_cum)
                counter += 1
                heapq.heappush(grow_heap, (-child_cum, counter, num_nodes))
                num_nodes += 1
                freed -= 1

    return DraftTreeCPU(
        token_ids=tokens,
        parent_indices=parents,
        depths=depths,
        num_nodes=num_nodes,
    )


def adjust_tree_to_size(
    tree: DraftTree,
    target_size: int,
    cond_logits: torch.Tensor | None,
    block_size: int,
    tree_width: int,
    device: torch.device | None = None,
) -> DraftTree:
    """Grow or shrink *tree* to *target_size* using conditioned logits."""
    if tree.num_nodes == target_size or target_size <= 0:
        return tree
    if device is None:
        device = tree.token_ids.device

    if cond_logits is not None:
        lp = torch.log_softmax(cond_logits, dim=-1)
        topk_lp, topk_tok = torch.topk(lp, tree_width, dim=-1)
        node_logprobs: list[float] | None = _gather_node_logprobs(lp, tree)
        topk_lp_cpu: list[list[float]] | None = topk_lp.tolist()
        topk_tok_cpu: list[list[int]] | None = topk_tok.tolist()
    else:
        node_logprobs = topk_lp_cpu = topk_tok_cpu = None

    tree_cpu = DraftTreeCPU(
        token_ids=tree.token_ids.tolist(),
        parent_indices=tree.parent_indices.tolist(),
        depths=tree.depth.tolist(),
        num_nodes=tree.num_nodes,
    )
    result = _adjust_tree_to_size_cpu(
        tree_cpu, target_size, node_logprobs, topk_lp_cpu, topk_tok_cpu,
        block_size, tree_width,
    )
    return result.to_gpu(device)


def find_closest_capture_size(
    num_nodes: int,
    capture_sizes: list[int],
) -> int:
    """Return the smallest capture size >= *num_nodes*.

    Falls back to the largest capture size if *num_nodes* exceeds all of them.
    *capture_sizes* must be sorted ascending.
    """
    import bisect
    idx = bisect.bisect_left(capture_sizes, num_nodes)
    if idx < len(capture_sizes):
        return capture_sizes[idx]
    return capture_sizes[-1]


def tree_signature(tree: DraftTreeCPU | DraftTree) -> str:
    """One-line summary of tree shape: ``N=150 D=5 L=30 depth=[1:5,2:20,...]``."""
    if isinstance(tree, DraftTree):
        depths = tree.depth.tolist()
        parents = tree.parent_indices.tolist()
    else:
        depths = tree.depths
        parents = tree.parent_indices
    n = tree.num_nodes
    if n == 0:
        return "N=0"
    max_d = max(depths) if depths else 0
    depth_counts: dict[int, int] = {}
    num_leaves = 0
    child_count = [0] * n
    for i in range(1, n):
        child_count[parents[i]] += 1
    for i in range(n):
        d = depths[i]
        depth_counts[d] = depth_counts.get(d, 0) + 1
        if child_count[i] == 0:
            num_leaves += 1
    dist_str = ",".join(f"{d}:{c}" for d, c in sorted(depth_counts.items()))
    return f"N={n} D={max_d} L={num_leaves} depth=[{dist_str}]"


def build_tree_entropy_guided(
    root_token: int,
    draft_logits: torch.Tensor,
    block_size: int,
    tree_width: int,
    budget: int,
    cond_logits: torch.Tensor | None = None,
    device: torch.device | None = None,
    prune_ratio: float = 0.25,
    depth_first: bool = True,
) -> DraftTree:
    if device is None:
        device = draft_logits.device
    topk_logprobs, topk_tokens = sample_topk_from_logits(draft_logits, tree_width)
    tree = build_tree_from_topk(
        root_token=root_token,
        topk_tokens=topk_tokens,
        topk_logprobs=topk_logprobs,
        budget=budget,
        device=device,
        depth_first=depth_first,
    )
    if cond_logits is not None:
        tree = prune_and_regrow(
            tree,
            cond_logits=cond_logits,
            block_size=block_size,
            tree_width=tree_width,
            budget=budget,
            device=device,
            prune_ratio=prune_ratio,
        )
    return tree


def _build_attention_bias_np(
    parents_list: list[int],
    neg_inf: float,
) -> np.ndarray:
    """Build tree attention bias as a numpy array on CPU."""
    query_len = len(parents_list)
    mask = np.zeros((query_len, query_len), dtype=np.bool_)
    for i in range(query_len):
        mask[i, i] = True
    mask[:, 0] = True
    for node_idx in range(1, query_len):
        parent = parents_list[node_idx]
        while parent > 0:
            mask[node_idx, parent] = True
            parent = parents_list[parent]

    bias_np = np.full((query_len, query_len), neg_inf, dtype=np.float32)
    bias_np[mask] = 0.0
    return bias_np


def _build_causal_bias_np(
    query_len: int,
    neg_inf: float,
) -> np.ndarray:
    """Build causal attention bias as a numpy array on CPU."""
    bias_np = np.full((query_len, query_len), neg_inf, dtype=np.float32)
    il = np.tril_indices(query_len)
    bias_np[il] = 0.0
    return bias_np


def build_ancestor_matrix_np(parents_list: list[int]) -> np.ndarray:
    """Build an (N, N) int32 ancestor matrix from parent indices on CPU.

    ancestor[i, j] == 1 iff j is on the root-to-i path (inclusive of i itself).
    This is the tree attention mask consumed by the optimus SM90 kernel.
    """
    N = len(parents_list)
    ancestor = np.eye(N, dtype=np.int32)
    for i in range(1, N):
        ancestor[i] |= ancestor[parents_list[i]]
    return ancestor


def build_causal_ancestor_matrix_np(query_len: int) -> np.ndarray:
    """Build a causal (lower-triangular) ancestor matrix for chain topology."""
    return np.tril(np.ones((query_len, query_len), dtype=np.int32))


def build_attention_bias_from_parents(
    parent_indices: torch.Tensor,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    parents_cpu = parent_indices.tolist()
    neg_inf = float(torch.finfo(dtype).min)
    bias_np = _build_attention_bias_np(parents_cpu, neg_inf)
    return torch.from_numpy(bias_np).to(dtype=dtype, device=device)


def build_causal_attention_bias(
    query_len: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    neg_inf = float(torch.finfo(dtype).min)
    bias_np = np.full((query_len, query_len), neg_inf, dtype=np.float32)
    il = np.tril_indices(query_len)
    bias_np[il] = 0.0
    return torch.from_numpy(bias_np).to(dtype=dtype, device=device)


def build_block_diagonal_attention_bias(
    biases: list[torch.Tensor],
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    total_query_len = sum(bias.shape[0] for bias in biases)
    if total_query_len == 0:
        return torch.empty((0, 0), dtype=dtype, device=device)
    neg_inf = float(torch.finfo(dtype).min)
    output_np = np.full((total_query_len, total_query_len), neg_inf,
                         dtype=np.float32)
    cursor = 0
    for bias in biases:
        qlen = bias.shape[0]
        if isinstance(bias, np.ndarray):
            output_np[cursor:cursor + qlen, cursor:cursor + qlen] = bias
        else:
            output_np[cursor:cursor + qlen, cursor:cursor + qlen] = (
                bias.cpu().numpy()
            )
        cursor += qlen
    return torch.from_numpy(output_np).to(dtype=dtype, device=device)


def gpu_tree_accept(
    tree_tokens: torch.Tensor,
    greedy_targets: torch.Tensor,
    parent_indices: torch.Tensor,
    depths: torch.Tensor,
    max_depth: int = 15,
) -> tuple[torch.Tensor, int, torch.Tensor]:
    """GPU-native greedy tree acceptance — no CPU round-trips.

    All matching and path extraction runs on GPU.  Only a single
    ``accepted_depth.item()`` sync is needed to slice the output path.

    Args:
        tree_tokens: [N] draft token ids (root + tree nodes).
        greedy_targets: [N] argmax of the *target* model logits (pre-computed).
        parent_indices: [N] parent node index per node (-1 for root).
        depths: [N] depth of each node (root = 0).
        max_depth: upper bound on tree depth (used for loop unrolling).

    Returns:
        accepted_path: 1-D tensor of node indices from root to deepest
            accepted node (length = accepted_len + 1).
        accepted_len: Python int — number of accepted *draft* tokens
            (excludes root).
        correction_token: 0-D tensor with the target-model greedy token
            at the last accepted position.
    """
    device = tree_tokens.device
    N = tree_tokens.shape[0]

    if N <= 1:
        return (
            torch.zeros(1, dtype=torch.long, device=device),
            0,
            greedy_targets[0],
        )

    safe_parents = parent_indices.clamp(min=0)

    # --- vectorised match: does each node agree with its parent's target? ---
    match = torch.ones(N, dtype=torch.bool, device=device)
    match[1:] = tree_tokens[1:] == greedy_targets[safe_parents[1:]]

    # --- prefix-match via parallel doubling (ceil-log2 iterations) ---
    prefix_match = match.clone()
    jump = safe_parents.clone()
    for _ in range(max(1, max_depth.bit_length())):
        anc_match = prefix_match[jump]
        prefix_match = prefix_match & anc_match
        jump = jump[jump]

    # --- deepest fully-accepted node ---
    score = torch.where(
        prefix_match,
        depths,
        torch.tensor(-1, device=device, dtype=depths.dtype),
    )
    best_node = torch.argmax(score)
    accepted_depth_t = depths[best_node]
    correction = greedy_targets[best_node]

    # --- extract root-to-best path (walk parent pointers on GPU) ---
    path_buf = torch.zeros(max_depth + 1, dtype=torch.long, device=device)
    current = best_node.unsqueeze(0)
    for d in range(max_depth, -1, -1):
        path_buf[d : d + 1] = current
        current = safe_parents[current]

    accepted_len = int(accepted_depth_t.item())  # single GPU→CPU sync
    valid_start = max_depth - accepted_len
    accepted_path = path_buf[valid_start : max_depth + 1].contiguous()

    return accepted_path, accepted_len, correction


def tree_accept(
    tree_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
    target_logits: torch.Tensor,
    temperature: float = 0.0,
) -> tuple[list[int], int, int]:
    """Reference-style tree acceptance for flattened vLLM tree tensors.

    Returns:
        accepted_path: root-inclusive accepted path indices.
        acceptance_length: accepted draft-token count (excludes root).
        correction_token: posterior token sampled at the last accepted node.
    """
    if temperature != 0.0:
        raise NotImplementedError(
            "Native DFlash tree acceptance currently supports greedy decoding "
            "only."
        )

    posterior_cpu = torch.argmax(target_logits, dim=-1).tolist()
    tokens_cpu = tree_tokens.tolist()
    parents_cpu = parent_indices.tolist()
    num_nodes = len(tokens_cpu)

    children: list[list[int]] = [[] for _ in range(num_nodes)]
    for node_idx in range(1, num_nodes):
        children[parents_cpu[node_idx]].append(node_idx)

    def _paths() -> list[list[int]]:
        leaves = [idx for idx in range(num_nodes) if not children[idx]]
        paths: list[list[int]] = []
        for leaf in leaves:
            path: list[int] = []
            node = leaf
            while node >= 0:
                path.append(node)
                node = parents_cpu[node]
            paths.append(path[::-1])
        return paths

    best_path = [0]
    best_len = 0
    for path in _paths():
        accepted = 0
        for depth_idx in range(1, len(path)):
            parent_node = path[depth_idx - 1]
            child_node = path[depth_idx]
            if tokens_cpu[child_node] == posterior_cpu[parent_node]:
                accepted += 1
            else:
                break
        if accepted > best_len:
            best_len = accepted
            best_path = path[: accepted + 1]

    correction_token = posterior_cpu[best_path[-1]]
    return best_path, best_len, correction_token


def tree_accept_greedy(
    tree_tokens: torch.Tensor,
    parent_indices: torch.Tensor,
    target_token_ids: torch.Tensor,
) -> tuple[list[int], int]:
    tokens_cpu = tree_tokens.tolist()
    parents_cpu = parent_indices.tolist()
    targets_cpu = target_token_ids.tolist()
    num_nodes = len(tokens_cpu)

    children: list[list[int]] = [[] for _ in range(num_nodes)]
    for node_idx in range(1, num_nodes):
        children[parents_cpu[node_idx]].append(node_idx)

    best_path = [0]
    best_len = 0
    stack: list[list[int]] = [[0]]
    while stack:
        path = stack.pop()
        leaf = path[-1]
        child_nodes = children[leaf]
        if not child_nodes:
            accepted = 0
            for depth_idx in range(1, len(path)):
                parent = path[depth_idx - 1]
                child = path[depth_idx]
                if tokens_cpu[child] == targets_cpu[parent]:
                    accepted += 1
                else:
                    break
            if accepted > best_len:
                best_len = accepted
                best_path = path[: accepted + 1]
            continue
        for child in reversed(child_nodes):
            stack.append([*path, child])
    return best_path, best_len
