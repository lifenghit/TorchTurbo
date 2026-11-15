# Adapted from the TorchTurbo MoE integration interfaces.
# Copyright (c) 2026, TorchTurbo contributors.

"""MegaMoE offline-DAG execution for TorchTurbo routed experts."""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from hyper_parallel.core.multicore import MegaMoeExperts


_MAX_LOCAL_NUM_TOKENS = 16384
_EXECUTORS = {}


def _local_weight(weight: torch.Tensor, shape: tuple[int, ...], hidden: torch.Tensor) -> torch.Tensor:
    """Preserve the autograd connection while unwrapping an FSDP DTensor."""
    if isinstance(weight, DTensor):
        return weight.view(shape).to(device=hidden.device, dtype=hidden.dtype).to_local()
    return weight.view(shape).to(device=hidden.device, dtype=hidden.dtype)


def _executor(
    hidden: torch.Tensor,
    expert_indices: torch.Tensor,
    experts: torch.nn.Module,
    group: dist.ProcessGroup,
) -> MegaMoeExperts:
    """Share static DAG resources across layers with identical activation semantics."""
    hidden_size = hidden.shape[1]
    intermediate_size = experts.config.moe_ffn_hidden_size
    top_k = expert_indices.shape[1]
    ep_size = dist.get_world_size(group)
    num_experts = experts.num_local_experts * ep_size
    swiglu_limit = experts.swiglu_limit if experts.swiglu_limit > 0 else None
    clamp_after = experts.config.moe_swiglu_clamp_gate_after_activation
    key = (
        id(group), hidden_size, intermediate_size, top_k, num_experts,
        swiglu_limit, clamp_after,
    )
    if not _EXECUTORS:
        limits = {swiglu_limit}
        for value in experts.config.moe_swiglu_limit or []:
            limits.add(float(value) if value > 0 else None)
        for limit in sorted(limits, key=lambda value: -1.0 if value is None else value):
            executor_key = (
                id(group), hidden_size, intermediate_size, top_k, num_experts,
                limit, clamp_after,
            )
            _EXECUTORS[executor_key] = MegaMoeExperts(
                max_local_num_tokens=_MAX_LOCAL_NUM_TOKENS,
                hidden_size=hidden_size,
                intermediate_size=intermediate_size,
                num_experts=num_experts,
                top_k=top_k,
                swiglu_limit=limit,
                swiglu_clamp_gate_after_activation=clamp_after,
                ep_size=ep_size,
                ep_group=group,
                create_parameters=False,
                dispatch_mode="pull",
                dispatch_split=512,
                combine_split=512,
            )
    return _EXECUTORS[key]


def fused_reg_moe(
    hidden_states: torch.Tensor,
    expert_indices: torch.Tensor,
    probs: torch.Tensor,
    experts: torch.nn.Module,
    group: dist.ProcessGroup,
    _config: object,
) -> torch.Tensor:
    """Execute dispatch, expert MLP and combine through a static device-side DAG."""
    if hidden_states.shape[0] > _MAX_LOCAL_NUM_TOKENS:
        raise ValueError(
            f"MegaMoE supports at most {_MAX_LOCAL_NUM_TOKENS} local tokens, "
            f"got {hidden_states.shape[0]}"
        )
    executor = _executor(hidden_states, expert_indices, experts, group)
    num_local_experts = experts.num_local_experts
    hidden_size = hidden_states.shape[1]
    intermediate_size = experts.config.moe_ffn_hidden_size
    weight1 = _local_weight(
        experts.weight1,
        (num_local_experts, hidden_size, intermediate_size * 2),
        hidden_states,
    )
    weight2 = _local_weight(
        experts.weight2,
        (num_local_experts, intermediate_size, hidden_size),
        hidden_states,
    )
    return executor(
        hidden_states,
        expert_indices.to(dtype=torch.int32),
        probs,
        expert_weights=(weight1, weight2),
    )
