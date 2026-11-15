"""Unit tests for the MegaMoE TorchTurbo integration boundary."""

from types import SimpleNamespace
from unittest.mock import Mock

import torch
import torch.nn.functional as F

from torchturbo.models.modules.moe import mega_moe_reg
from torchturbo.models.modules.moe.moe_config import MoEConfig


def _experts(swiglu_limit=0.0):
    config = MoEConfig(
        hidden_size=4,
        moe_ffn_hidden_size=3,
        gated_linear_unit=True,
        activation_func=F.silu,
        moe_swiglu_limit=[0.0, 4.0],
        moe_swiglu_clamp_gate_after_activation=True,
    )
    return SimpleNamespace(
        config=config,
        num_local_experts=2,
        swiglu_limit=swiglu_limit,
        weight1=torch.zeros(8, 6),
        weight2=torch.zeros(6, 4),
    )


def test_fused_reg_moe_preserves_external_weight_layout(monkeypatch):
    experts = _experts()
    captured = {}

    class _Executor:
        def __call__(self, hidden, ids, probs, *, expert_weights):
            captured["hidden"] = hidden
            captured["ids"] = ids
            captured["probs"] = probs
            captured["weights"] = expert_weights
            return hidden

    monkeypatch.setattr(mega_moe_reg, "_executor", lambda *args: _Executor())
    hidden = torch.zeros(2, 4)
    ids = torch.zeros(2, 2, dtype=torch.int64)
    probs = torch.zeros(2, 2)
    output = mega_moe_reg.fused_reg_moe(hidden, ids, probs, experts, object(), None)

    assert output is hidden
    assert captured["ids"].dtype == torch.int32
    assert captured["probs"] is probs
    assert captured["weights"][0].shape == (2, 4, 6)
    assert captured["weights"][1].shape == (2, 3, 4)


def test_executor_is_shared_only_for_identical_activation_semantics(monkeypatch):
    monkeypatch.setattr(mega_moe_reg.dist, "get_world_size", lambda group: 1)
    created = []

    class _Executor:
        def __init__(self, **kwargs):
            created.append(kwargs)

    monkeypatch.setattr(mega_moe_reg, "MegaMoeExperts", _Executor)
    mega_moe_reg._EXECUTORS.clear()
    hidden = torch.zeros(2, 4)
    ids = torch.zeros(2, 2, dtype=torch.int32)
    experts = _experts()
    group = object()
    first = mega_moe_reg._executor(hidden, ids, experts, group)
    second = mega_moe_reg._executor(hidden, ids, experts, group)
    clamped = mega_moe_reg._executor(hidden, ids, _experts(4.0), group)

    assert first is second
    assert first is not clamped
    assert len(created) == 2
    assert created[0]["max_local_num_tokens"] == 16384
    assert created[0]["create_parameters"] is False
    assert created[0]["swiglu_limit"] is None
    assert created[1]["swiglu_limit"] == 4.0
    assert created[1]["swiglu_clamp_gate_after_activation"] is True
    mega_moe_reg._EXECUTORS.clear()


def test_dtensor_weight_unwrap_preserves_dtensor_to_copy():
    weight = Mock(spec=mega_moe_reg.DTensor)
    local_weight = torch.zeros(2, 4, 6)
    weight.view.return_value.to.return_value.to_local.return_value = local_weight
    result = mega_moe_reg._local_weight(weight, (2, 4, 6), torch.zeros(1, 4))

    assert result is local_weight
    weight.view.assert_called_once_with((2, 4, 6))
    weight.view.return_value.to.assert_called_once()
    weight.view.return_value.to.return_value.to_local.assert_called_once_with()
