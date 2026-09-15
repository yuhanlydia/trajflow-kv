import torch
from torch import nn

from trajflow_kv.projector import StateConditionedLowRankResidual, attach_gated_kv_projectors


class _Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.k_proj = nn.Linear(4, 4, bias=False)
        self.v_proj = nn.Linear(4, 4, bias=False)


class _Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = _Attention()


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_Layer(), _Layer()])


def test_state_conditioned_residual_changes_only_masked_memory_tokens():
    module = StateConditionedLowRankResidual(4, rank=2, alpha=2, gate_rank=2)
    with torch.no_grad():
        module.up.weight.fill_(0.25)
    x = torch.ones(1, 5, 4)
    mask = torch.tensor([[False, True, True, False, False]])
    module.set_context(mask, decision_index=4)
    output = module(x)
    assert torch.allclose(output[:, [0, 3, 4]], x[:, [0, 3, 4]])
    assert not torch.allclose(output[:, 1:3], x[:, 1:3])


def test_gated_bundle_builds_visual_history_mask():
    model = _Toy()
    bundle = attach_gated_kv_projectors(model, 2, 2, target="v", layers=[1], gate_rank=2)
    ids = torch.tensor([[151652, 9, 151653, 151652, 8, 151653, 4]])
    bundle.set_visual_memory_context(ids, history_image_count=1, decision_index=6)
    module = bundle.modules[0]
    assert module._memory_mask.tolist() == [[True, True, True, False, False, False, False]]
    bundle.clear_context()
    assert module._memory_mask is None
    bundle.close()


def test_signed_gate_can_suppress_selected_memory_tokens():
    module = StateConditionedLowRankResidual(4, rank=2, alpha=2, gate_rank=2, signed_gate=True)
    with torch.no_grad():
        module.gate_bias.fill_(-2.0)
    module.set_context(torch.tensor([[True, False]]), decision_index=1)
    output = module(torch.ones(1, 2, 4))
    assert output.shape == (1, 2, 4)
    assert module.last_gate[0, 0, 0] < 0
    assert module.last_gate[0, 1, 0] == 0
