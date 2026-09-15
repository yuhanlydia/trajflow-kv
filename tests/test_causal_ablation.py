import torch
from torch import nn

from trajflow_kv.causal_ablation import (
    attach_kv_block_ablator,
    decoder_layer_index,
    vision_token_spans,
)


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


def test_vision_token_spans_are_half_open_and_ordered():
    ids = torch.tensor([[1, 151652, 9, 151653, 2, 151652, 8, 7, 151653]])
    assert vision_token_spans(ids) == [(1, 4), (5, 9)]


def test_decoder_layer_index():
    assert decoder_layer_index("model.language_model.layers.17.self_attn.k_proj") == 17
    assert decoder_layer_index("visual.blocks.2.attn.k_proj") is None


def test_kv_block_ablator_targets_selected_layer_and_projection():
    model = _Toy()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.eye(4))
    ablator = attach_kv_block_ablator(model, target="v", layers=[1])
    ablator.set_image(torch.tensor([[151652, 3, 151653, 4]]), 0)
    x = torch.ones(1, 4, 4)
    untouched = model.layers[0].self_attn.v_proj(x)
    ablated = model.layers[1].self_attn.v_proj(x)
    key = model.layers[1].self_attn.k_proj(x)
    assert torch.all(untouched == 1)
    assert torch.all(ablated[:, :3] == 0)
    assert torch.all(ablated[:, 3:] == 1)
    assert torch.all(key == 1)
    ablator.close()


def test_kv_block_ablator_can_patch_from_matched_visual_span():
    model = _Toy()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.copy_(torch.eye(4))
    ablator = attach_kv_block_ablator(model, target="k", layers=[0])
    ids = torch.tensor([[151652, 9, 151653, 4, 151652, 8, 151653]])
    ablator.set_image(ids, 0, replacement_image_index=1)
    x = torch.arange(28, dtype=torch.float32).reshape(1, 7, 4)
    patched = model.layers[0].self_attn.k_proj(x)
    assert torch.equal(patched[:, 0:3], x[:, 4:7])
    assert torch.equal(patched[:, 4:7], x[:, 4:7])
    ablator.close()
