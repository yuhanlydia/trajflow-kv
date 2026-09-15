from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .causal_ablation import decoder_layer_index, vision_token_spans


class LowRankResidual(nn.Module):
    """x -> x + alpha * A(B(x)); initialized as an identity modification."""

    def __init__(self, width: int, rank: int, alpha: float):
        super().__init__()
        if not 0 < rank <= width:
            raise ValueError(f"rank must be in [1, {width}], got {rank}")
        self.down = nn.Linear(width, rank, bias=False)
        self.up = nn.Linear(rank, width, bias=False)
        nn.init.orthogonal_(self.down.weight)
        nn.init.zeros_(self.up.weight)
        self.scale = alpha / rank
        self.last_delta: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        delta = self.up(self.down(x)) * self.scale
        self.last_delta = delta
        return x + delta

    def energy(self) -> torch.Tensor:
        if self.last_delta is None:
            return self.up.weight.square().mean()
        return self.last_delta.float().square().mean()

    def orthogonality_loss(self) -> torch.Tensor:
        rows = self.down.weight.float()
        gram = rows @ rows.T
        eye = torch.eye(gram.shape[0], device=gram.device)
        return (gram - eye).square().mean()


class StateConditionedLowRankResidual(LowRankResidual):
    """Low-rank transport gated per memory token by the decision state."""

    def __init__(self, width: int, rank: int, alpha: float, gate_rank: int = 16, signed_gate: bool = False):
        super().__init__(width, rank, alpha)
        self.query_gate = nn.Linear(width, gate_rank, bias=False)
        self.memory_gate = nn.Linear(width, gate_rank, bias=False)
        self.signed_gate = bool(signed_gate)
        self.gate_bias = nn.Parameter(torch.tensor(0.0 if signed_gate else -2.0))
        self._memory_mask: torch.Tensor | None = None
        self._decision_index: int | None = None
        self.last_gate: torch.Tensor | None = None

    def set_context(self, memory_mask: torch.Tensor, decision_index: int) -> None:
        self._memory_mask = memory_mask
        self._decision_index = int(decision_index)

    def clear_context(self) -> None:
        self._memory_mask = None
        self._decision_index = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._memory_mask is None or self._decision_index is None:
            self.last_delta = torch.zeros_like(x)
            self.last_gate = None
            return x
        if self._decision_index >= x.shape[-2]:
            raise ValueError("decision index outside K/V sequence")
        mask = self._memory_mask.to(device=x.device)
        if mask.shape[-1] != x.shape[-2]:
            raise ValueError("memory mask length does not match K/V sequence")
        query = self.query_gate(x[..., self._decision_index : self._decision_index + 1, :])
        memory = self.memory_gate(x)
        logits = (query * memory).sum(dim=-1, keepdim=True) / (query.shape[-1] ** 0.5)
        gate = torch.tanh(logits + self.gate_bias) if self.signed_gate else torch.sigmoid(logits + self.gate_bias)
        gate = gate * mask[..., None].to(gate.dtype)
        delta = self.up(self.down(x)) * self.scale * gate
        self.last_gate = gate
        self.last_delta = delta
        return x + delta


@dataclass
class HookedProjectors:
    modules: nn.ModuleList
    handles: list
    names: list[str]

    def energy(self) -> torch.Tensor:
        return torch.stack([m.energy() for m in self.modules]).mean()

    def orthogonality_loss(self) -> torch.Tensor:
        return torch.stack([m.orthogonality_loss() for m in self.modules]).mean()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()

    def set_visual_memory_context(
        self, input_ids: torch.Tensor, history_image_count: int, decision_index: int
    ) -> None:
        spans = vision_token_spans(input_ids)
        if history_image_count > len(spans):
            raise ValueError("history image count exceeds visual token blocks")
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for start, end in spans[:history_image_count]:
            mask[..., start:end] = True
        for module in self.modules:
            setter = getattr(module, "set_context", None)
            if setter is not None:
                setter(mask, decision_index)

    def clear_context(self) -> None:
        for module in self.modules:
            clearer = getattr(module, "clear_context", None)
            if clearer is not None:
                clearer()


def attach_kv_projectors(
    model: nn.Module,
    rank: int,
    alpha: float,
    target: str = "both",
    last_n_layers: int | None = None,
) -> HookedProjectors:
    """Attach projectors to real decoder K/V projection outputs.

    Hooks preserve autograd. Projectors are registered on the root model so
    optimizers and checkpoints see them, while the frozen base stays fixed.
    """
    if target not in {"k", "v", "both"}:
        raise ValueError("target must be k, v, or both")
    suffixes = {"k": ("k_proj",), "v": ("v_proj",), "both": ("k_proj", "v_proj")}[target]
    selected = [(name, module) for name, module in model.named_modules()
                if name.rsplit(".", 1)[-1] in suffixes and isinstance(module, nn.Linear)]
    if not selected:
        raise RuntimeError("No decoder k_proj/v_proj Linear modules found")
    if last_n_layers is not None:
        if last_n_layers <= 0:
            raise ValueError("last_n_layers must be positive")
        # Keep K and V from the requested number of final transformer layers.
        keep = last_n_layers * len(suffixes)
        selected = selected[-keep:]

    projectors = nn.ModuleList()
    handles, names = [], []
    for name, linear in selected:
        projector = LowRankResidual(linear.out_features, rank, alpha).to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )
        projectors.append(projector)
        handles.append(linear.register_forward_hook(
            lambda _module, _inputs, output, p=projector: p(output)
        ))
        names.append(name)
    model.add_module("trajflow_kv_projectors", projectors)
    return HookedProjectors(projectors, handles, names)


def attach_gated_kv_projectors(
    model: nn.Module,
    rank: int,
    alpha: float,
    target: str = "both",
    layers: Iterable[int] | None = None,
    gate_rank: int = 16,
    signed_gate: bool = False,
) -> HookedProjectors:
    """Attach state-conditioned transport only to selected decoder layers."""
    if target not in {"k", "v", "both"}:
        raise ValueError("target must be k, v, or both")
    suffixes = {"k": ("k_proj",), "v": ("v_proj",), "both": ("k_proj", "v_proj")}[target]
    layer_set = set(int(layer) for layer in layers) if layers is not None else None
    selected = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in suffixes or not isinstance(module, nn.Linear):
            continue
        layer = decoder_layer_index(name)
        if layer is None:
            continue
        if layer_set is None or layer in layer_set:
            selected.append((name, module))
    if not selected:
        raise RuntimeError("No decoder K/V modules selected for gated transport")

    projectors = nn.ModuleList()
    handles, names = [], []
    for name, linear in selected:
        projector = StateConditionedLowRankResidual(
            linear.out_features, rank, alpha, gate_rank=gate_rank, signed_gate=signed_gate
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        projectors.append(projector)
        handles.append(linear.register_forward_hook(
            lambda _module, _inputs, output, p=projector: p(output)
        ))
        names.append(name)
    model.add_module("trajflow_kv_gated_projectors", projectors)
    return HookedProjectors(projectors, handles, names)


def trainable_parameters(bundle: HookedProjectors) -> Iterable[nn.Parameter]:
    return bundle.modules.parameters()


def merge_projectors_into_model(model: nn.Module, bundle: HookedProjectors) -> dict[str, torch.Tensor]:
    """Fold linear residual projectors into their K/V projection weights."""
    modules = dict(model.named_modules())
    merged = {}
    bundle.close()
    with torch.no_grad():
        for name, projector in zip(bundle.names, bundle.modules, strict=True):
            linear = modules[name]
            delta = (
                projector.up.weight.float()
                @ projector.down.weight.float()
                @ linear.weight.float()
            ) * projector.scale
            linear.weight.add_(delta.to(linear.weight.dtype))
            merged[f"{name}.weight"] = linear.weight.detach().cpu().clone()
            if linear.bias is not None:
                bias_delta = (
                    projector.up.weight.float()
                    @ projector.down.weight.float()
                    @ linear.bias.float()
                ) * projector.scale
                linear.bias.add_(bias_delta.to(linear.bias.dtype))
                merged[f"{name}.bias"] = linear.bias.detach().cpu().clone()
    return merged


def load_merged_weights(model: nn.Module, checkpoint: str) -> list[str]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    weights = payload.get("weights", payload)
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for name, value in weights.items():
            parameters[name].copy_(value.to(parameters[name].device, parameters[name].dtype))
    return sorted(weights)


def resize_low_rank_factors(
    down: torch.Tensor,
    up: torch.Tensor,
    source_alpha: float,
    target_rank: int,
    target_alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """SVD-reparameterize a residual map at another rank and scale."""
    source_scale = source_alpha / down.shape[0]
    target_scale = target_alpha / target_rank
    effective = source_scale * up.float() @ down.float()
    left, singular, right = torch.linalg.svd(effective, full_matrices=False)
    take = min(target_rank, len(singular))
    new_up = torch.zeros(up.shape[0], target_rank)
    new_down = torch.zeros(target_rank, down.shape[1])
    for index in range(take):
        if singular[index] > 1e-10:
            root = torch.sqrt(singular[index] / target_scale)
            new_up[:, index] = left[:, index] * root
            new_down[index] = right[index] * root
        else:
            # A zero-up/nonzero-down slot preserves the map but can learn.
            new_down[index] = right[index]
    return new_down.to(down.dtype), new_up.to(up.dtype)
