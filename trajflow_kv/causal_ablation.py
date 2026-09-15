"""Token-aligned causal ablations for visual K/V memory blocks."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable

import torch
from torch import nn


VISION_START_TOKEN_ID = 151652
VISION_END_TOKEN_ID = 151653


def vision_token_spans(
    input_ids: torch.Tensor | Iterable[int],
    *,
    start_token_id: int = VISION_START_TOKEN_ID,
    end_token_id: int = VISION_END_TOKEN_ID,
) -> list[tuple[int, int]]:
    """Return half-open token spans for images, including boundary tokens."""
    values = input_ids.detach().flatten().tolist() if isinstance(input_ids, torch.Tensor) else list(input_ids)
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, token_id in enumerate(values):
        if int(token_id) == start_token_id:
            if start is not None:
                raise ValueError("nested vision token span")
            start = index
        elif int(token_id) == end_token_id:
            if start is None:
                raise ValueError("vision end token without start")
            spans.append((start, index + 1))
            start = None
    if start is not None:
        raise ValueError("unterminated vision token span")
    return spans


def decoder_layer_index(module_name: str) -> int | None:
    match = re.search(r"(?:^|\.)layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else None


@dataclass
class KVBlockAblator:
    """Zero one visual token block at selected decoder K/V projections."""

    handles: list
    names: list[str]
    enabled: bool = False
    span: tuple[int, int] | None = None
    replacement_span: tuple[int, int] | None = None

    def set_image(self, input_ids: torch.Tensor, image_index: int, replacement_image_index: int | None = None) -> None:
        spans = vision_token_spans(input_ids)
        if not -len(spans) <= image_index < len(spans):
            raise IndexError(f"image index {image_index} outside {len(spans)} visual blocks")
        self.span = spans[image_index]
        self.replacement_span = None if replacement_image_index is None else spans[replacement_image_index]
        if self.replacement_span is not None:
            target_length = self.span[1] - self.span[0]
            source_length = self.replacement_span[1] - self.replacement_span[0]
            if target_length != source_length:
                raise ValueError(f"KV patch spans must have equal length, got {target_length} and {source_length}")
        self.enabled = True

    def clear(self) -> None:
        self.enabled = False
        self.span = None
        self.replacement_span = None

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()


def attach_kv_block_ablator(
    model: nn.Module,
    *,
    target: str = "both",
    layers: Iterable[int] | None = None,
) -> KVBlockAblator:
    """Attach a controllable zero-ablation hook to decoder K/V projections."""
    if target not in {"k", "v", "both"}:
        raise ValueError("target must be k, v, or both")
    suffixes = {"k": {"k_proj"}, "v": {"v_proj"}, "both": {"k_proj", "v_proj"}}[target]
    layer_set = set(int(layer) for layer in layers) if layers is not None else None
    selected: list[tuple[str, nn.Module]] = []
    for name, module in model.named_modules():
        if name.rsplit(".", 1)[-1] not in suffixes or not isinstance(module, nn.Linear):
            continue
        layer = decoder_layer_index(name)
        if layer is None or (layer_set is not None and layer not in layer_set):
            continue
        selected.append((name, module))
    if not selected:
        raise RuntimeError("No decoder K/V modules selected for causal ablation")

    ablator = KVBlockAblator(handles=[], names=[name for name, _ in selected])

    def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> torch.Tensor:
        if not ablator.enabled or ablator.span is None:
            return output
        start, end = ablator.span
        if output.ndim < 3 or end > output.shape[-2]:
            raise ValueError(
                f"ablation span {(start, end)} incompatible with K/V output {tuple(output.shape)}"
            )
        ablated = output.clone()
        if ablator.replacement_span is None:
            ablated[..., start:end, :] = 0
        else:
            source_start, source_end = ablator.replacement_span
            ablated[..., start:end, :] = output[..., source_start:source_end, :]
        return ablated

    ablator.handles = [module.register_forward_hook(hook) for _, module in selected]
    return ablator
