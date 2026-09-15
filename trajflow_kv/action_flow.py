from __future__ import annotations

import hashlib
import json

import numpy as np
import torch
from PIL import Image
from torch import nn


ACTION_TYPES = ("click", "pan", "status", "wait", "open_app", "pad", "other")
DIRECTIONS = ("down", "up", "left", "right")
STEP_WIDTH = len(ACTION_TYPES) + 2 + len(DIRECTIONS)


def _group(action_type: str) -> str:
    if action_type in {"swipe", "scroll"}:
        return "pan"
    return action_type if action_type in ACTION_TYPES else "other"


def encode_action_chunk(trajectory: dict, horizon: int = 4) -> torch.Tensor:
    encoded = []
    for index in range(horizon):
        action = (
            json.loads(trajectory["steps"][index]["action"])
            if index < len(trajectory["steps"])
            else {"action_type": "pad"}
        )
        vector = torch.zeros(STEP_WIDTH)
        group = _group(action["action_type"])
        vector[ACTION_TYPES.index(group)] = 1.0
        vector[len(ACTION_TYPES)] = float(action.get("x", 0)) / 999.0
        vector[len(ACTION_TYPES) + 1] = float(action.get("y", 0)) / 999.0
        direction = action.get("direction")
        if direction in DIRECTIONS:
            vector[len(ACTION_TYPES) + 2 + DIRECTIONS.index(direction)] = 1.0
        encoded.append(vector)
    return torch.cat(encoded)


def flow_condition(trajectory: dict, text_bins: int = 32, image_size: int = 8) -> torch.Tensor:
    text = torch.zeros(text_bins)
    for token in trajectory["instruction"].lower().replace("-", " ").split():
        digest = hashlib.sha256(token.encode()).digest()
        text[int.from_bytes(digest[:2], "little") % text_bins] += (
            1.0 if digest[2] % 2 else -1.0
        )
    if text.norm() > 0:
        text /= text.norm()
    first = trajectory["steps"][0]
    image = Image.open(first["image"]).convert("RGB").resize((image_size, image_size))
    pixels = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).flatten() / 255.0
    return torch.cat([text, pixels])


class ActionRectifiedFlow(nn.Module):
    def __init__(self, action_width: int, condition_width: int, hidden: int = 128):
        super().__init__()
        self.action_width = action_width
        self.condition_width = condition_width
        self.network = nn.Sequential(
            nn.Linear(action_width + condition_width + 1, hidden),
            nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, action_width),
        )

    def forward(self, state: torch.Tensor, time: torch.Tensor, condition: torch.Tensor):
        if time.ndim == 1:
            time = time[:, None]
        return self.network(torch.cat([state, time, condition], dim=-1))

    def sample(self, noise: torch.Tensor, condition: torch.Tensor, steps: int = 50):
        state = noise
        dt = 1.0 / steps
        for index in range(steps):
            time = torch.full((len(state),), index * dt, device=state.device)
            state = state + dt * self(state, time, condition)
        return state


def project_action_chunks(chunks: torch.Tensor, horizon: int) -> torch.Tensor:
    """Snap continuous flow endpoints to the structured action manifold."""
    original_shape = chunks.shape
    values = chunks.reshape(-1, horizon, STEP_WIDTH).clone()
    type_scores = values[:, :, :len(ACTION_TYPES)]
    type_index = type_scores.argmax(-1)
    values[:, :, :len(ACTION_TYPES)] = torch.nn.functional.one_hot(
        type_index, len(ACTION_TYPES)
    ).to(values.dtype)
    coordinate_start = len(ACTION_TYPES)
    values[:, :, coordinate_start:coordinate_start + 2].clamp_(0, 1)
    direction_start = coordinate_start + 2
    direction_scores = values[:, :, direction_start:]
    directions = torch.nn.functional.one_hot(
        direction_scores.argmax(-1), len(DIRECTIONS)
    ).to(values.dtype)
    is_pan = (type_index == ACTION_TYPES.index("pan")).unsqueeze(-1)
    values[:, :, direction_start:] = directions * is_pan
    return values.reshape(original_shape)
