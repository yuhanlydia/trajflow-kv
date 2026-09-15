from __future__ import annotations

import json

import torch
from torch import nn


ACTION_GROUPS = ("click", "pan", "wait", "status", "open_app", "other")


def critic_features(trajectory: dict, mean_token_logprob: float | torch.Tensor) -> torch.Tensor:
    counts = {name: 0.0 for name in ACTION_GROUPS}
    for step in trajectory["steps"]:
        action_type = json.loads(step["action"])["action_type"]
        if action_type in {"swipe", "scroll"}:
            group = "pan"
        elif action_type in counts:
            group = action_type
        else:
            group = "other"
        counts[group] += 1.0
    steps = max(len(trajectory["steps"]), 1)
    scalar = torch.as_tensor(mean_token_logprob).reshape(1)
    static = torch.tensor(
        [min(steps, 20) / 20.0]
        + [counts[name] / steps for name in ACTION_GROUPS]
        + [float(trajectory.get("metadata", {}).get("invalid_actions", 0)) / steps],
        dtype=scalar.dtype,
        device=scalar.device,
    )
    return torch.cat([scalar, static])


class TrajectoryCritic(nn.Module):
    def __init__(self, width: int, hidden: int = 16):
        super().__init__()
        self.network = (
            nn.Linear(width, 1)
            if hidden == 0
            else nn.Sequential(nn.Linear(width, hidden), nn.Tanh(), nn.Linear(hidden, 1))
        )
        self.register_buffer("feature_mean", torch.zeros(width))
        self.register_buffer("feature_std", torch.ones(width))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        normalized = (features - self.feature_mean) / self.feature_std.clamp_min(1e-6)
        return self.network(normalized).squeeze(-1)

    def set_normalization(self, features: torch.Tensor) -> None:
        self.feature_mean.copy_(features.mean(0))
        self.feature_std.copy_(features.std(0).clamp_min(1e-6))
