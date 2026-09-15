from __future__ import annotations

import re

import torch


def controlled_returns(
    observed: torch.Tensor,
    mode: str,
    *,
    seed: int = 0,
    epoch: int = 0,
) -> torch.Tensor:
    """Build return-control labels without changing the observed tensor.

    ``random`` is a deterministic global permutation. Unlike within-task
    shuffling it deliberately breaks both task difficulty and return pairing,
    while retaining the empirical label histogram.
    """
    if mode == "observed":
        return observed.clone()
    if mode == "zero":
        return torch.zeros_like(observed)
    if mode == "sign_flip":
        return -observed
    if mode == "random":
        generator = torch.Generator(device=observed.device).manual_seed(seed + epoch)
        return observed[torch.randperm(len(observed), device=observed.device, generator=generator)]
    raise ValueError(f"unsupported return_mode: {mode}")


def parse_trajectory_horizon(value: str | int | None) -> int | None:
    """Return None for a full episode, otherwise a positive step horizon."""
    if value is None or str(value).lower() == "full":
        return None
    horizon = int(value)
    if horizon < 1:
        raise ValueError("trajectory_horizon must be a positive integer or 'full'")
    return horizon


def select_trajectory_steps(
    steps: list[dict], horizon: str | int | None = "full", selection: str = "all"
) -> list[dict]:
    """Select the evidence used by the trajectory-policy objective.

    Horizon truncates the trajectory to its first H transitions. ``first`` and
    ``final`` then select one transition from that window. Thus H=1 and final
    are intentionally identical to first, making token-vs-trajectory controls
    unambiguous.
    """
    limit = parse_trajectory_horizon(horizon)
    window = steps if limit is None else steps[:limit]
    if selection == "all":
        return window
    if selection == "first":
        return window[:1]
    if selection == "final":
        return window[-1:]
    raise ValueError(f"unsupported step_selection: {selection}")


def remove_prompt_history(prompt: str) -> str:
    """Blank a serialized one-line History field while retaining candidates."""
    return re.sub(r"(?m)^History:.*$", "History: []", prompt)
