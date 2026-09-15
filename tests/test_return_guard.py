import pytest
import torch

from trajflow_kv.objective import normalized_advantages


def test_repeated_task_mixed_returns_create_nonzero_advantages():
    advantages = normalized_advantages(torch.tensor([0.0, 1.0, 0.0, 1.0]), ["a"] * 4)
    assert torch.count_nonzero(advantages) == 4


def test_singleton_groups_have_zero_advantage():
    advantages = normalized_advantages(torch.tensor([0.0, 1.0]), ["a", "b"])
    assert torch.count_nonzero(advantages) == 0
