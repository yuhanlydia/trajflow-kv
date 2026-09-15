import pytest
import torch

from trajflow_kv.training_controls import (
    controlled_returns,
    parse_trajectory_horizon,
    remove_prompt_history,
    select_trajectory_steps,
)


def test_sign_flip_and_zero_return_controls():
    observed = torch.tensor([0.0, 0.5, 1.0])
    assert torch.equal(controlled_returns(observed, "sign_flip"), -observed)
    assert torch.equal(controlled_returns(observed, "zero"), torch.zeros_like(observed))
    assert torch.equal(controlled_returns(observed, "observed"), observed)


def test_random_return_control_is_seeded_and_preserves_histogram():
    observed = torch.arange(12.0)
    first = controlled_returns(observed, "random", seed=17, epoch=2)
    repeated = controlled_returns(observed, "random", seed=17, epoch=2)
    other_epoch = controlled_returns(observed, "random", seed=17, epoch=3)
    assert torch.equal(first, repeated)
    assert not torch.equal(first, other_epoch)
    assert sorted(first.tolist()) == sorted(observed.tolist())


def test_horizon_and_first_final_controls_have_explicit_order():
    steps = [{"id": index} for index in range(7)]
    assert select_trajectory_steps(steps, "full", "all") == steps
    assert select_trajectory_steps(steps, 3, "all") == steps[:3]
    assert select_trajectory_steps(steps, 5, "first") == steps[:1]
    assert select_trajectory_steps(steps, 5, "final") == steps[4:5]
    assert select_trajectory_steps(steps, 1, "final") == steps[:1]


def test_horizon_rejects_nonpositive_values():
    assert parse_trajectory_horizon("full") is None
    with pytest.raises(ValueError, match="positive integer"):
        parse_trajectory_horizon("0")


def test_remove_history_retains_explicit_candidate_prompt():
    prompt = "Task: click red\nHistory: ['old action']\nCandidates:\n[0] red"
    cleaned = remove_prompt_history(prompt)
    assert "History: []" in cleaned
    assert "old action" not in cleaned
    assert "Candidates:\n[0] red" in cleaned
