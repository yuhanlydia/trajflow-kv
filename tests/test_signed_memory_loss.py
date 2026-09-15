import pytest
import torch

from trajflow_kv.train import memory_gate_supervision_loss


def test_huber_memory_loss_preserves_credit_sign():
    advantages = torch.tensor([-0.2, 0.0, 0.2])
    good = torch.tensor([-1.0, 0.0, 1.0])
    wrong = -good
    assert memory_gate_supervision_loss(good, advantages, scale=0.2, loss_type="huber", signed_gate=True) < memory_gate_supervision_loss(wrong, advantages, scale=0.2, loss_type="huber", signed_gate=True)


def test_huber_memory_loss_requires_signed_gate():
    with pytest.raises(ValueError, match="signed_gate"):
        memory_gate_supervision_loss(torch.zeros(1), torch.ones(1), scale=1.0, loss_type="huber")
