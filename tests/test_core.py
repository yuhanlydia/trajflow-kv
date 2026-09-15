import torch
from torch import nn

from trajflow_kv.objective import (
    normalized_advantages,
    shuffle_within_tasks,
    trajectory_policy_loss,
)
from trajflow_kv.projector import (
    attach_kv_projectors, merge_projectors_into_model, resize_low_rank_factors,
)
from trajflow_kv.forks import build_coordinate_fork_pairs, build_fork_pairs
from trajflow_kv.qwen_policy import action_signature, exclude_repeated_candidates
from trajflow_kv.state_router import routed_action_type, router_features
from trajflow_kv.critic import TrajectoryCritic, critic_features
from trajflow_kv.action_flow import (
    ActionRectifiedFlow, STEP_WIDTH, encode_action_chunk, project_action_chunks,
)


class Tiny(nn.Module):
    def __init__(self):
        super().__init__(); self.k_proj = nn.Linear(8, 8); self.v_proj = nn.Linear(8, 8)
    def forward(self, x): return self.k_proj(x) + self.v_proj(x)


def test_hooks_change_output_and_receive_gradients():
    model = Tiny()
    baseline = model(torch.ones(2, 8)).detach()
    for p in model.parameters(): p.requires_grad_(False)
    bundle = attach_kv_projectors(model, rank=2, alpha=1.0)
    bundle.modules[0].up.weight.data.fill_(0.1)
    output = model(torch.ones(2, 8))
    assert not torch.allclose(output, baseline)
    output.sum().backward()
    assert bundle.modules[0].up.weight.grad is not None


def test_last_n_layers_limits_hook_count():
    model = nn.Sequential(Tiny(), Tiny(), Tiny())
    bundle = attach_kv_projectors(
        model, rank=2, alpha=1.0, target="both", last_n_layers=1
    )
    assert len(bundle.modules) == 2


def test_projector_merge_matches_hooked_linear_output():
    model = Tiny().eval()
    bundle = attach_kv_projectors(model, rank=2, alpha=2.0, target="v")
    bundle.modules[0].up.weight.data.normal_(std=0.02)
    value = torch.randn(3, 8)
    hooked = model(value).detach()
    merged = merge_projectors_into_model(model, bundle)
    assert list(merged) == ["v_proj.weight", "v_proj.bias"]
    assert torch.allclose(model(value), hooked, atol=1e-5, rtol=1e-5)


def test_rank_expansion_preserves_effective_residual():
    down = torch.randn(3, 8)
    up = torch.randn(8, 3)
    new_down, new_up = resize_low_rank_factors(down, up, 6.0, 5, 10.0)
    original = (6.0 / 3) * up @ down
    expanded = (10.0 / 5) * new_up @ new_down
    assert torch.allclose(expanded, original, atol=1e-4, rtol=1e-4)


def test_return_objective():
    returns = torch.tensor([1., 0., 0., 1.])
    adv = normalized_advantages(returns, ["a", "a", "b", "b"])
    assert torch.allclose(adv.mean(), torch.tensor(0.))
    loss = trajectory_policy_loss(torch.tensor([-1., -2., -3., -1.]), adv)
    assert torch.isfinite(loss)


def test_shuffle_preserves_each_task_return_histogram():
    returns = torch.tensor([0.0, 1.0, 1.0, 0.0, 0.0, 1.0])
    task_ids = ["a", "a", "a", "b", "b", "b"]
    shuffled = shuffle_within_tasks(
        returns, task_ids, generator=torch.Generator().manual_seed(4)
    )
    assert sorted(shuffled[:3].tolist()) == sorted(returns[:3].tolist())
    assert sorted(shuffled[3:].tolist()) == sorted(returns[3:].tolist())


def test_fork_pairs_use_success_state_and_different_failed_action():
    records = [
        {"task_id": "t", "instruction": "do it", "return": 1, "steps": [
            {"image": "good.png", "history": [], "action": '{"action_type":"click","x":1,"y":2}'}
        ]},
        {"task_id": "t", "instruction": "do it", "return": 0, "steps": [
            {"image": "bad.png", "history": [], "action": '{"action_type":"wait"}'}
        ]},
    ]
    pairs = build_fork_pairs(records)
    assert len(pairs) == 1
    assert pairs[0]["image"] == "good.png"
    assert pairs[0]["chosen"] != pairs[0]["rejected"]
    assert build_fork_pairs(records, chosen_action_type="swipe") == []
    assert build_fork_pairs(records, rejected_action_type="click") == []


def test_coordinate_forks_keep_click_type_and_move_point():
    records = [{"task_id": "t", "instruction": "do it", "return": 1, "steps": [
        {"image": "good.png", "history": [], "action": '{"action_type":"click","x":500,"y":500}'}
    ]}]
    pairs = build_coordinate_fork_pairs(records, offsets=(100,))
    assert len(pairs) == 4
    assert all('"action_type":"click"' in pair["rejected"] for pair in pairs)
    assert all(pair["chosen"] != pair["rejected"] for pair in pairs)


def test_loop_signature_ignores_incidental_swipe_coordinates():
    assert action_signature({"action_type": "swipe", "direction": "down"}) == action_signature(
        {"action_type": "swipe", "direction": "down", "x": 12, "y": 20}
    )
    assert action_signature({"action_type": "scroll", "direction": "down"}) == action_signature(
        {"action_type": "swipe", "direction": "down"}
    )


def test_exact_candidate_loop_guard_keeps_other_click_coordinates():
    candidates = [
        '{"action_type":"click","x":10,"y":20}',
        '{"action_type":"click","x":30,"y":40}',
    ]
    kept = exclude_repeated_candidates(candidates, [candidates[0]], (100, 100), 1)
    assert kept == [candidates[1]]


def test_state_router_features_and_labels():
    from PIL import Image
    features = router_features(Image.new("RGB", (40, 80), "white"), 3, "turn wifi off")
    assert features.shape == (64 * 64 * 3 + 1 + 32,)
    assert routed_action_type('{"action_type":"swipe","direction":"down"}') == "pan"
    assert routed_action_type('{"action_type":"click","x":1,"y":2}') == "click"


def test_critic_features_keep_logprob_differentiable():
    logprob = torch.tensor(-1.0, requires_grad=True)
    trajectory = {"steps": [{"action": '{"action_type":"click","x":1,"y":2}'}]}
    features = critic_features(trajectory, logprob)
    critic = TrajectoryCritic(len(features))
    critic(features).backward()
    assert logprob.grad is not None


def test_action_flow_chunk_shape_and_gradient():
    trajectory = {"steps": [
        {"action": '{"action_type":"swipe","direction":"down"}'},
        {"action": '{"action_type":"click","x":500,"y":200}'},
    ]}
    target = encode_action_chunk(trajectory, horizon=3)
    assert target.shape == (3 * STEP_WIDTH,)
    model = ActionRectifiedFlow(len(target), 5, hidden=16)
    state = torch.randn(2, len(target))
    output = model(state, torch.rand(2), torch.randn(2, 5))
    output.square().mean().backward()
    assert any(parameter.grad is not None for parameter in model.parameters())
    projected = project_action_chunks(output.detach(), horizon=3)
    assert projected.shape == output.shape
