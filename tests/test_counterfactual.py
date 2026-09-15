import json

import pytest
import torch

from trajflow_kv.counterfactual import (
    HiddenCueTask,
    IrreversibleForkTask,
    OrderDependencyTask,
    build_counterfactual_dataset,
    build_counterfactual_examples,
    enumerate_continuations,
    load_counterfactual_jsonl,
    write_counterfactual_jsonl,
)
from trajflow_kv.tango_advantage import (
    action_ce_loss,
    counterfactual_objective_loss,
    global_return_loss,
    tango_advantage_loss,
    train_tabular_counterfactual,
)
from trajflow_kv.train import _counterfactual_groups, _counterfactual_prompt


def _rows_for(family: str):
    task = {
        "hidden_cue": HiddenCueTask,
        "order_dependency": OrderDependencyTask,
        "irreversible_fork": IrreversibleForkTask,
    }[family]()
    return build_counterfactual_examples(task, seed=3)


@pytest.mark.parametrize("family", ["hidden_cue", "order_dependency", "irreversible_fork"])
def test_required_families_have_same_prefix_counterfactuals(family):
    rows = _rows_for(family)
    assert rows
    assert {"prefix", "action", "continuations", "Q", "V", "advantage"} <= rows[0].keys()
    grouped = {}
    for row in rows:
        grouped.setdefault(row["prefix_id"], []).append(row)
    # Every decision prefix has multiple legal candidates evaluated from the
    # exact same serialized observation/history.
    assert any(len(group) >= 2 for group in grouped.values())
    for group in grouped.values():
        if len(group) > 1:
            assert len({json.dumps(row["prefix"], sort_keys=True) for row in group}) == 1
            assert all(row["advantage"] == pytest.approx(row["Q"] - row["V"]) for row in group)


def test_hidden_cue_q_prefers_remembered_color():
    rows = _rows_for("hidden_cue")
    choose_rows = [row for row in rows if row["prefix"]["state"]["phase"] == "choose"]
    q = {row["action"]: row["Q"] for row in choose_rows if row["prefix"]["state"]["cue"] == "red"}
    assert q["choose_red"] == 1.0
    assert q["choose_blue"] == 0.0


def test_counterfactual_rollout_does_not_mutate_prefix():
    task = HiddenCueTask()
    state = task.initial_state(0)
    snapshot = json.loads(json.dumps(state, sort_keys=True))
    enumerate_continuations(task, state, "reveal", horizon=3)
    assert state == snapshot


def test_order_and_irreversible_have_delayed_good_continuation():
    order = _rows_for("order_dependency")
    first = [row for row in order if row["prefix"]["step_index"] == 0]
    assert {row["action"]: row["Q"] for row in first}["A"] == 1.0
    fork = _rows_for("irreversible_fork")
    root = [row for row in fork if row["prefix"]["state"]["phase"] == "fork"]
    values = {row["action"]: row["Q"] for row in root}
    assert values["submit_final"] == 1.0
    assert values["save_draft"] == 0.0
    draft = next(row for row in fork if row["prefix"]["state"]["phase"] == "draft")
    assert draft["continuations"]
    assert all(item["return"] == 0.0 for item in draft["continuations"])


def test_dataset_roundtrip_and_seed_generator(tmp_path):
    rows = build_counterfactual_dataset((seed for seed in [0, 1]))
    assert {row["seed"] for row in rows} == {0, 1}
    path = tmp_path / "counterfactual.jsonl"
    assert write_counterfactual_jsonl(rows, path) == len(rows)
    loaded = load_counterfactual_jsonl(path)
    assert len(loaded) == len(rows)
    assert loaded[0]["schema_version"] == "tango.counterfactual.v1"


def test_objectives_validate_shapes_and_are_differentiable():
    logits = torch.randn(4, 3, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 0])
    logp = logits.log_softmax(-1).gather(1, labels[:, None]).squeeze(1)
    adv = torch.tensor([1.0, -1.0, 0.5, -0.5])
    loss = tango_advantage_loss(logp, adv) + action_ce_loss(logits, labels)
    loss.backward()
    assert logits.grad is not None
    global_loss = global_return_loss(logp, torch.tensor([0.0, 1.0, 0.0, 1.0]), ["x"] * 4)
    assert torch.isfinite(global_loss)
    with pytest.raises(ValueError):
        tango_advantage_loss(logp, adv[:2])


def test_tabular_runner_exposes_tango_global_and_ce_interfaces():
    rows = build_counterfactual_dataset(range(1), horizon=4)
    for objective in ("tango", "global_return", "ce"):
        model, history = train_tabular_counterfactual(rows, objective=objective, epochs=2)
        assert model.logits.weight.shape[0] > 0
        assert len(history) == 2


def test_counterfactual_group_objectives_use_same_prefix_candidates():
    rows = [
        {"prefix_id": "p", "action": "bad", "Q": 0.0, "advantage": -0.5},
        {"prefix_id": "p", "action": "good", "Q": 1.0, "advantage": 0.5},
    ]
    scores = torch.tensor([0.0, 0.0], requires_grad=True)
    tango = counterfactual_objective_loss(scores, rows, objective="tango")
    tango.backward()
    # Increasing the good candidate score lowers the policy loss.
    assert scores.grad[1] < 0
    ce_scores = torch.tensor([0.0, 0.0], requires_grad=True)
    ce = counterfactual_objective_loss(ce_scores, rows, objective="ce")
    ce.backward()
    assert ce_scores.grad[1] < 0
    with pytest.raises(ValueError):
        counterfactual_objective_loss(scores.detach(), rows, objective="unknown")


def test_qwen_counterfactual_rows_are_grouped_without_losing_candidates():
    rows = [
        {
            "prefix_id": "p",
            "action": "good",
            "candidate_actions": ["bad", "good"],
            "prefix": {"history": [], "observation": {"screen": "choose"}},
        },
        {
            "prefix_id": "p",
            "action": "bad",
            "candidate_actions": ["bad", "good"],
            "prefix": {"history": [], "observation": {"screen": "choose"}},
        },
    ]
    groups = _counterfactual_groups(rows)
    assert [[row["action"] for row in group] for group in groups] == [["bad", "good"]]
    assert "Candidates" in _counterfactual_prompt(rows[0])
