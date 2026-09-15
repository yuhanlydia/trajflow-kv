import json

from PIL import Image

from trajflow_kv.counterfactual import load_counterfactual_jsonl, write_counterfactual_jsonl
from trajflow_kv.visual_delayed import (
    DistractorCreditTask,
    HiddenMemoryTask,
    InterferenceUpdateTask,
    InterferenceChainTask,
    MultiCueBindingTask,
    NonceVisualBindingTask,
    build_visual_counterfactual_dataset,
)


def _group(rows):
    groups = {}
    for row in rows:
        groups.setdefault(row["prefix_id"], []).append(row)
    return groups


def test_distractor_credit_has_equal_noncritical_q_and_critical_fork(tmp_path):
    task = DistractorCreditTask()
    rows = build_visual_counterfactual_dataset([0], output_dir=tmp_path / "images", tasks=[task], horizon=8)
    groups = _group(rows)
    assert rows
    assert all(row["visual"] for row in rows)
    assert all(row["benchmark"] == "tango_visual_delayed_gui" for row in rows)
    x_rows = next(group for group in groups.values() if group[0]["prefix"]["state"]["phase"] == "x")
    assert {row["Q"] for row in x_rows} == {x_rows[0]["Q"]}
    fork_rows = next(group for group in groups.values() if group[0]["prefix"]["state"]["phase"] == "fork")
    q = {row["action"]: row["Q"] for row in fork_rows}
    assert q["choose_A"] != q["choose_B"]
    assert all(row["is_critical_action"] for row in fork_rows)
    assert not any(row["is_critical_action"] for row in x_rows)


def test_hidden_memory_hides_cue_in_choice_screenshot_and_executes(tmp_path):
    task = HiddenMemoryTask()
    rows = build_visual_counterfactual_dataset([1], output_dir=tmp_path / "images", tasks=[task], horizon=8)
    choose = next(group for group in _group(rows).values() if group[0]["prefix"]["state"]["phase"] == "choose")
    assert all(row["critical_step"] for row in choose)
    assert all(row["is_critical_action"] for row in choose)
    # The delayed cue is available as an earlier visual observation, while
    # the current choice screenshot intentionally hides it.
    assert choose[0]["history_images"]
    assert choose[0]["memory_advantages"] == [1.0, 0.0]
    assert all(Image.open(path).size == (960, 600) for path in choose[0]["history_images"])
    image_path = choose[0]["image"]
    assert Image.open(image_path).size == (960, 600)
    # Seed 1 selects blue; the state machine is directly executable.
    state = task.initial_state(1)
    state, done = task.step(state, "continue")
    assert not done and state["phase"] == "distractor"
    state, done = task.step(state, "acknowledge")
    assert not done and state["phase"] == "choose"
    state, done = task.step(state, "choose_blue")
    assert not done and state["phase"] == "submit"
    state, done = task.step(state, "submit")
    assert done and state["terminal_return"] == 1.0


def test_visual_rows_roundtrip_existing_counterfactual_schema(tmp_path):
    rows = build_visual_counterfactual_dataset([0], output_dir=tmp_path / "images", tasks=[HiddenMemoryTask()])
    path = tmp_path / "visual.jsonl"
    assert write_counterfactual_jsonl(rows, path) == len(rows)
    loaded = load_counterfactual_jsonl(path)
    assert len(loaded) == len(rows)
    assert {row["task_family"] for row in loaded} == {"hidden_memory"}
    assert all(json.loads(json.dumps(row["prefix"]))["screenshot_path"] for row in loaded)
    initial = next(row for row in loaded if not row["prefix"]["history"])
    assert initial["history_images"] == []


def test_multi_cue_binding_requires_both_history_blocks(tmp_path):
    task = MultiCueBindingTask()
    rows = build_visual_counterfactual_dataset([2], output_dir=tmp_path / "images", tasks=[task])
    choose = next(group for group in _group(rows).values() if group[0]["prefix"]["state"]["phase"] == "choose")
    state = choose[0]["prefix"]["state"]
    correct = f"choose_{state['color']}_{state['symbol']}"
    assert len(choose) == 4
    assert len(choose[0]["history_images"]) == 3
    assert choose[0]["memory_advantages"] == [1.0, 1.0, 0.0]
    best_q = max(row["Q"] for row in choose)
    assert {row["action"] for row in choose if row["Q"] == best_q} == {correct}


def test_interference_update_marks_stale_and_updated_memory(tmp_path):
    task = InterferenceUpdateTask()
    rows = build_visual_counterfactual_dataset([3], output_dir=tmp_path / "images", tasks=[task])
    choose = next(group for group in _group(rows).values() if group[0]["prefix"]["state"]["phase"] == "choose")
    state = choose[0]["prefix"]["state"]
    assert state["old"] != state["new"]
    assert len(choose[0]["history_images"]) == 3
    assert choose[0]["memory_advantages"] == [-1.0, 1.0, 0.0]
    best_q = max(row["Q"] for row in choose)
    assert {row["action"] for row in choose if row["Q"] == best_q} == {f"choose_{state['new']}"}

    live = task.initial_state(3)
    for action, phase in (("continue", "new_cue"), ("continue", "distractor"), ("acknowledge", "choose")):
        live, done = task.step(live, action)
        assert not done and live["phase"] == phase
    live, done = task.step(live, f"choose_{live['new']}")
    assert not done and live["phase"] == "submit"
    live, done = task.step(live, "submit")
    assert done and live["terminal_return"] == 1.0


def test_nonce_visual_binding_uses_neutral_slot_actions(tmp_path):
    task = NonceVisualBindingTask()
    rows = build_visual_counterfactual_dataset([4], output_dir=tmp_path / "images", tasks=[task])
    choose = next(group for group in _group(rows).values() if group[0]["prefix"]["state"]["phase"] == "choose")
    assert len(choose) == 6
    assert all(row["action"].startswith("select_slot_") for row in choose)
    assert len(choose[0]["history_images"]) == 3
    assert choose[0]["memory_advantages"] == [1.0, 1.0, 0.0]
    best_q = max(row["Q"] for row in choose)
    correct = f"select_slot_{choose[0]['prefix']['state']['correct_slot'] + 1}"
    assert {row["action"] for row in choose if row["Q"] == best_q} == {correct}


def test_interference_chain_exposes_roles_separately_from_causal_signs(tmp_path):
    task = InterferenceChainTask()
    rows = build_visual_counterfactual_dataset([7], output_dir=tmp_path / "images", tasks=[task])
    choose = next(group for group in _group(rows).values() if group[0]["prefix"]["state"]["phase"] == "choose")
    assert len(choose) == 8
    assert len(choose[0]["history_images"]) == 5
    assert choose[0]["memory_roles"] == ["stale", "irrelevant", "stale", "irrelevant", "useful"]
    assert choose[0]["memory_advantage_source"] == "designer_role_prior"
    state = choose[0]["prefix"]["state"]
    assert state["entries"][-1]["code"] == state["target"]


def test_interference_chain_seeds_and_ood_templates_are_distinct(tmp_path):
    a = InterferenceChainTask(template="A").initial_state(100)
    b = InterferenceChainTask(template="B").initial_state(101)
    assert set(a["options"]) != set(b["options"])
    assert a["template"] == "A" and b["template"] == "B"
    task = InterferenceChainTask(template="D")
    path = tmp_path / "ood.png"
    task.render_screenshot(task.initial_state(500), path)
    assert Image.open(path).size == (960, 600)
