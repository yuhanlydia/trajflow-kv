import json

import numpy as np

from trajflow_kv.actions import InvalidAction, parse_action
from trajflow_kv.rollout import collect_rollout


class SequencePolicy:
    def __init__(self, outputs): self.outputs = iter(outputs)
    def act(self, instruction, image, history, screen_size): return next(self.outputs)


def test_action_parser_normalizes_coordinates_and_fences():
    action = parse_action('```json\n{"action_type":"click","x":0.5,"y":0.25}\n```', (100, 200))
    assert action == {"action_type": "click", "x": 50, "y": 50}


def test_action_parser_rejects_invalid_scroll():
    try:
        parse_action('{"action_type":"scroll","direction":"diagonal"}', (100, 200))
    except InvalidAction:
        pass
    else:
        raise AssertionError("invalid direction accepted")


def test_aitw_two_point_swipe_converts_to_androidworld_direction():
    action = parse_action(
        '{"action_type":"swipe","coordinate_1":[256,598],"coordinate_2":[256,250]}',
        (512, 768),
    )
    assert action == {"action_type": "swipe", "direction": "up"}


def test_parser_uses_first_json_and_tolerates_trailing_reasoning():
    action = parse_action(
        '{"action_type":"click","x":10,"y":20}\nI clicked the button.\n'
        '{"action_type":"wait"}',
        (100, 200),
    )
    assert action == {"action_type": "click", "x": 10, "y": 20}


def test_action_parser_repairs_bare_key_and_terminate_alias():
    assert parse_action('{"action_type":"click","x":125, y:330}', (1080, 2400)) == {
        "action_type": "click", "x": 125, "y": 330,
    }
    assert parse_action(
        '{"action_type":"terminate","status":"success"}', (1080, 2400)
    ) == {"action_type": "status", "goal_status": "complete"}


def test_action_parser_normalizes_common_android_aliases():
    assert parse_action(
        '{"action_type":"swipe","coordinate":[10,90],"coordinate2":[10,20]}',
        (100, 100),
    ) == {"action_type": "swipe", "direction": "up"}
    assert parse_action(
        '{"action_type":"system_button","button":"Home"}', (100, 100)
    ) == {"action_type": "navigate_home"}


def test_mixed_return_rollout_schema(tmp_path):
    executed = []
    policy = SequencePolicy(["not json", '{"action_type":"status","goal_status":"complete"}'])
    result = collect_rollout(
        policy=policy, instruction="demo", task_id="task-a", max_steps=4,
        image_dir=tmp_path / "images",
        get_pixels=lambda: np.zeros((20, 10, 3), dtype=np.uint8),
        screen_size=lambda: (10, 20), execute=executed.append,
        evaluate=lambda: 1.0,
        rollout_metadata={"seed": 7},
    )
    record = result.as_record()
    assert record["return"] == 1.0 and len(record["steps"]) == 2
    assert record["metadata"]["invalid_actions"] == 1
    assert record["metadata"]["seed"] == 7
    assert executed[0] == {"action_type": "wait"}
    assert record["steps"][0]["model_output"] == "not json"
    assert json.loads(record["steps"][1]["action"])["action_type"] == "status"
