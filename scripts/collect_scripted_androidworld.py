#!/usr/bin/env python3
"""Collect AndroidWorld trajectories from an explicit UI action sequence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from trajflow_kv.androidworld_http import AndroidWorldHTTPClient
from trajflow_kv.actions import canonical_action
from trajflow_kv.rollout import append_jsonl, collect_rollout, stable_task_id


class ScriptedPolicy:
    def __init__(self, actions: list[dict]):
        self.actions = actions
        self.index = 0

    def act(self, instruction, image, history, screen_size):  # noqa: ARG002
        if self.index >= len(self.actions):
            return canonical_action({"action_type": "wait"})
        action = self.actions[self.index]
        self.index += 1
        return canonical_action(action)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:5000")
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--actions-json", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="scripted")
    parser.add_argument(
        "--fast-screenshots",
        action="store_true",
        help="Skip the server's expensive screenshot stabilization delay.",
    )
    parser.add_argument(
        "--post-action-delay",
        type=float,
        default=0.0,
        help="Pause after execution without adding artificial wait actions to the trajectory.",
    )
    args = parser.parse_args()

    actions = json.loads(args.actions_json)
    if not isinstance(actions, list) or not all(isinstance(action, dict) for action in actions):
        raise ValueError("actions-json must decode to a list of action objects")
    client = AndroidWorldHTTPClient(args.server_url, timeout=args.http_timeout)
    instruction = client.goal(args.task, args.task_index)
    task_id = stable_task_id(args.task, instruction, {"task_idx": args.task_index})
    initialized = False
    try:
        client.reset(go_home=True)
        client.initialize_task(args.task, args.task_index)
        initialized = True
        result = collect_rollout(
            policy=ScriptedPolicy(actions),
            instruction=instruction,
            task_id=task_id,
            max_steps=len(actions),
            image_dir=Path(args.output).parent / "images" / f"{task_id}-{args.label}",
            get_pixels=lambda: client.screenshot(
                wait_to_stabilize=not args.fast_screenshots
            ),
            screen_size=lambda: client.screen_size,
            execute=lambda action: (
                client.execute(action),
                time.sleep(args.post_action_delay),
            ),
            evaluate=lambda: client.score(args.task, args.task_index),
            rollout_metadata={"policy": "scripted", "label": args.label},
        )
        append_jsonl(args.output, [result])
        print(f"return={result.return_} invalid={result.invalid_actions} output={args.output}")
    finally:
        if initialized:
            client.tear_down_task(args.task, args.task_index)


if __name__ == "__main__":
    main()
