#!/usr/bin/env python3
"""Collect reliable system-toggle demonstrations with screenshot feedback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from trajflow_kv.actions import canonical_action
from trajflow_kv.androidworld_http import AndroidWorldHTTPClient
from trajflow_kv.rollout import append_jsonl, collect_rollout, stable_task_id


class AdaptiveTogglePolicy:
    def __init__(self, mode: str):
        self.mode = mode

    @staticmethod
    def _is_shade(image) -> bool:
        pixels = np.asarray(image)
        # The launcher is bright in this upper band; Quick Settings has a black
        # background and dark rounded tiles even with the pointer overlay.
        return float(pixels[140:430].mean()) < 115.0

    @staticmethod
    def _is_internet_dialog(image) -> bool:
        pixels = np.asarray(image)
        return float(pixels[500:1500, 80:1000].mean()) > 185.0

    def act(self, instruction, image, history, screen_size):  # noqa: ARG002
        last = json.loads(history[-1]) if history else None
        if self.mode == "wifi-off":
            if last and last.get("action_type") == "click" and last.get("x") == 870:
                return canonical_action(
                    {"action_type": "status", "goal_status": "complete"}
                )
            if last and last.get("action_type") == "click" and last.get("x") == 260:
                if self._is_internet_dialog(image):
                    return canonical_action({"action_type": "click", "x": 870, "y": 920})
                return canonical_action({"action_type": "swipe", "direction": "down"})
            if self._is_shade(image):
                return canonical_action({"action_type": "click", "x": 260, "y": 200})
            return canonical_action({"action_type": "swipe", "direction": "down"})

        if self._is_shade(image):
            if last and last.get("action_type") == "click" and last.get("x") == 780:
                return canonical_action(
                    {"action_type": "status", "goal_status": "complete"}
                )
            return canonical_action({"action_type": "click", "x": 780, "y": 200})
        return canonical_action({"action_type": "swipe", "direction": "down"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("wifi-off", "bluetooth-on"), required=True)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--server-url", default="http://127.0.0.1:5000")
    parser.add_argument("--output", required=True)
    parser.add_argument("--label", default="adaptive")
    args = parser.parse_args()
    task = "SystemWifiTurnOff" if args.mode == "wifi-off" else "SystemBluetoothTurnOn"
    client = AndroidWorldHTTPClient(args.server_url, timeout=300.0)
    instruction = client.goal(task, 0)
    task_id = stable_task_id(task, instruction, {"task_idx": 0})
    for rollout_index in range(args.rollouts):
        initialized = False
        try:
            client.reset(go_home=True)
            client.initialize_task(task, 0)
            initialized = True
            result = collect_rollout(
                policy=AdaptiveTogglePolicy(args.mode),
                instruction=instruction,
                task_id=task_id,
                max_steps=args.max_steps,
                image_dir=Path(args.output).parent / "images" /
                f"{task_id}-{args.label}-r{rollout_index:03d}",
                get_pixels=client.screenshot,
                screen_size=lambda: client.screen_size,
                execute=client.execute,
                evaluate=lambda: client.score(task, 0),
                rollout_metadata={
                    "policy": "adaptive-scripted",
                    "mode": args.mode,
                    "rollout_index": rollout_index,
                },
            )
            append_jsonl(args.output, [result])
            print(f"rollout={rollout_index} return={result.return_}", flush=True)
        finally:
            if initialized:
                client.tear_down_task(task, 0)


if __name__ == "__main__":
    main()
