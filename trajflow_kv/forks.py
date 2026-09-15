from __future__ import annotations

import json
from collections import defaultdict

from .actions import canonical_action


def _canonical(value: str) -> str:
    return canonical_action(json.loads(value))


def build_fork_pairs(
    trajectories: list[dict],
    task_prefix: str | None = None,
    max_per_decision: int = 4,
    chosen_action_type: str | None = None,
    rejected_action_type: str | None = None,
) -> list[dict]:
    """Place failed actions into successful states at the aligned step index.

    These are counterfactual preference examples, not claims that the failed
    action was actually executed from the chosen screenshot.
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for trajectory in trajectories:
        if task_prefix and not trajectory["task_id"].startswith(task_prefix):
            continue
        grouped[trajectory["task_id"]].append(trajectory)

    pairs = []
    for task_id, records in grouped.items():
        positives = [record for record in records if float(record["return"]) > 0]
        negatives = [record for record in records if float(record["return"]) <= 0]
        seen: dict[tuple, int] = defaultdict(int)
        for positive in positives:
            for step_index, chosen_step in enumerate(positive["steps"]):
                chosen = _canonical(chosen_step["action"])
                if chosen_action_type and json.loads(chosen)["action_type"] != chosen_action_type:
                    continue
                for negative in negatives:
                    if step_index >= len(negative["steps"]):
                        continue
                    rejected = _canonical(negative["steps"][step_index]["action"])
                    if rejected == chosen:
                        continue
                    if rejected_action_type and json.loads(rejected)["action_type"] != rejected_action_type:
                        continue
                    key = (task_id, step_index, chosen, rejected)
                    if seen[key] >= max_per_decision:
                        continue
                    seen[key] += 1
                    pairs.append({
                        "task_id": task_id,
                        "instruction": positive["instruction"],
                        "step_index": step_index,
                        "image": chosen_step.get("image"),
                        "screen_size": chosen_step.get("screen_size", [1000, 1000]),
                        "history": chosen_step.get("history", []),
                        "chosen": chosen,
                        "rejected": rejected,
                    })
    return pairs


def build_coordinate_fork_pairs(
    trajectories: list[dict], offsets: tuple[int, ...] = (140, 300)
) -> list[dict]:
    """Create click-coordinate hard negatives around successful actions."""
    pairs = []
    seen = set()
    for trajectory in trajectories:
        if float(trajectory["return"]) <= 0:
            continue
        for step_index, step in enumerate(trajectory["steps"]):
            chosen_dict = json.loads(_canonical(step["action"]))
            if chosen_dict.get("action_type") != "click":
                continue
            x, y = int(chosen_dict["x"]), int(chosen_dict["y"])
            for dx, dy in [(value, 0) for value in offsets] + [
                (-value, 0) for value in offsets
            ] + [(0, value) for value in offsets] + [
                (0, -value) for value in offsets
            ]:
                rejected_dict = dict(chosen_dict)
                rejected_dict["x"] = min(999, max(0, x + dx))
                rejected_dict["y"] = min(999, max(0, y + dy))
                rejected = canonical_action(rejected_dict)
                chosen = canonical_action(chosen_dict)
                key = (trajectory["task_id"], step_index, chosen, rejected)
                if rejected == chosen or key in seen:
                    continue
                seen.add(key)
                pairs.append({
                    "task_id": trajectory["task_id"],
                    "instruction": trajectory["instruction"],
                    "step_index": step_index,
                    "image": step.get("image"),
                    "screen_size": step.get("screen_size", [1000, 1000]),
                    "history": step.get("history", []),
                    "chosen": chosen,
                    "rejected": rejected,
                })
    return pairs
