from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol

from PIL import Image

from .actions import InvalidAction, canonical_action, parse_action


class Policy(Protocol):
    def act(self, instruction: str, image: Image.Image, history: list[str],
            screen_size: tuple[int, int]) -> str: ...


@dataclass
class RolloutResult:
    task_id: str
    instruction: str
    return_: float
    steps: list[dict[str, Any]]
    invalid_actions: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_record(self) -> dict[str, Any]:
        metadata = dict(self.metadata)
        metadata["invalid_actions"] = self.invalid_actions
        return {"task_id": self.task_id, "instruction": self.instruction,
                "return": self.return_, "steps": self.steps,
                "metadata": metadata}


def stable_task_id(task_name: str, instruction: str, params: Any) -> str:
    payload = json.dumps({"task": task_name, "goal": instruction, "params": repr(params)}, sort_keys=True)
    return f"{task_name}-{hashlib.sha256(payload.encode()).hexdigest()[:12]}"


def collect_rollout(
    *, policy: Policy, instruction: str, task_id: str, max_steps: int,
    image_dir: Path, get_pixels: Callable[[], Any],
    screen_size: Callable[[], tuple[int, int]],
    execute: Callable[[dict[str, Any]], None], evaluate: Callable[[], float],
    rollout_metadata: dict[str, Any] | None = None,
) -> RolloutResult:
    """Collect one trajectory using callbacks supplied by an environment adapter."""
    image_dir.mkdir(parents=True, exist_ok=True)
    history: list[str] = []
    steps: list[dict[str, Any]] = []
    invalid = 0
    for step_index in range(max_steps):
        image = Image.fromarray(get_pixels()).convert("RGB")
        image_path = (image_dir / f"{task_id}-{step_index:03d}.png").resolve()
        image.save(image_path)
        model_output = policy.act(instruction, image, history.copy(), screen_size())
        try:
            action = parse_action(model_output, screen_size())
        except InvalidAction as error:
            invalid += 1
            action = {"action_type": "wait"}
            raw = canonical_action(action)
            parse_error = str(error)
        else:
            raw = canonical_action(action)
            parse_error = None
        steps.append({"image": str(image_path), "screen_size": list(screen_size()), "history": history.copy(),
                      "action": raw, "model_output": model_output, "parse_error": parse_error})
        execute(action)
        history.append(raw)
        if action["action_type"] == "status":
            break
    return RolloutResult(
        task_id, instruction, float(evaluate()), steps, invalid,
        metadata=dict(rollout_metadata or {}),
    )


def append_jsonl(path: str | Path, results: list[RolloutResult]) -> None:
    output = Path(path); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("a", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(result.as_record(), ensure_ascii=False) + "\n")
