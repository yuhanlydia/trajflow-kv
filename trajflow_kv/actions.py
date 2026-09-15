from __future__ import annotations

import json
import re
from typing import Any


ACTION_TYPES = {
    "click", "double_tap", "input_text", "keyboard_enter", "long_press",
    "navigate_back", "navigate_home", "open_app", "scroll", "swipe",
    "status", "wait", "answer",
}
DIRECTIONS = {"left", "right", "down", "up"}
ALLOWED_KEYS = {
    "action_type", "index", "x", "y", "text", "direction", "app_name",
    "goal_status", "keycode", "clear_text",
}


class InvalidAction(ValueError):
    pass


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S | re.I)
    candidate = fenced.group(1) if fenced else text
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start = text.find("{")
        if start < 0:
            raise InvalidAction("model output contains no JSON object")
        # Small VLMs occasionally emit otherwise valid JSON with an unquoted
        # key (for example ``{"x": 12, y: 34}``). Repair keys only at object
        # boundaries; values and free-form input text remain untouched.
        candidate = re.sub(
            r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)',
            r'\1"\2"\3',
            text[start:],
        )
        try:
            # Decode the first complete object and tolerate trailing reasoning
            # or a second proposal. The executed canonical action is recorded
            # separately from the untouched model output.
            value, _ = json.JSONDecoder().raw_decode(candidate)
        except json.JSONDecodeError as error:
            raise InvalidAction(f"invalid action JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise InvalidAction("action must be a JSON object")
    return value


def parse_action(text: str, screen_size: tuple[int, int]) -> dict[str, Any]:
    """Parse and validate model output against AndroidWorld JSONAction."""
    raw = _extract_json(text)
    # Normalize common Qwen/AITW aliases before strict schema validation.
    aliases = {"tap": "click", "back": "navigate_back", "home": "navigate_home"}
    if raw.get("action_type") in aliases:
        raw["action_type"] = aliases[raw["action_type"]]
    if raw.get("action_type") == "terminate":
        status = str(raw.pop("status", "success")).lower()
        raw["action_type"] = "status"
        raw["goal_status"] = (
            "complete" if status in {"success", "complete", "done"} else "infeasible"
        )
    if raw.get("action_type") == "system_button":
        button = str(raw.pop("button", "")).lower()
        system_aliases = {"home": "navigate_home", "back": "navigate_back"}
        if button in system_aliases:
            raw["action_type"] = system_aliases[button]
    if "coordinate1" in raw:
        raw["coordinate_1"] = raw.pop("coordinate1")
    if "coordinate2" in raw:
        raw["coordinate_2"] = raw.pop("coordinate2")
        if "coordinate" in raw and "coordinate_1" not in raw:
            raw["coordinate_1"] = raw.pop("coordinate")
    # Qwen/AITW commonly emits point or two-point coordinate fields. Convert
    # those at the environment boundary to AndroidWorld's JSONAction schema.
    if "coordinate" in raw and isinstance(raw["coordinate"], list) and len(raw["coordinate"]) == 2:
        raw["x"], raw["y"] = raw.pop("coordinate")
    if "coordinate_1" in raw and "coordinate_2" in raw:
        start, end = raw.pop("coordinate_1"), raw.pop("coordinate_2")
        if isinstance(start, list) and start and isinstance(start[0], list):
            start = start[0]
        if isinstance(end, list) and end and isinstance(end[0], list):
            end = end[0]
        if raw.get("action_type") != "swipe" or len(start) != 2 or len(end) != 2:
            raise InvalidAction("two-point coordinates are only valid for swipe")
        dx, dy = end[0] - start[0], end[1] - start[1]
        raw["direction"] = ("right" if dx > 0 else "left") if abs(dx) > abs(dy) else ("down" if dy > 0 else "up")
    unknown = set(raw) - ALLOWED_KEYS
    if unknown:
        raise InvalidAction(f"unknown action keys: {sorted(unknown)}")
    action_type = raw.get("action_type")
    if action_type not in ACTION_TYPES:
        raise InvalidAction(f"unsupported action_type: {action_type!r}")
    width, height = screen_size
    if action_type in {"click", "double_tap", "long_press"} and "index" not in raw:
        if not all(key in raw for key in ("x", "y")):
            raise InvalidAction(f"{action_type} requires index or x/y")
    for key, limit in (("x", width), ("y", height)):
        if key in raw:
            if isinstance(raw[key], bool) or not isinstance(raw[key], (int, float)):
                raise InvalidAction(f"{key} must be numeric")
            # Accept normalized coordinates, but serialize absolute integer pixels.
            value = raw[key] * limit if isinstance(raw[key], float) and 0 <= raw[key] <= 1 else raw[key]
            raw[key] = min(max(int(round(value)), 0), limit - 1)
    if "index" in raw:
        raw["index"] = int(raw["index"])
    if action_type in {"scroll", "swipe"} and raw.get("direction") not in DIRECTIONS:
        raise InvalidAction(f"{action_type} requires direction: left/right/down/up")
    if action_type == "input_text" and not isinstance(raw.get("text"), str):
        raise InvalidAction("input_text requires string text")
    return raw


def canonical_action(action: dict[str, Any]) -> str:
    return json.dumps(action, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
