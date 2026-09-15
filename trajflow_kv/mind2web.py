from __future__ import annotations

import json
import xml.etree.ElementTree as ET


def describe_candidate(root: ET.Element, candidate: dict, max_words: int = 24) -> str:
    """Create a compact, deterministic element description from a Mind2Web node."""
    node_id = str(candidate["backend_node_id"])
    node = next(
        (item for item in root.iter() if item.attrib.get("backend_node_id") == node_id),
        None,
    )
    tag = candidate.get("tag") or (node.tag if node is not None else "element")
    words: list[str] = []
    if node is not None:
        for key in ("aria-label", "title", "placeholder", "value", "name"):
            if node.attrib.get(key):
                words.extend(node.attrib[key].split())
        words.extend(" ".join(node.itertext()).split())
    if not words:
        try:
            attributes = json.loads(candidate.get("attributes", "{}"))
        except json.JSONDecodeError:
            attributes = {}
        for key in ("aria-label", "title", "placeholder", "value", "name", "id"):
            if attributes.get(key):
                words.extend(str(attributes[key]).split())
    text = " ".join(words[:max_words]).strip() or "unlabeled"
    return f"[{tag}] {text}"


def format_target(description: str, operation: dict) -> str:
    target = f"Element: {description}\nAction: {operation['op']}"
    if operation["op"] != "CLICK" and operation.get("value"):
        target += f"\nValue: {operation['value']}"
    return target


def build_prompt(task: str, previous_actions: list[str], choices: list[str]) -> str:
    history = "\n".join(previous_actions[-5:]) if previous_actions else "None"
    options = "\n".join(f"{index + 1}. {choice}" for index, choice in enumerate(choices))
    return (
        "Choose the webpage element and operation that best advances the task.\n"
        f"Task: {task}\nPrevious actions:\n{history}\nCandidate elements:\n{options}\n"
        "Return exactly `Element: ...` followed by `Action: ...` and an optional `Value: ...`."
    )
