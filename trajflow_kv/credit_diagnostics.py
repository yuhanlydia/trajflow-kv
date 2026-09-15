"""Data-only diagnostics for visual delayed-consequence counterfactuals.

The diagnostics in this module deliberately do not change the policy loss.  A
counterfactual JSONL file contains one row per candidate action, so filtering
rows independently can silently destroy the same-prefix comparison.  The
helpers below always operate on complete prefix groups and add explicit
critical/non-critical annotations for downstream analysis.
"""

from __future__ import annotations

import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from .counterfactual import load_counterfactual_jsonl, write_counterfactual_jsonl


def group_counterfactual_rows(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows by prefix and validate that every candidate is present.

    Groups retain first-seen prefix order.  Within a group, rows follow the
    serialized ``candidate_actions`` order when available.  This mirrors the
    training path and makes it impossible for a critical-only filter to keep
    only the winning candidate and lose its counterfactual alternatives.
    """
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        prefix_id = str(row.get("prefix_id", ""))
        if not prefix_id:
            raise ValueError("counterfactual rows require prefix_id")
        grouped.setdefault(prefix_id, []).append(row)

    ordered: list[list[dict[str, Any]]] = []
    for prefix_id, group in grouped.items():
        candidates = group[0].get("candidate_actions")
        if candidates is None:
            candidates = [row["action"] for row in group]
        by_action: dict[str, dict[str, Any]] = {}
        for row in group:
            action = str(row["action"])
            if action in by_action:
                raise ValueError(f"prefix {prefix_id} contains duplicate action {action!r}")
            by_action[action] = row
        candidate_names = [str(action) for action in candidates]
        missing = [action for action in candidate_names if action not in by_action]
        if missing:
            raise ValueError(f"prefix {prefix_id} is missing candidate rows: {missing}")
        if len(set(candidate_names)) != len(candidate_names):
            raise ValueError(f"prefix {prefix_id} contains duplicate candidate_actions")
        ordered.append([by_action[action] for action in candidate_names])
    return ordered


def _critical_prefix(group: list[dict[str, Any]]) -> bool:
    """Return whether a prefix is a labeled consequential decision point."""
    return any(bool(row.get("critical_step")) for row in group) or any(
        bool(row.get("is_critical_action")) for row in group
    )


def _memory_source(row: dict[str, Any]) -> dict[str, Any] | None:
    """Infer the known visual-memory source for the controlled task families.

    The benchmark intentionally has a tiny, explicit ground truth.  Keeping
    this annotation in the diagnostic layer avoids adding task-specific logic
    to the generic trainer.  ``None`` means this prefix is not a known memory
    decision point.
    """
    family = str(row.get("task_family", ""))
    state = (row.get("prefix") or {}).get("state") or {}
    phase = state.get("phase")
    if family == "hidden_memory" and phase == "choose":
        return {"kind": "cue", "source_step": 0, "source_field": "cue"}
    if family in {"hidden_cue", "distractor_credit"} and phase == "fork":
        return {"kind": "target", "source_step": 0, "source_field": "target"}
    return None


def annotate_credit_ground_truth(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add explicit critical/non-critical and memory-source labels.

    The returned rows are deep copies.  Existing schema-v1 fields remain
    untouched; consumers that do not understand ``ground_truth`` can continue
    reading the original counterfactual fields.
    """
    groups = group_counterfactual_rows(rows)
    annotated: list[dict[str, Any]] = []
    for group in groups:
        critical_prefix = _critical_prefix(group)
        source = _memory_source(group[0])
        critical_actions = sorted({
            str(action)
            for row in group
            for action in row.get("critical_actions", [])
        })
        optimal_actions = sorted({
            str(action)
            for row in group
            for action in row.get("optimal_actions", [])
        })
        step_index = (group[0].get("prefix") or {}).get("step_index")
        for original in group:
            row = copy.deepcopy(original)
            action_critical = bool(original.get("is_critical_action", False))
            # ``critical_action`` is a row-level label.  For the visual task,
            # all candidates in a critical prefix are candidates at the
            # critical fork, even if the candidate itself is suboptimal.
            row["ground_truth"] = {
                "critical_prefix": critical_prefix,
                "critical_action": action_critical,
                "credit_role": "critical" if critical_prefix else "noncritical",
                "critical_actions": critical_actions,
                "optimal_actions": optimal_actions,
                "prefix_step": step_index,
                "memory_source": source,
            }
            row["critical_prefix"] = critical_prefix
            row["noncritical_prefix"] = not critical_prefix
            annotated.append(row)
    return annotated


def critical_prefix_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep *all* candidate rows belonging to critical prefixes."""
    annotated = annotate_credit_ground_truth(rows)
    groups = group_counterfactual_rows(annotated)
    return [row for group in groups if _critical_prefix(group) for row in group]


def balanced_memory_prefix_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Pair each critical prefix with one same-family all-zero memory negative."""
    annotated = annotate_credit_ground_truth(rows)
    groups = group_counterfactual_rows(annotated)
    critical_by_family: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    negative_by_family: dict[str, list[list[dict[str, Any]]]] = defaultdict(list)
    for group in groups:
        family = str(group[0].get("task_family", "unknown"))
        if _critical_prefix(group):
            critical_by_family[family].append(group)
            continue
        history = group[0].get("history_images", [])
        advantages = group[0].get("memory_advantages", [])
        if history and advantages and len(history) == len(advantages) and not any(advantages):
            negative_by_family[family].append(group)
    selected: list[list[dict[str, Any]]] = []
    for family in sorted(critical_by_family):
        positives = critical_by_family[family]
        negatives = negative_by_family[family]
        if len(negatives) < len(positives):
            raise ValueError(f"not enough memory-negative prefixes for {family}")
        selected.extend(positives)
        selected.extend(negatives[: len(positives)])
    return [row for group in selected for row in group]


def _oracle_text(row: dict[str, Any]) -> str | None:
    family = str(row.get("task_family", ""))
    state = (row.get("prefix") or {}).get("state") or {}
    phase = state.get("phase")
    if family in {"hidden_memory", "hidden_cue"} and phase == "choose":
        cue = state.get("cue")
        if cue is not None:
            return f"History oracle: the earlier hidden cue color was {str(cue).upper()}."
    if family == "distractor_credit" and phase == "fork":
        target = state.get("target")
        if target is not None:
            return f"History oracle: the earlier target option was {str(target).upper()}."
    return None


def _prompt_with_observation(row: dict[str, Any]) -> str:
    """Render the canonical text prompt, including a transformed observation."""
    prefix = row.get("prefix") or {}
    observation = prefix.get("observation") or {}
    screen = observation.get("screen", prefix.get("screen", ""))
    history = prefix.get("history", [])
    candidates = row.get("candidate_actions", [row.get("action")])
    return (
        "You are a GUI policy. Choose exactly one action from the candidate list "
        "and return only that action, with no explanation.\n"
        f"Task: {row.get('instruction', '')}\n"
        f"Observation: {screen}\n"
        f"History: {json.dumps(history, ensure_ascii=False)}\n"
        f"Candidates: {json.dumps([str(item) for item in candidates], ensure_ascii=False)}"
    )


def add_history_oracle(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return rows whose choose/fork observation explicitly reveals its cue.

    This is an *oracle diagnostic*: the screenshot is intentionally left
    unchanged, while the textual observation and policy prompt receive the
    hidden target.  It tests whether the VLM/policy path can use the evidence
    once memory retrieval is no longer the bottleneck.
    """
    transformed: list[dict[str, Any]] = []
    for original in annotate_credit_ground_truth(rows):
        row = copy.deepcopy(original)
        oracle = _oracle_text(row)
        if oracle is None:
            transformed.append(row)
            continue
        prefix = row.setdefault("prefix", {})
        observation = prefix.setdefault("observation", {})
        screen = str(observation.get("screen", ""))
        if oracle not in screen:
            observation["screen"] = f"{screen}\n{oracle}" if screen else oracle
        prefix["history_oracle"] = oracle
        row["history_oracle"] = oracle
        # Rebuild rather than append to an existing visual-only prompt so the
        # oracle text is guaranteed to be visible to text and multimodal paths.
        row["prompt"] = _prompt_with_observation(row)
        transformed.append(row)
    return transformed


def summarize_credit_rows(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Summarize critical/non-critical ground truth without a model run."""
    annotated = annotate_credit_ground_truth(rows)
    groups = group_counterfactual_rows(annotated)
    by_family: dict[str, dict[str, int]] = defaultdict(
        lambda: {"prefixes": 0, "critical_prefixes": 0, "noncritical_prefixes": 0,
                 "candidate_rows": 0, "critical_rows": 0, "noncritical_rows": 0}
    )
    memory_sources: dict[str, int] = defaultdict(int)
    for group in groups:
        family = str(group[0].get("task_family", "unknown"))
        summary = by_family[family]
        critical = _critical_prefix(group)
        summary["prefixes"] += 1
        summary["candidate_rows"] += len(group)
        summary["critical_prefixes"] += int(critical)
        summary["noncritical_prefixes"] += int(not critical)
        summary["critical_rows"] += len(group) if critical else 0
        summary["noncritical_rows"] += 0 if critical else len(group)
        source = _memory_source(group[0])
        if source is not None:
            memory_sources[str(source["kind"])] += 1
    critical_prefixes = sum(item["critical_prefixes"] for item in by_family.values())
    return {
        "rows": sum(len(group) for group in groups),
        "prefixes": len(groups),
        "critical_prefixes": critical_prefixes,
        "noncritical_prefixes": len(groups) - critical_prefixes,
        "critical_rows": sum(item["critical_rows"] for item in by_family.values()),
        "noncritical_rows": sum(item["noncritical_rows"] for item in by_family.values()),
        "complete_candidate_groups": True,
        "memory_source_prefixes": dict(sorted(memory_sources.items())),
        "families": {family: values for family, values in sorted(by_family.items())},
    }


def prepare_diagnostic_files(
    input_path: str | Path,
    *,
    critical_output: str | Path | None = None,
    history_oracle_output: str | Path | None = None,
    balanced_memory_output: str | Path | None = None,
    summary_output: str | Path | None = None,
) -> dict[str, Any]:
    """Create requested P0 diagnostic JSONL files and return their summary."""
    rows = load_counterfactual_jsonl(input_path)
    summary = summarize_credit_rows(rows)
    summary["input"] = str(input_path)
    if critical_output is not None:
        critical = critical_prefix_rows(rows)
        summary["critical_output"] = str(critical_output)
        summary["critical_output_rows"] = write_counterfactual_jsonl(critical, critical_output)
    if history_oracle_output is not None:
        oracle = add_history_oracle(rows)
        summary["history_oracle_output"] = str(history_oracle_output)
        summary["history_oracle_rows"] = write_counterfactual_jsonl(oracle, history_oracle_output)
    if balanced_memory_output is not None:
        balanced = balanced_memory_prefix_rows(rows)
        summary["balanced_memory_output"] = str(balanced_memory_output)
        summary["balanced_memory_rows"] = write_counterfactual_jsonl(
            balanced, balanced_memory_output
        )
    if summary_output is not None:
        destination = Path(summary_output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary
