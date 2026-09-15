#!/usr/bin/env python3
"""Estimate policy-specific Q(M)-Q(M^-j) for visual history blocks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch

try:
    from scripts.evaluate_counterfactual_qwen import _groups, _score_groups
except ModuleNotFoundError:  # Direct ``python scripts/...`` execution.
    from evaluate_counterfactual_qwen import _groups, _score_groups
from trajflow_kv.counterfactual import load_counterfactual_jsonl, write_counterfactual_jsonl


def expected_continuation_return(group: list[dict], temperature: float = 1.0) -> float:
    scores = torch.tensor([float(row["score"]) for row in group], dtype=torch.float64)
    q_values = torch.tensor([float(row.get("Q", 0.0)) for row in group], dtype=torch.float64)
    probabilities = torch.softmax(scores / temperature, dim=0)
    return float((probabilities * q_values).sum())


def annotate_memory_advantages(
    rows: list[dict],
    full_groups: list[list[dict]],
    dropped_by_index: dict[int, list[list[dict]]],
    *,
    temperature: float = 1.0,
) -> tuple[list[dict], dict]:
    groups = _groups(rows)
    if len(groups) != len(full_groups):
        raise ValueError("full score groups do not match data groups")
    advantages_by_prefix: dict[str, list[float]] = {}
    records = []
    for group_index, (data_group, full_group) in enumerate(zip(groups, full_groups, strict=True)):
        prefix_id = str(data_group[0]["prefix_id"])
        history_count = len(data_group[0].get("history_images", []))
        q_full = expected_continuation_return(full_group, temperature)
        advantages = []
        for history_index in range(history_count):
            dropped_groups = dropped_by_index.get(history_index)
            if dropped_groups is None or len(dropped_groups) != len(groups):
                raise ValueError(f"missing dropped scores for history index {history_index}")
            q_drop = expected_continuation_return(dropped_groups[group_index], temperature)
            advantages.append(q_full - q_drop)
            records.append({
                "prefix_id": prefix_id,
                "history_index": history_index,
                "Q_memory_full": q_full,
                "Q_memory_dropped": q_drop,
                "memory_advantage": q_full - q_drop,
            })
        advantages_by_prefix[prefix_id] = advantages

    annotated = []
    for row in rows:
        item = copy.deepcopy(row)
        item["memory_advantages"] = advantages_by_prefix[str(row["prefix_id"])]
        item["memory_advantage_source"] = "qwen_policy_counterfactual_q"
        item.setdefault("prefix", {})["memory_advantages"] = list(item["memory_advantages"])
        annotated.append(item)
    values = [record["memory_advantage"] for record in records]
    summary = {
        "prefixes": len(groups),
        "history_blocks": len(records),
        "mean_memory_advantage": sum(values) / len(values) if values else None,
        "mean_absolute_memory_advantage": (
            sum(abs(value) for value in values) / len(values) if values else None
        ),
        "positive_blocks": sum(value > 0 for value in values),
        "negative_blocks": sum(value < 0 for value in values),
        "zero_blocks": sum(value == 0 for value in values),
        "records": records,
    }
    return annotated, summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = load_counterfactual_jsonl(args.data)
    groups = _groups(rows)
    max_history = max((len(group[0].get("history_images", [])) for group in groups), default=0)
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    ).eval()
    full = _score_groups(model, processor, groups, args.device)
    dropped = {
        index: _score_groups(model, processor, groups, args.device, drop_history_index=index)
        for index in range(max_history)
    }
    annotated, summary = annotate_memory_advantages(
        rows, full, dropped, temperature=args.temperature
    )
    write_counterfactual_jsonl(annotated, args.output)
    summary.update({
        "model": args.model,
        "data": args.data,
        "output": args.output,
        "temperature": args.temperature,
    })
    destination = Path(args.summary_output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key != "records"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
