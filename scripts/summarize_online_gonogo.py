#!/usr/bin/env python3
"""Summarize paired online go/no-go rollout files across tasks and seeds."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path


def load_policy(root: Path, policy: str) -> dict[tuple[str, int], dict]:
    rows = {}
    pattern = re.compile(rf"gonogo_online_(.+)_{re.escape(policy)}_s(\d+)\.jsonl$")
    for path in root.glob(f"gonogo_online_*_{policy}_s*.jsonl"):
        match = pattern.match(path.name)
        if not match:
            continue
        task, seed = match.group(1), int(match.group(2))
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if len(records) != 1:
            raise ValueError(f"expected exactly one rollout in {path}, got {len(records)}")
        rows[(task, seed)] = records[0]
    return rows


def summarize_pairs(pairs: list[tuple[dict, dict]]) -> dict:
    before = [float(left["return"]) for left, _ in pairs]
    after = [float(right["return"]) for _, right in pairs]
    improved = sum(x < y for x, y in zip(before, after))
    regressed = sum(x > y for x, y in zip(before, after))
    return {
        "pairs": len(pairs),
        "baseline_successes": sum(before),
        "candidate_successes": sum(after),
        "baseline_success_rate": sum(before) / len(before),
        "candidate_success_rate": sum(after) / len(after),
        "success_rate_delta": (sum(after) - sum(before)) / len(before),
        "improved_pairs": improved,
        "regressed_pairs": regressed,
        "unchanged_pairs": len(pairs) - improved - regressed,
        "baseline_invalid_actions": sum(
            int(left.get("metadata", {}).get("invalid_actions", 0)) for left, _ in pairs
        ),
        "candidate_invalid_actions": sum(
            int(right.get("metadata", {}).get("invalid_actions", 0)) for _, right in pairs
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/androidworld")
    parser.add_argument("--baseline-policy", default="initial")
    parser.add_argument("--candidate-policy", default="return_no_energy")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline = load_policy(Path(args.data_dir), args.baseline_policy)
    candidate = load_policy(Path(args.data_dir), args.candidate_policy)
    keys = sorted(set(baseline) & set(candidate))
    if not keys:
        raise ValueError("no paired rollout files found")
    missing = sorted(set(baseline) ^ set(candidate))
    if missing:
        raise ValueError(f"unpaired task/seed records: {missing}")

    task_pairs: dict[str, list[tuple[dict, dict]]] = defaultdict(list)
    all_pairs = []
    for task, seed in keys:
        pair = (baseline[(task, seed)], candidate[(task, seed)])
        task_pairs[task].append(pair)
        all_pairs.append(pair)
    summary = {
        "baseline_policy": args.baseline_policy,
        "candidate_policy": args.candidate_policy,
        "overall": summarize_pairs(all_pairs),
        "tasks": {task: summarize_pairs(pairs) for task, pairs in sorted(task_pairs.items())},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
