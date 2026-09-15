#!/usr/bin/env python3
"""Validate trajectory schema and mixed-return coverage before RL training."""
import argparse
from collections import defaultdict

from trajflow_kv.data import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    records = load_jsonl(args.path)
    groups = defaultdict(list)
    for record in records:
        groups[record["task_id"]].append(float(record["return"]))
    mixed = {task: values for task, values in groups.items() if min(values) < max(values)}
    print(f"trajectories={len(records)} task_groups={len(groups)} mixed_groups={len(mixed)}")
    if not mixed:
        raise SystemExit("no mixed-return task group; pure return gradient would be zero")
    for task, values in mixed.items():
        print(f"{task}: {values}")


if __name__ == "__main__":
    main()
