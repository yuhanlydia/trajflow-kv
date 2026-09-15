#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajflow_kv.data import load_jsonl
from trajflow_kv.forks import build_fork_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--task-prefix")
    parser.add_argument("--max-per-decision", type=int, default=4)
    parser.add_argument("--chosen-action-type")
    parser.add_argument("--rejected-action-type")
    args = parser.parse_args()
    pairs = build_fork_pairs(
        load_jsonl(args.input), args.task_prefix, args.max_per_decision,
        args.chosen_action_type, args.rejected_action_type,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(pair) + "\n" for pair in pairs), encoding="utf-8"
    )
    task_counts = {}
    for pair in pairs:
        task_counts[pair["task_id"]] = task_counts.get(pair["task_id"], 0) + 1
    print(json.dumps({"pairs": len(pairs), "tasks": task_counts}, indent=2))


if __name__ == "__main__":
    main()
