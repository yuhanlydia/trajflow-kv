#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajflow_kv.data import load_jsonl
from trajflow_kv.forks import build_coordinate_fork_pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--offsets", type=int, nargs="+", default=[140, 300])
    parser.add_argument("--task-prefix")
    args = parser.parse_args()
    trajectories = load_jsonl(args.input)
    if args.task_prefix:
        trajectories = [
            item for item in trajectories if item["task_id"].startswith(args.task_prefix)
        ]
    pairs = build_coordinate_fork_pairs(trajectories, tuple(args.offsets))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(pair) + "\n" for pair in pairs), encoding="utf-8"
    )
    print(json.dumps({"pairs": len(pairs)}, indent=2))


if __name__ == "__main__":
    main()
