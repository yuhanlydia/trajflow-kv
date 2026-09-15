#!/usr/bin/env python3
"""Paired online statistics with task-clustered bootstrap and exact McNemar."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajflow_kv.statistics import summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSONL with task_id, baseline, candidate")
    source.add_argument("--baseline", help="MiniWoB evaluator JSON used with --candidate")
    parser.add_argument("--candidate", help="MiniWoB evaluator JSON used with --baseline")
    parser.add_argument("--output", required=True)
    parser.add_argument("--samples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.input:
        rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    else:
        if not args.candidate:
            parser.error("--baseline requires --candidate")
        baseline = json.loads(Path(args.baseline).read_text())["rows"]
        candidate = json.loads(Path(args.candidate).read_text())["rows"]
        left = {(row["task"], row["seed"]): row for row in baseline}
        right = {(row["task"], row["seed"]): row for row in candidate}
        if set(left) != set(right):
            parser.error("baseline and candidate episode keys differ")
        rows = [{"task_id": task, "seed": seed, "baseline": left[(task, seed)]["return"],
                 "candidate": right[(task, seed)]["return"]}
                for task, seed in sorted(left)]
    result = summarize(rows, args.samples, args.seed)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
