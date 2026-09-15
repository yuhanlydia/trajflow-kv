#!/usr/bin/env python3
"""Aggregate paired MiniWoB result files and audit action-level agreement."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def margins(row: dict) -> list[float]:
    values = []
    for step in row["steps"]:
        scores = sorted(step.get("scores", []), reverse=True)
        if len(scores) > 1:
            values.append(scores[0] - scores[1])
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", nargs="+", required=True)
    parser.add_argument("--candidate", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base_rows, candidate_rows = [], []
    for base_path in args.baseline:
        base_rows.extend(json.loads(Path(base_path).read_text())["rows"])
    for candidate_path in args.candidate:
        candidate_rows.extend(json.loads(Path(candidate_path).read_text())["rows"])
    base = {(row["task"], row["seed"]): row for row in base_rows}
    candidate = {(row["task"], row["seed"]): row for row in candidate_rows}
    if base.keys() != candidate.keys():
        raise ValueError("paired result files contain different task/seed keys")
    paired = []
    all_base_margins, all_candidate_margins = [], []
    for key in sorted(base):
        left, right = base[key], candidate[key]
        left_actions = [step.get("action") for step in left["steps"]]
        right_actions = [step.get("action") for step in right["steps"]]
        left_margins, right_margins = margins(left), margins(right)
        all_base_margins.extend(left_margins)
        all_candidate_margins.extend(right_margins)
        paired.append({
            "task": key[0], "seed": key[1],
            "baseline_return": left["return"], "candidate_return": right["return"],
            "actions_identical": left_actions == right_actions,
        })
    mean = lambda values: sum(values) / len(values) if values else None
    summary = {
        "pairs": len(paired),
        "baseline_successes": sum(row["baseline_return"] > 0 for row in paired),
        "candidate_successes": sum(row["candidate_return"] > 0 for row in paired),
        "improved_pairs": sum(row["candidate_return"] > row["baseline_return"] for row in paired),
        "regressed_pairs": sum(row["candidate_return"] < row["baseline_return"] for row in paired),
        "identical_action_episodes": sum(row["actions_identical"] for row in paired),
        "baseline_mean_top1_margin": mean(all_base_margins),
        "candidate_mean_top1_margin": mean(all_candidate_margins),
        "rows": paired,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
