#!/usr/bin/env python3
"""Create a paired summary for two controlled rollout JSONL files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    baseline_path, candidate_path = Path(args.baseline), Path(args.candidate)
    baseline, candidate = load(baseline_path), load(candidate_path)
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("paired evaluation requires equal non-empty rollout counts")
    for index, (left, right) in enumerate(zip(baseline, candidate)):
        if left["task_id"] != right["task_id"]:
            raise ValueError(f"task mismatch in pair {index}")
        left_meta, right_meta = left.get("metadata", {}), right.get("metadata", {})
        for key in ("seed", "temperature", "rollout_index"):
            if left_meta.get(key) != right_meta.get(key):
                raise ValueError(f"metadata mismatch for {key} in pair {index}")

    before = [float(row["return"]) for row in baseline]
    after = [float(row["return"]) for row in candidate]
    improved = sum(x < y for x, y in zip(before, after))
    regressed = sum(x > y for x, y in zip(before, after))
    summary = {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "pairs": len(before),
        "baseline_returns": before,
        "candidate_returns": after,
        "baseline_success_rate": sum(before) / len(before),
        "candidate_success_rate": sum(after) / len(after),
        "success_rate_delta": (sum(after) - sum(before)) / len(before),
        "improved_pairs": improved,
        "regressed_pairs": regressed,
        "unchanged_pairs": len(before) - improved - regressed,
        "baseline_invalid_actions": sum(
            int(row.get("metadata", {}).get("invalid_actions", 0)) for row in baseline
        ),
        "candidate_invalid_actions": sum(
            int(row.get("metadata", {}).get("invalid_actions", 0)) for row in candidate
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
