#!/usr/bin/env python3
"""Create deterministic return-stratified train/held-out rollout splits."""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--train-output", required=True)
    parser.add_argument("--heldout-output", required=True)
    parser.add_argument("--heldout-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if not 0 < args.heldout_fraction < 1:
        raise ValueError("heldout-fraction must be in (0, 1)")

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line.strip()]
    groups: dict[tuple[str, float], list[dict]] = defaultdict(list)
    for row in rows:
        groups[(row["task_id"], float(row["return"]))].append(row)

    generator = random.Random(args.seed)
    train, heldout = [], []
    for key in sorted(groups):
        group = groups[key]
        generator.shuffle(group)
        count = max(1, round(len(group) * args.heldout_fraction))
        if len(group) > 1:
            count = min(count, len(group) - 1)
        heldout.extend(group[:count])
        train.extend(group[count:])
    generator.shuffle(train)
    generator.shuffle(heldout)
    write_jsonl(Path(args.train_output), train)
    write_jsonl(Path(args.heldout_output), heldout)
    print(f"train={len(train)} heldout={len(heldout)} groups={len(groups)}")


if __name__ == "__main__":
    main()
