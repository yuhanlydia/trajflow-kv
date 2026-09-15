#!/usr/bin/env python3
"""Merge rollout JSONL files while preserving complete trajectory records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    records = []
    for source_name in args.inputs:
        source = Path(source_name)
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                if line.strip():
                    record = json.loads(line)
                    if not {"task_id", "return", "steps"} <= set(record):
                        raise ValueError(f"invalid rollout at {source}:{line_number}")
                    records.append(record)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"merged={len(records)} output={output}")


if __name__ == "__main__":
    main()
