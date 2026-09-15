#!/usr/bin/env python3
"""Filter trajectory JSONL by return interval without changing records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-return", type=float, default=float("-inf"))
    parser.add_argument("--max-return", type=float, default=float("inf"))
    args = parser.parse_args()
    records = []
    with Path(args.input).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = json.loads(line)
            value = float(record["return"])
            if args.min_return <= value <= args.max_return:
                records.append(record)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"selected={len(records)} output={output}")


if __name__ == "__main__":
    main()
