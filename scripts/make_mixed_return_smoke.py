#!/usr/bin/env python3
"""Build a two-trajectory mixed-return smoke set from an existing JSONL."""
import argparse
import json
from pathlib import Path

from trajflow_kv.data import load_jsonl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("output")
    args = parser.parse_args()
    records = load_jsonl(args.source)
    if len(records) < 2:
        raise ValueError("source needs at least two trajectories")
    selected = [records[0], records[-2]]
    for index, record in enumerate(selected):
        record["task_id"] = "mixed-return-smoke"
        record["return"] = float(index)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for record in selected:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(output)


if __name__ == "__main__":
    main()
