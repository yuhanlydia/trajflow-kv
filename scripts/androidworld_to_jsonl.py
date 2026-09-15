#!/usr/bin/env python3
"""Normalize an AndroidWorld rollout log into TrajFlow-KV JSONL."""
import argparse, json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("input"); p.add_argument("output")
    args = p.parse_args()
    source = json.loads(Path(args.input).read_text())
    episodes = source if isinstance(source, list) else source.get("episodes", [])
    with Path(args.output).open("w") as out:
        for episode in episodes:
            record = {"task_id": str(episode["task_id"]), "instruction": episode["instruction"],
                      "return": float(episode.get("return", episode.get("success", 0))),
                      "steps": episode["steps"]}
            out.write(json.dumps(record) + "\n")


if __name__ == "__main__": main()

