#!/usr/bin/env python3
"""Convert an official Mind2Web training shard to a bounded ranking fixture."""
from __future__ import annotations

import argparse
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path

from trajflow_kv.mind2web import describe_candidate, format_target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("output")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--negatives", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    tasks = json.loads(Path(args.input).read_text(encoding="utf-8"))
    rng = random.Random(args.seed)
    rows = []
    for task in tasks:
        previous: list[str] = []
        for action_index, action in enumerate(task["actions"]):
            if not action.get("pos_candidates") or not action.get("neg_candidates"):
                previous.append(task["action_reprs"][action_index])
                continue
            root = ET.fromstring(action["cleaned_html"])
            positive = action["pos_candidates"][0]
            negatives = rng.sample(
                action["neg_candidates"], min(args.negatives, len(action["neg_candidates"]))
            )
            candidates = [positive, *negatives]
            descriptions = [describe_candidate(root, item) for item in candidates]
            rows.append({
                "sample_id": f"{task['annotation_id']}_{action['action_uid']}",
                "website": task["website"],
                "task": task["confirmed_task"],
                "previous_actions": previous[-5:],
                "operation": action["operation"],
                "choices": descriptions,
                "targets": [format_target(item, action["operation"]) for item in descriptions],
                "correct_index": 0,
            })
            previous.append(task["action_reprs"][action_index])
            if len(rows) >= args.limit:
                break
        if len(rows) >= args.limit:
            break
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"rows": len(rows), "websites": sorted({r['website'] for r in rows})}))


if __name__ == "__main__":
    main()
