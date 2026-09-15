#!/usr/bin/env python3
"""Stream a bounded AITW mirror and normalize available fields to JSONL.

AITW mirrors vary in feature names. This script fails loudly with the observed
schema instead of silently producing invalid training records.
"""
import argparse
import json
from pathlib import Path


def main():
    from datasets import load_dataset
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--local-parquet")
    parser.add_argument("--split", default="train")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", default="data/aitw/images")
    args = parser.parse_args()
    if not args.local_parquet and not args.dataset:
        parser.error("provide --local-parquet or --dataset")
    if args.local_parquet:
        stream = load_dataset("parquet", data_files=args.local_parquet, split="train")
    else:
        stream = load_dataset(args.dataset, split=args.split, streaming=True)
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.image_dir); image_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    with output.open("w", encoding="utf-8") as target:
        for row in stream:
            instruction = row.get("goal") or row.get("instruction") or row.get("task")
            action = row.get("action") or row.get("action_type")
            task_id = row.get("episode_id") or row.get("task_id")
            image = row.get("image_path") or row.get("screenshot_path")
            embedded = row.get("image")
            if embedded is not None:
                image_path = image_dir / f"{task_id}_{row.get('step_id', written)}.png"
                embedded.save(image_path)
                image = str(image_path.resolve())
            if instruction is None or action is None or task_id is None:
                if written == 0:
                    raise RuntimeError(f"Unsupported AITW mirror schema: {sorted(row.keys())}")
                continue
            record = {"task_id": str(task_id), "instruction": str(instruction), "return": 1.0,
                      "steps": [{"image": image, "history": [], "action": str(action)}]}
            target.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1
            if written >= args.limit:
                break
    print(f"wrote {written} records to {output}")


if __name__ == "__main__":
    main()
