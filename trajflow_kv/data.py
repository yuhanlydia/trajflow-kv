from __future__ import annotations

import json
from pathlib import Path


def load_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            missing = {"task_id", "instruction", "return", "steps"} - item.keys()
            if missing:
                raise ValueError(f"line {line_no}: missing {sorted(missing)}")
            records.append(item)
    return records

