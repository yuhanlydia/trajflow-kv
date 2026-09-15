#!/usr/bin/env python3
"""Summarize critical/non-critical ground truth in visual counterfactual JSONL."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajflow_kv.counterfactual import load_counterfactual_jsonl
from trajflow_kv.credit_diagnostics import summarize_credit_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = summarize_credit_rows(load_counterfactual_jsonl(args.input))
    result["input"] = str(args.input)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

