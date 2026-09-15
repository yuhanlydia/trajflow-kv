#!/usr/bin/env python3
"""Prepare critical-only and history-oracle visual counterfactual files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajflow_kv.credit_diagnostics import prepare_diagnostic_files


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--critical-output", type=Path)
    parser.add_argument("--history-oracle-output", type=Path)
    parser.add_argument("--balanced-memory-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    if not any((args.critical_output, args.history_oracle_output, args.balanced_memory_output, args.summary_output)):
        parser.error("request at least one output")
    result = prepare_diagnostic_files(
        args.input,
        critical_output=args.critical_output,
        history_oracle_output=args.history_oracle_output,
        balanced_memory_output=args.balanced_memory_output,
        summary_output=args.summary_output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
