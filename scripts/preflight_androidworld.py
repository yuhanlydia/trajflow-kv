#!/usr/bin/env python3
"""Reject unhealthy/unsupported AndroidWorld hosts before costly VLM rollout."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from trajflow_kv.androidworld_preflight import inspect_androidworld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:5000")
    parser.add_argument("--adb", default="/opt/android/platform-tools/adb")
    parser.add_argument("--require-kvm", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--output")
    args = parser.parse_args()
    result = inspect_androidworld(
        args.server_url, args.adb, args.require_kvm, args.timeout
    )
    rendered = json.dumps(result, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    if not result["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
