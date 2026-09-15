#!/usr/bin/env python3
"""Run a preflighted, same-seed Base/KV AndroidWorld online gate."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from trajflow_kv.androidworld_preflight import inspect_androidworld


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://127.0.0.1:5000")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--state-router-checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--allow-software-emulator", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    health = inspect_androidworld(
        args.server_url, require_kvm=not args.allow_software_emulator
    )
    if not health["ok"]:
        raise SystemExit("AndroidWorld preflight failed: " + "; ".join(health["failures"]))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base = output_dir / "base.jsonl"
    candidate = output_dir / "kv.jsonl"
    summary = output_dir / "paired_summary.json"
    common = [
        sys.executable, "scripts/collect_androidworld.py",
        "--server-url", args.server_url, "--task", args.task,
        "--task-index", str(args.task_index), "--rollouts", str(args.rollouts),
        "--max-steps", str(args.max_steps), "--seed", str(args.seed),
        "--temperature", str(args.temperature), "--target", "v",
        "--last-n-layers", "8", "--rank", "8", "--alpha", "8",
        "--candidate-mode", "system_hierarchical",
        "--max-identical-actions", "2", "--max-identical-candidates", "1",
    ]
    if args.state_router_checkpoint:
        common += ["--state-router-checkpoint", args.state_router_checkpoint]
    subprocess.run(common + ["--no-checkpoint", "--output", str(base)], check=True)
    # Recheck between arms so a Base run that destabilizes the guest cannot
    # silently contaminate only the KV arm.
    between = inspect_androidworld(
        args.server_url, require_kvm=not args.allow_software_emulator
    )
    if not between["ok"]:
        raise SystemExit("AndroidWorld failed between-arm preflight: " + "; ".join(between["failures"]))
    subprocess.run(
        common + ["--checkpoint", args.checkpoint, "--output", str(candidate)], check=True
    )
    subprocess.run([
        sys.executable, "scripts/compare_rollouts.py", "--baseline", str(base),
        "--candidate", str(candidate), "--output", str(summary),
    ], check=True)


if __name__ == "__main__":
    main()
