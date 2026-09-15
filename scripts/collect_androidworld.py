#!/usr/bin/env python3
"""Collect mixed-return rollouts through AndroidWorld's official HTTP server."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch

from trajflow_kv.androidworld_http import AndroidWorldHTTPClient
from trajflow_kv.qwen_policy import QwenKVPolicy
from trajflow_kv.rollout import append_jsonl, collect_rollout, stable_task_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:5000")
    parser.add_argument("--http-timeout", type=float, default=300.0)
    parser.add_argument("--task", default="ContactsAddContact")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", default="outputs/qwen-aitw/kv_projectors.pt")
    parser.add_argument("--no-checkpoint", action="store_true")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--target", choices=("k", "v", "both"), default="both")
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--max-pixels", type=int, default=401408)
    parser.add_argument("--candidate-mode", choices=("system", "system_hierarchical"))
    parser.add_argument("--max-identical-actions", type=int)
    parser.add_argument("--max-identical-candidates", type=int)
    parser.add_argument("--state-router-checkpoint")
    parser.add_argument("--output", default="data/androidworld/rollouts.jsonl")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    client = AndroidWorldHTTPClient(args.server_url, timeout=args.http_timeout)
    if not client.health():
        raise RuntimeError(
            f"AndroidWorld server is not healthy at {args.server_url}. "
            "Start its official Docker server before collecting rollouts."
        )
    instruction = client.goal(args.task, args.task_index)
    task_id = stable_task_id(args.task, instruction, {"task_idx": args.task_index})
    checkpoint = None if args.no_checkpoint else args.checkpoint
    policy = QwenKVPolicy(
        args.model,
        checkpoint,
        rank=args.rank,
        alpha=args.alpha,
        target=args.target,
        last_n_layers=args.last_n_layers,
        max_pixels=args.max_pixels,
        temperature=args.temperature,
        candidate_mode=args.candidate_mode,
        max_identical_actions=args.max_identical_actions,
        max_identical_candidates=args.max_identical_candidates,
        state_router_checkpoint=args.state_router_checkpoint,
    )
    results = []
    for rollout_index in range(args.rollouts):
        initialized = False
        try:
            client.reset(go_home=True)
            client.initialize_task(args.task, args.task_index)
            initialized = True
            run_id = f"{task_id}-r{rollout_index:03d}"
            result = collect_rollout(
                policy=policy, instruction=instruction, task_id=task_id,
                max_steps=args.max_steps,
                image_dir=Path(args.output).parent / "images" / run_id,
                get_pixels=client.screenshot, screen_size=lambda: client.screen_size,
                execute=client.execute,
                evaluate=lambda: client.score(args.task, args.task_index),
                rollout_metadata={
                    "rollout_index": rollout_index,
                    "seed": args.seed,
                    "temperature": args.temperature,
                    "checkpoint": checkpoint,
                },
            )
            append_jsonl(args.output, [result])
            results.append(result)
            print(f"rollout={rollout_index} return={result.return_} invalid={result.invalid_actions}")
        except Exception as error:  # Keep a long stochastic collection batch alive.
            print(f"rollout={rollout_index} error={type(error).__name__}: {error}", flush=True)
        finally:
            if initialized:
                try:
                    client.tear_down_task(args.task, args.task_index)
                except Exception as error:
                    print(
                        f"rollout={rollout_index} teardown_error="
                        f"{type(error).__name__}: {error}",
                        flush=True,
                    )
    values = [result.return_ for result in results]
    print(f"collected={len(values)} returns={values} output={args.output}")


if __name__ == "__main__":
    main()
