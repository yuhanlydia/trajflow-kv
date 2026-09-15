#!/usr/bin/env python3
"""Collect mixed-return MiniWoB trajectories from seeded random legal clicks."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

from PIL import Image

from trajflow_kv.miniwob import (
    build_miniwob_prompt, candidate_target, clickable_candidates, exclude_acted_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--miniwob-url", default="file:///root/miniwob-plusplus/miniwob/html/miniwob/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--image-dir", default="data/miniwob/images")
    args = parser.parse_args()
    os.environ["MINIWOB_URL"] = args.miniwob_url
    import browsergym.miniwob  # noqa: F401
    import gymnasium as gym

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image_root = Path(args.image_dir)
    records = []
    for task in args.tasks:
        for seed in args.seeds:
            rng = random.Random(f"{task}:{seed}")
            env = gym.make(f"browsergym/miniwob.{task}", headless=True)
            steps, history, acted_bids = [], [], set()
            total_reward = 0.0
            try:
                observation, _ = env.reset(seed=seed)
                instruction = str(observation["goal"])
                for step_index in range(args.max_steps):
                    candidates = exclude_acted_candidates(
                        clickable_candidates(observation), acted_bids
                    )
                    if not candidates:
                        break
                    prompt = build_miniwob_prompt(instruction, history, candidates)
                    selected = rng.choice(candidates)
                    target = candidate_target(selected)
                    episode_dir = image_root / task / str(seed)
                    episode_dir.mkdir(parents=True, exist_ok=True)
                    image_path = (episode_dir / f"{step_index:03d}.png").resolve()
                    Image.fromarray(observation["screenshot"][:, :, :3]).save(image_path)
                    observation, reward, terminated, truncated, _ = env.step(
                        f"click('{selected['bid']}')"
                    )
                    total_reward += float(reward)
                    steps.append({
                        "image": str(image_path), "screen_size": [498, 321],
                        "history": history.copy(), "prompt": prompt, "action": target,
                        "bid": selected["bid"],
                        "action_error": str(observation.get("last_action_error", "")),
                    })
                    history.append(target)
                    if not observation.get("last_action_error"):
                        acted_bids.add(selected["bid"])
                    if terminated or truncated:
                        break
            finally:
                env.close()
            records.append({
                "task_id": task, "instruction": instruction, "return": total_reward,
                "steps": steps, "metadata": {"benchmark": "miniwob", "seed": seed,
                                                "policy": "seeded_random_legal_click"},
            })
            print(f"task={task} seed={seed} return={total_reward}")
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({
        "trajectories": len(records),
        "successes": sum(record["return"] > 0 for record in records),
        "by_task": {task: {
            "count": sum(record["task_id"] == task for record in records),
            "successes": sum(record["task_id"] == task and record["return"] > 0 for record in records),
        } for task in args.tasks},
    }, indent=2))


if __name__ == "__main__":
    main()
