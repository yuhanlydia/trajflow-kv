#!/usr/bin/env python3
"""Paired-friendly online MiniWoB evaluator for Base or Return-KV Qwen."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from PIL import Image

from trajflow_kv.miniwob import (
    build_miniwob_prompt, candidate_target, clickable_candidates, exclude_acted_candidates,
)
from trajflow_kv.projector import attach_kv_projectors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint")
    parser.add_argument("--tasks", nargs="+", default=["click-test", "click-button-sequence"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[401, 402, 403])
    parser.add_argument("--suite", choices=("paired24",))
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--miniwob-url", default="file:///root/miniwob-plusplus/miniwob/html/miniwob/")
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--target", choices=("k", "v", "both"), default="v")
    parser.add_argument("--last-n-layers", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=100352)
    args = parser.parse_args()
    os.environ["MINIWOB_URL"] = args.miniwob_url
    import browsergym.miniwob  # noqa: F401
    import gymnasium as gym
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    bundle = attach_kv_projectors(
        model, args.rank, args.alpha, args.target, args.last_n_layers
    )
    if args.checkpoint:
        bundle.modules.load_state_dict(
            torch.load(args.checkpoint, map_location="cuda", weights_only=True)
        )

    episodes = []
    with torch.inference_mode():
        if args.suite == "paired24":
            cases = (
                [(task, seed, 3) for task in ("click-button", "click-button-sequence", "click-tab")
                 for seed in (401, 402, 403)]
                + [("click-checkboxes", seed, 8) for seed in range(404, 409)]
                + [("click-option", seed, 8) for seed in range(409, 414)]
                + [("click-color", seed, 1) for seed in range(414, 419)]
            )
        else:
            cases = [
                (task, seed, args.max_steps) for task in args.tasks for seed in args.seeds
            ]
        for task, seed, episode_max_steps in cases:
                env = gym.make(f"browsergym/miniwob.{task}", headless=True)
                trajectory = []
                total_reward = 0.0
                try:
                    observation, _ = env.reset(seed=seed)
                    history = []
                    acted_bids: set[str] = set()
                    for step in range(episode_max_steps):
                        candidates = exclude_acted_candidates(
                            clickable_candidates(observation), acted_bids
                        )
                        if not candidates:
                            trajectory.append({"step": step, "error": "no_clickable_candidates"})
                            break
                        prompt = build_miniwob_prompt(
                            str(observation["goal"]), history, candidates
                        )
                        image = Image.fromarray(observation["screenshot"][:, :, :3])
                        user = {"role": "user", "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt},
                        ]}
                        prefix = processor.apply_chat_template(
                            [user], tokenize=True, add_generation_prompt=True,
                            return_dict=True, return_tensors="pt",
                        )
                        scores = []
                        for candidate in candidates:
                            target = candidate_target(candidate)
                            batch = processor.apply_chat_template(
                                [user, {"role": "assistant", "content": [
                                    {"type": "text", "text": target}
                                ]}], tokenize=True, return_dict=True, return_tensors="pt",
                            ).to("cuda")
                            labels = batch["input_ids"].clone()
                            labels[:, :prefix["input_ids"].shape[1]] = -100
                            scores.append(-float(model(**batch, labels=labels, use_cache=False).loss))
                        selected_index = max(range(len(scores)), key=scores.__getitem__)
                        selected = candidates[selected_index]
                        action = f"click('{selected['bid']}')"
                        observation, reward, terminated, truncated, info = env.step(action)
                        total_reward += float(reward)
                        rendered = candidate_target(selected)
                        history.append(rendered)
                        if not observation.get("last_action_error"):
                            acted_bids.add(selected["bid"])
                        trajectory.append({
                            "step": step, "action": action, "selected": selected,
                            "scores": scores, "reward": float(reward),
                            "terminated": bool(terminated), "truncated": bool(truncated),
                            "last_action_error": str(observation.get("last_action_error", "")),
                        })
                        if terminated or truncated:
                            break
                finally:
                    env.close()
                episodes.append({
                    "task": task, "seed": seed, "return": total_reward,
                    "success": total_reward > 0, "steps": trajectory,
                })
                print(f"task={task} seed={seed} return={total_reward}")
    summary = {
        "checkpoint": args.checkpoint or "base",
        "tasks": sorted({row["task"] for row in episodes}),
        "seeds": sorted({row["seed"] for row in episodes}),
        "suite": args.suite,
        "episodes": len(episodes),
        "successes": sum(row["success"] for row in episodes),
        "success_rate": sum(row["success"] for row in episodes) / len(episodes),
        "rows": episodes,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
