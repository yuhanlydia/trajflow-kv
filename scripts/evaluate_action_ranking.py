#!/usr/bin/env python3
"""Evaluate held-out correct-action rank under a fixed legal proposal set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from trajflow_kv.actions import canonical_action
from trajflow_kv.data import load_jsonl
from trajflow_kv.projector import attach_kv_projectors, load_merged_weights
from trajflow_kv.qwen_policy import build_action_prompt


def candidates(instruction: str, correct: str) -> list[str]:
    values = [
        canonical_action({"action_type": "swipe", "direction": "down"}),
        canonical_action({"action_type": "navigate_back"}),
        canonical_action({"action_type": "wait"}),
        canonical_action({"action_type": "status", "goal_status": "complete"}),
    ]
    if "wifi" in instruction.lower():
        values += [
            canonical_action({"action_type": "click", "x": 260, "y": 200}),
            canonical_action({"action_type": "click", "x": 260, "y": 340}),
            canonical_action({"action_type": "click", "x": 870, "y": 920}),
        ]
    else:
        values += [
            canonical_action({"action_type": "click", "x": 780, "y": 200}),
            canonical_action({"action_type": "click", "x": 780, "y": 340}),
        ]
    correct = canonical_action(json.loads(correct))
    if correct not in values:
        values.append(correct)
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--checkpoint")
    checkpoint_group.add_argument("--merged-checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--target", choices=("k", "v", "both"), default="both")
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument("--max-pixels", type=int, default=100352)
    args = parser.parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    if args.checkpoint:
        bundle = attach_kv_projectors(
            model, args.rank, args.alpha, args.target, args.last_n_layers
        )
        bundle.modules.load_state_dict(
            torch.load(args.checkpoint, map_location="cuda", weights_only=True)
        )
    else:
        load_merged_weights(model, args.merged_checkpoint)
    rows = []
    with torch.inference_mode():
        for trajectory_index, trajectory in enumerate(load_jsonl(args.data)):
            if float(trajectory["return"]) <= 0:
                continue
            for step_index, step in enumerate(trajectory["steps"]):
                image = Image.open(step["image"]).convert("RGB")
                prompt = build_action_prompt(
                    trajectory["instruction"], step.get("history", []), image.size
                )
                correct = canonical_action(json.loads(step["action"]))
                proposals = candidates(trajectory["instruction"], correct)
                scores = []
                for proposal in proposals:
                    user = {"role": "user", "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ]}
                    prompt_batch = processor.apply_chat_template(
                        [user], tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                    )
                    batch = processor.apply_chat_template(
                        [user, {"role": "assistant", "content": [
                            {"type": "text", "text": proposal}
                        ]}], tokenize=True, return_dict=True, return_tensors="pt",
                    ).to("cuda")
                    labels = batch["input_ids"].clone()
                    labels[:, :prompt_batch["input_ids"].shape[1]] = -100
                    scores.append(-float(model(**batch, labels=labels).loss))
                correct_index = proposals.index(correct)
                order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
                action_rank = order.index(correct_index) + 1
                correct_type = json.loads(correct)["action_type"]
                same_type = [
                    index for index, proposal in enumerate(proposals)
                    if json.loads(proposal)["action_type"] == correct_type
                ]
                type_order = sorted(same_type, key=lambda i: scores[i], reverse=True)
                type_rank = type_order.index(correct_index) + 1
                rows.append({
                    "trajectory": trajectory_index,
                    "step": step_index,
                    "correct": correct,
                    "rank": action_rank,
                    "type_rank": type_rank,
                    "same_type_candidates": len(same_type),
                    "candidates": len(proposals),
                    "correct_score": scores[correct_index],
                    "best_incorrect_score": max(
                        score for index, score in enumerate(scores) if index != correct_index
                    ),
                })
    summary = {
        "checkpoint": args.checkpoint or args.merged_checkpoint,
        "steps": len(rows),
        "top1_accuracy": sum(row["rank"] == 1 for row in rows) / len(rows),
        "mrr": sum(1.0 / row["rank"] for row in rows) / len(rows),
        "type_conditioned_top1": sum(row["type_rank"] == 1 for row in rows) / len(rows),
        "type_conditioned_mrr": sum(1.0 / row["type_rank"] for row in rows) / len(rows),
        "mean_correct_margin": sum(
            row["correct_score"] - row["best_incorrect_score"] for row in rows
        ) / len(rows),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
