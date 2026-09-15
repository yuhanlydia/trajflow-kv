#!/usr/bin/env python3
"""Measure successful-vs-failed trajectory log-prob margin for a KV checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from trajflow_kv.data import load_jsonl
from trajflow_kv.projector import attach_kv_projectors, load_merged_weights
from trajflow_kv.qwen_policy import build_action_prompt


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
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--allow-single-return", action="store_true")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    ).eval()
    bundle = None
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
            logprob_sum, token_count = 0.0, 0
            transport_energies = []
            for step in trajectory["steps"]:
                content = []
                loaded_image = None
                if step.get("image"):
                    loaded_image = Image.open(step["image"]).convert("RGB")
                    content.append({"type": "image", "image": loaded_image})
                inferred_size = (
                    loaded_image.size
                    if loaded_image else tuple(step.get("screen_size", (1000, 1000)))
                )
                prompt = step.get("prompt") or build_action_prompt(
                    trajectory["instruction"], step.get("history", []), inferred_size
                )
                content.append({"type": "text", "text": prompt})
                prompt_messages = [{"role": "user", "content": content}]
                messages = prompt_messages + [{"role": "assistant", "content": [
                    {"type": "text", "text": step["action"]}
                ]}]
                prompt_batch = processor.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                batch = processor.apply_chat_template(
                    messages, tokenize=True, return_dict=True, return_tensors="pt"
                ).to("cuda")
                labels = batch["input_ids"].clone()
                labels[:, :prompt_batch["input_ids"].shape[1]] = -100
                output = model(**batch, labels=labels)
                if bundle is not None:
                    transport_energies.append(float(bundle.energy()))
                valid = int((labels != -100).sum())
                logprob_sum += -float(output.loss) * valid
                token_count += valid
            rows.append({
                "trajectory": trajectory_index,
                "task_id": trajectory["task_id"],
                "return": float(trajectory["return"]),
                "tokens": token_count,
                "mean_token_logprob": logprob_sum / max(token_count, 1),
                "mean_transport_energy": (
                    sum(transport_energies) / len(transport_energies)
                    if transport_energies else None
                ),
            })

    positive = [row["mean_token_logprob"] for row in rows if row["return"] > 0]
    negative = [row["mean_token_logprob"] for row in rows if row["return"] <= 0]
    if (not positive or not negative) and not args.allow_single_return:
        raise ValueError("return-margin evaluation requires successes and failures")
    summary = {
        "checkpoint": args.checkpoint or args.merged_checkpoint,
        "data": args.data,
        "trajectories": len(rows),
        "positive_count": len(positive),
        "negative_count": len(negative),
        "overall_mean_token_logprob": sum(
            row["mean_token_logprob"] for row in rows
        ) / len(rows),
        "positive_mean_token_logprob": sum(positive) / len(positive) if positive else None,
        "negative_mean_token_logprob": sum(negative) / len(negative) if negative else None,
        "return_margin": (
            sum(positive) / len(positive) - sum(negative) / len(negative)
            if positive and negative else None
        ),
        "mean_transport_energy": (
            sum(row["mean_transport_energy"] for row in rows) / len(rows)
            if bundle is not None else None
        ),
        "rows": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
