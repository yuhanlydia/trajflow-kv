#!/usr/bin/env python3
"""Score compact Mind2Web element candidates with a base or KV-modified Qwen."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trajflow_kv.mind2web import build_prompt
from trajflow_kv.projector import attach_kv_projectors, load_merged_weights


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--checkpoint")
    group.add_argument("--merged-checkpoint")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--target", choices=("k", "v", "both"), default="v")
    parser.add_argument("--last-n-layers", type=int, default=8)
    args = parser.parse_args()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model)
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
    elif args.merged_checkpoint:
        load_merged_weights(model, args.merged_checkpoint)

    rows = []
    with torch.inference_mode(), Path(args.data).open(encoding="utf-8") as stream:
        for line in stream:
            sample = json.loads(line)
            prompt = build_prompt(sample["task"], sample["previous_actions"], sample["choices"])
            user = {"role": "user", "content": [{"type": "text", "text": prompt}]}
            prefix = processor.apply_chat_template(
                [user], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt",
            )
            scores = []
            for target in sample["targets"]:
                batch = processor.apply_chat_template(
                    [user, {"role": "assistant", "content": [{"type": "text", "text": target}]}],
                    tokenize=True, return_dict=True, return_tensors="pt",
                ).to("cuda")
                labels = batch["input_ids"].clone()
                labels[:, :prefix["input_ids"].shape[1]] = -100
                scores.append(-float(model(**batch, labels=labels, use_cache=False).loss))
            correct = int(sample["correct_index"])
            order = sorted(range(len(scores)), key=lambda index: scores[index], reverse=True)
            rank = order.index(correct) + 1
            rows.append({"sample_id": sample["sample_id"], "rank": rank,
                         "correct_score": scores[correct],
                         "best_incorrect_score": max(s for i, s in enumerate(scores) if i != correct)})
    summary = {
        "checkpoint": args.checkpoint or args.merged_checkpoint or "base",
        "samples": len(rows),
        "top1_accuracy": sum(row["rank"] == 1 for row in rows) / len(rows),
        "mrr": sum(1.0 / row["rank"] for row in rows) / len(rows),
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
