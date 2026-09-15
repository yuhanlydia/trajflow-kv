#!/usr/bin/env python3
"""Fine-tune KV projectors with state-conditioned action preferences."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from trajflow_kv.projector import attach_kv_projectors
from trajflow_kv.qwen_policy import build_action_prompt


def load_pairs(path: str) -> list[dict]:
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--target", choices=("k", "v", "both"), default="v")
    parser.add_argument("--last-n-layers", type=int, default=8)
    parser.add_argument("--max-pixels", type=int, default=200704)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=2.0)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--lambda-energy", type=float, default=0.001)
    parser.add_argument("--lambda-orth", type=float, default=0.01)
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    bundle = attach_kv_projectors(
        model, args.rank, args.alpha, args.target, args.last_n_layers
    )
    bundle.modules.load_state_dict(
        torch.load(args.checkpoint, map_location="cuda", weights_only=True)
    )
    optimizer = torch.optim.AdamW(bundle.modules.parameters(), lr=args.lr)
    pairs = load_pairs(args.data)

    def action_score(pair: dict, action: str) -> torch.Tensor:
        image = Image.open(pair["image"]).convert("RGB")
        prompt = build_action_prompt(
            pair["instruction"], pair.get("history", []), image.size
        )
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
                {"type": "text", "text": action}
            ]}], tokenize=True, return_dict=True, return_tensors="pt",
        ).to("cuda")
        labels = batch["input_ids"].clone()
        labels[:, :prompt_batch["input_ids"].shape[1]] = -100
        return -model(**batch, labels=labels, use_cache=False).loss

    history = []
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        for index, pair in enumerate(pairs):
            # First obtain a detached logistic coefficient. Recomputing the two
            # scores avoids retaining both VLM activation graphs simultaneously.
            with torch.no_grad():
                margin = action_score(pair, pair["chosen"]) - action_score(
                    pair, pair["rejected"]
                )
                coefficient = args.beta * torch.sigmoid(-args.beta * margin)
            chosen_score = action_score(pair, pair["chosen"])
            chosen_loss = -coefficient * chosen_score
            (chosen_loss / args.gradient_accumulation_steps).backward()
            rejected_score = action_score(pair, pair["rejected"])
            rejected_loss = coefficient * rejected_score
            regularizer = (
                args.lambda_energy * bundle.energy()
                + args.lambda_orth * bundle.orthogonality_loss()
            )
            ((rejected_loss + regularizer) / args.gradient_accumulation_steps).backward()
            if (index + 1) % args.gradient_accumulation_steps == 0 or index + 1 == len(pairs):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            history.append({
                "epoch": epoch,
                "pair": index,
                "margin_before_update": float(margin),
                "coefficient": float(coefficient),
            })

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.modules.state_dict(), output / "kv_projectors.pt")
    (output / "metrics.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "pairs": len(pairs),
        "epochs": args.epochs,
        "last_mean_margin": sum(row["margin_before_update"] for row in history[-len(pairs):]) / len(pairs),
    }, indent=2))


if __name__ == "__main__":
    main()
