#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajflow_kv.projector import attach_kv_projectors, merge_projectors_into_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--target", choices=("k", "v", "both"), default="v")
    parser.add_argument("--last-n-layers", type=int, default=8)
    args = parser.parse_args()
    from transformers import Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cpu"
    ).eval()
    bundle = attach_kv_projectors(
        model, args.rank, args.alpha, args.target, args.last_n_layers
    )
    bundle.modules.load_state_dict(torch.load(args.checkpoint, weights_only=True))
    weights = merge_projectors_into_model(model, bundle)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "weights": weights,
        "source_projector": args.checkpoint,
        "rank": args.rank,
        "alpha": args.alpha,
        "target": args.target,
        "last_n_layers": args.last_n_layers,
    }, output)
    print({"merged_weights": len(weights), "output": str(output)})


if __name__ == "__main__":
    main()
