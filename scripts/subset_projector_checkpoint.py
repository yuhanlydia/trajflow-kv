#!/usr/bin/env python3
"""Extract a layer/target subset from an ordered KV projector checkpoint."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-layers", type=int, required=True)
    parser.add_argument("--source-target", choices=("k", "v", "both"), default="both")
    parser.add_argument("--target", choices=("k", "v"), required=True)
    parser.add_argument("--last-n-layers", type=int, required=True)
    args = parser.parse_args()
    if args.last_n_layers > args.source_layers:
        raise ValueError("last-n-layers exceeds source-layers")
    source_targets = ("k", "v") if args.source_target == "both" else (args.source_target,)
    if args.target not in source_targets:
        raise ValueError("requested target is absent from source checkpoint")
    source = torch.load(args.input, map_location="cpu", weights_only=True)
    output_state = {}
    target_slot = source_targets.index(args.target)
    width = len(source_targets)
    first_layer = args.source_layers - args.last_n_layers
    for output_index, layer in enumerate(range(first_layer, args.source_layers)):
        source_index = layer * width + target_slot
        for parameter in ("down.weight", "up.weight"):
            output_state[f"{output_index}.{parameter}"] = source[
                f"{source_index}.{parameter}"
            ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_state, output)
    print(f"modules={args.last_n_layers} output={output}")


if __name__ == "__main__":
    main()
