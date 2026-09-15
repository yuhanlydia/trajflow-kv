#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from trajflow_kv.projector import resize_low_rank_factors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-alpha", type=float, required=True)
    parser.add_argument("--target-rank", type=int, required=True)
    parser.add_argument("--target-alpha", type=float, required=True)
    args = parser.parse_args()
    source = torch.load(args.input, map_location="cpu", weights_only=True)
    module_indices = sorted({int(key.split(".", 1)[0]) for key in source})
    resized = {}
    for index in module_indices:
        down, up = resize_low_rank_factors(
            source[f"{index}.down.weight"], source[f"{index}.up.weight"],
            args.source_alpha, args.target_rank, args.target_alpha,
        )
        resized[f"{index}.down.weight"] = down
        resized[f"{index}.up.weight"] = up
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(resized, output)
    print(f"modules={len(module_indices)} rank={args.target_rank} output={output}")


if __name__ == "__main__":
    main()
