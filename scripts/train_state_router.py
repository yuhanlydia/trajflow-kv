#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from trajflow_kv.data import load_jsonl
from trajflow_kv.state_router import StateRouter, routed_action_type, router_features


def examples(path: str, task_prefix: str | None) -> list[tuple[torch.Tensor, str]]:
    rows = []
    for trajectory in load_jsonl(path):
        if float(trajectory["return"]) <= 0:
            continue
        if task_prefix and not trajectory["task_id"].startswith(task_prefix):
            continue
        for step in trajectory["steps"]:
            label = routed_action_type(step["action"])
            if label and step.get("image"):
                image = Image.open(step["image"]).convert("RGB")
                rows.append((router_features(
                    image, len(step.get("history", [])), trajectory["instruction"]
                ), label))
    return rows


def evaluate(model: StateRouter, rows: list[tuple[torch.Tensor, str]]) -> dict:
    correct = 0
    confusion = {}
    with torch.inference_mode():
        for features, label in rows:
            predicted = model.classes[int(model(features.unsqueeze(0)).argmax(-1))]
            correct += predicted == label
            key = f"{label}->{predicted}"
            confusion[key] = confusion.get(key, 0) + 1
    return {"examples": len(rows), "accuracy": correct / len(rows), "confusion": confusion}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-prefix", default="System")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train_rows = examples(args.train, args.task_prefix)
    heldout_rows = examples(args.heldout, args.task_prefix)
    classes = sorted({label for _, label in train_rows})
    model = StateRouter(len(train_rows[0][0]), classes, args.hidden)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    x = torch.stack([features for features, _ in train_rows])
    y = torch.tensor([classes.index(label) for _, label in train_rows])
    for _ in range(args.epochs):
        loss = torch.nn.functional.cross_entropy(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model.save(output / "state_router.pt", args.hidden)
    summary = {
        "classes": classes,
        "train": evaluate(model, train_rows),
        "heldout": evaluate(model, heldout_rows),
    }
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
