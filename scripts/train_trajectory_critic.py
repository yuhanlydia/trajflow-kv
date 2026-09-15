#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trajflow_kv.critic import TrajectoryCritic, critic_features
from trajflow_kv.data import load_jsonl


def make_rows(data_path: str, scores_path: str, task_prefix: str | None):
    trajectories = load_jsonl(data_path)
    scores = json.loads(Path(scores_path).read_text())["rows"]
    by_index = {row["trajectory"]: row for row in scores}
    rows = []
    for index, trajectory in enumerate(trajectories):
        if task_prefix and not trajectory["task_id"].startswith(task_prefix):
            continue
        score = by_index[index]["mean_token_logprob"]
        rows.append((critic_features(trajectory, score), float(trajectory["return"])))
    return rows


def metrics(model: TrajectoryCritic, rows) -> dict:
    x = torch.stack([row[0] for row in rows])
    y = torch.tensor([row[1] for row in rows])
    with torch.inference_mode():
        probabilities = model(x).sigmoid()
    positive = probabilities[y > 0.5]
    negative = probabilities[y <= 0.5]
    auc = torch.stack([(p > n).float() + 0.5 * (p == n).float()
                       for p in positive for n in negative]).mean()
    return {
        "examples": len(rows),
        "accuracy": float(((probabilities >= 0.5) == (y > 0.5)).float().mean()),
        "auc": float(auc),
        "positive_mean": float(positive.mean()),
        "negative_mean": float(negative.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--train-scores", required=True)
    parser.add_argument("--heldout-data", required=True)
    parser.add_argument("--heldout-scores", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-prefix", default="System")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--hidden", type=int, default=0)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train = make_rows(args.train_data, args.train_scores, args.task_prefix)
    heldout = make_rows(args.heldout_data, args.heldout_scores, args.task_prefix)
    x = torch.stack([row[0] for row in train])
    y = torch.tensor([row[1] for row in train])
    model = TrajectoryCritic(x.shape[1], args.hidden)
    model.set_normalization(x)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    positives = y.sum().clamp_min(1)
    pos_weight = (len(y) - positives) / positives
    for _ in range(args.epochs):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            model(x), y, pos_weight=pos_weight
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "width": x.shape[1], "hidden": args.hidden
    }, output / "critic.pt")
    summary = {"train": metrics(model, train), "heldout": metrics(model, heldout)}
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
