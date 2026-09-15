#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trajflow_kv.action_flow import (
    ActionRectifiedFlow, encode_action_chunk, flow_condition, project_action_chunks,
)
from trajflow_kv.data import load_jsonl


def successful(path: str, prefix: str):
    return [item for item in load_jsonl(path)
            if float(item["return"]) > 0 and item["task_id"].startswith(prefix)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-prefix", default="System")
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    train = successful(args.train, args.task_prefix)
    heldout = successful(args.heldout, args.task_prefix)
    targets = torch.stack([encode_action_chunk(item, args.horizon) for item in train])
    conditions = torch.stack([flow_condition(item) for item in train])
    model = ActionRectifiedFlow(targets.shape[1], conditions.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(args.seed + 1)
    losses = []
    for epoch in range(args.epochs):
        noise = torch.randn(targets.shape, generator=generator)
        time = torch.rand(len(targets), generator=generator)
        state = (1 - time[:, None]) * noise + time[:, None] * targets
        velocity = targets - noise
        loss = torch.nn.functional.mse_loss(model(state, time, conditions), velocity)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0 or epoch + 1 == args.epochs:
            losses.append({"epoch": epoch, "loss": float(loss)})

    per_task = []
    with torch.inference_mode():
        for item in heldout:
            target = encode_action_chunk(item, args.horizon)
            single_condition = flow_condition(item)
            condition = single_condition.repeat(args.samples, 1)
            noise = torch.randn(args.samples, len(target), generator=generator)
            generated = model.sample(noise, condition)
            generated_mse = (generated - target).square().mean(-1)
            projected = project_action_chunks(generated, args.horizon)
            projected_mse = (projected - target).square().mean(-1)
            noise_mse = (noise - target).square().mean(-1)
            type_width = 7
            reshaped = generated.reshape(args.samples, args.horizon, -1)
            type_confident = (reshaped[:, :, :type_width].max(-1).values >= 0.5)
            coordinates = reshaped[:, :, type_width:type_width + 2]
            coordinate_valid = ((coordinates >= -0.1) & (coordinates <= 1.1)).all(-1)
            structured_valid = (type_confident & coordinate_valid).float().mean()
            mean_bc_mse = (targets.mean(0) - target).square().mean()
            nearest_index = (conditions - single_condition).square().mean(-1).argmin()
            nearest_mse = (targets[nearest_index] - target).square().mean()
            per_task.append({
                "task_id": item["task_id"],
                "generated_mean_mse": float(generated_mse.mean()),
                "generated_best_of_n_mse": float(generated_mse.min()),
                "projected_mean_mse": float(projected_mse.mean()),
                "projected_best_of_n_mse": float(projected_mse.min()),
                "noise_mean_mse": float(noise_mse.mean()),
                "mean_bc_mse": float(mean_bc_mse),
                "condition_nearest_neighbor_mse": float(nearest_mse),
                "endpoint_diversity": float(generated.var(0).mean()),
                "structured_valid_rate": float(structured_valid),
            })
    summary = {
        "train_trajectories": len(train), "heldout_trajectories": len(heldout),
        "horizon": args.horizon, "samples_per_task": args.samples,
        "final_loss": losses[-1]["loss"], "tasks": per_task,
        "mean_generated_mse": sum(row["generated_mean_mse"] for row in per_task) / len(per_task),
        "mean_noise_mse": sum(row["noise_mean_mse"] for row in per_task) / len(per_task),
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "action_width": targets.shape[1],
                "condition_width": conditions.shape[1]}, output / "flow.pt")
    (output / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "loss.json").write_text(json.dumps(losses, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
