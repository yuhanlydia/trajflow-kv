#!/usr/bin/env python3
"""Scan which decoder K/V layers causally carry a historical visual block."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

try:  # Script execution adds scripts/ to sys.path; tests import it as a package.
    from evaluate_counterfactual_qwen import _groups, _optimal_actions, _score_groups, summarize_policy_scores
except ModuleNotFoundError:
    from scripts.evaluate_counterfactual_qwen import _groups, _optimal_actions, _score_groups, summarize_policy_scores
from trajflow_kv.causal_ablation import attach_kv_block_ablator, decoder_layer_index
from trajflow_kv.counterfactual import load_counterfactual_jsonl


def _layer_groups(layer_count: int) -> dict[str, list[int]]:
    if layer_count < 3:
        return {"all": list(range(layer_count))}
    first = layer_count // 3
    second = 2 * layer_count // 3
    return {
        "early": list(range(0, first)),
        "middle": list(range(first, second)),
        "late": list(range(second, layer_count)),
        "all": list(range(layer_count)),
    }


def _compact(summary: dict) -> dict:
    return {
        "prefixes": summary["prefixes"],
        "candidate_top1_accuracy": summary["candidate_top1_accuracy"],
        "critical_prefixes": summary["critical_prefixes"],
        "critical_fork_accuracy": summary["critical_fork_accuracy"],
        "families": summary["families"],
    }


def signed_score_effects(baseline_groups: list[list[dict]], ablated_groups: list[list[dict]], image_index: int) -> dict:
    """Measure signed causal credit from correct-action log-probability changes.

    A positive `correct_score_effect` means removing the KV block lowers the
    correct action score (useful memory). A negative value means removal helps
    (harmful/stale memory). This is a policy-score diagnostic, not environment Q.
    """
    ablated_by_prefix = {str(group[0]["prefix_id"]): group for group in ablated_groups}
    rows = []
    for full in baseline_groups:
        prefix_id = str(full[0]["prefix_id"])
        dropped = ablated_by_prefix[prefix_id]
        optimal = _optimal_actions(full)
        full_scores = {str(row["action"]): float(row["score"]) for row in full}
        drop_scores = {str(row["action"]): float(row["score"]) for row in dropped}
        correct_full = max(full_scores[action] for action in optimal)
        correct_drop = max(drop_scores[action] for action in optimal)
        wrong_full = max((score for action, score in full_scores.items() if action not in optimal), default=correct_full)
        wrong_drop = max((score for action, score in drop_scores.items() if action not in optimal), default=correct_drop)
        labels = full[0].get("memory_advantages", [])
        ground_truth = float(labels[image_index]) if image_index < len(labels) else None
        effect = correct_full - correct_drop
        rows.append({
            "prefix_id": prefix_id,
            "seed": full[0].get("seed"),
            "ground_truth_sign": None if ground_truth is None else (1 if ground_truth > 0 else -1 if ground_truth < 0 else 0),
            "correct_score_effect": effect,
            "margin_effect": (correct_full - wrong_full) - (correct_drop - wrong_drop),
        })
    effects = [row["correct_score_effect"] for row in rows]
    labeled = [row for row in rows if row["ground_truth_sign"] is not None]
    tolerance = 1e-4
    for row in labeled:
        value = row["correct_score_effect"]
        row["predicted_sign"] = 1 if value > tolerance else -1 if value < -tolerance else 0
    return {
        "prefixes": len(rows),
        "mean_correct_score_effect": sum(effects) / len(effects) if effects else None,
        "mean_absolute_correct_score_effect": sum(abs(value) for value in effects) / len(effects) if effects else None,
        "memory_sign_accuracy": (
            sum(row["predicted_sign"] == row["ground_truth_sign"] for row in labeled) / len(labeled)
            if labeled else None
        ),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--family", default="hidden_memory")
    parser.add_argument("--targets", nargs="+", choices=("k", "v", "both"), default=("k", "v", "both"))
    parser.add_argument("--image-indices", nargs="+", type=int, default=(0, 1))
    parser.add_argument("--layer-groups", nargs="+", choices=("early", "middle", "late", "all"), default=("early", "middle", "late", "all"))
    parser.add_argument("--patch-source-index", type=int, help="Patch each target block with this in-sequence visual block instead of zeroing it.")
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    ).eval()
    rows = load_counterfactual_jsonl(args.data)
    rows = [row for row in rows if not args.family or row.get("task_family") == args.family]
    groups = _groups(rows)
    if not groups:
        raise ValueError("no matching counterfactual prefix groups")

    layer_indices = sorted({
        layer for name, _ in model.named_modules()
        if (layer := decoder_layer_index(name)) is not None
    })
    if not layer_indices:
        raise RuntimeError("could not discover decoder layers")
    layer_groups = _layer_groups(max(layer_indices) + 1)
    layer_groups = {name: layer_groups[name] for name in args.layer_groups}

    baseline_scored = _score_groups(model, processor, groups, args.device)
    baseline = _compact(summarize_policy_scores(baseline_scored))
    conditions = []
    for image_index in args.image_indices:
        for target in args.targets:
            for group_name, layers in layer_groups.items():
                ablator = attach_kv_block_ablator(model, target=target, layers=layers)
                try:
                    scored = _score_groups(
                        model, processor, groups, args.device,
                        kv_ablator=ablator, ablate_image_index=image_index,
                        ablate_source_image_index=args.patch_source_index,
                    )
                finally:
                    ablator.close()
                summary = _compact(summarize_policy_scores(scored))
                conditions.append({
                    "image_index": image_index,
                    "intervention": "patch" if args.patch_source_index is not None else "zero",
                    "patch_source_index": args.patch_source_index,
                    "target": target,
                    "layer_group": group_name,
                    "layers": layers,
                    "evaluation": summary,
                    "critical_accuracy_delta": (
                        summary["critical_fork_accuracy"] - baseline["critical_fork_accuracy"]
                    ),
                    "signed_score_effects": signed_score_effects(baseline_scored, scored, image_index),
                })

    result = {
        "model": args.model,
        "data": args.data,
        "family": args.family,
        "layer_count": max(layer_indices) + 1,
        "baseline": baseline,
        "conditions": conditions,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
