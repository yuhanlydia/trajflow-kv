#!/usr/bin/env python3
"""Create TANGO-v2 teacher labels from matched latent-KV interventions."""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import torch

try:
    from scripts.evaluate_counterfactual_qwen import _groups, _score_groups
    from scripts.estimate_memory_advantages_qwen import expected_continuation_return
except ModuleNotFoundError:
    from evaluate_counterfactual_qwen import _groups, _score_groups
    from estimate_memory_advantages_qwen import expected_continuation_return
from trajflow_kv.causal_ablation import attach_kv_block_ablator
from trajflow_kv.counterfactual import load_counterfactual_jsonl, write_counterfactual_jsonl


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise ValueError("percentile requires values")
    position = (len(ordered) - 1) * q
    lower, upper = int(position), min(int(position) + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def aggregate_matched_patch_credit(
    rows: list[dict],
    full_groups: list[list[dict]],
    patched: dict[int, dict[int, list[list[dict]]]],
    *, temperature: float = 1.0,
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 0,
) -> tuple[list[dict], dict]:
    """Aggregate Q(full)-Q(patch) over multiple matched donors per block."""
    data_groups = _groups(rows)
    if len(data_groups) != len(full_groups):
        raise ValueError("full score groups do not match data groups")
    rng = random.Random(bootstrap_seed)
    labels_by_prefix: dict[str, list[float]] = {}
    records = []
    for group_index, (data_group, full_group) in enumerate(zip(data_groups, full_groups, strict=True)):
        prefix_id = str(data_group[0]["prefix_id"])
        history_count = len(data_group[0].get("history_images", []))
        q_full = expected_continuation_return(full_group, temperature)
        labels = []
        for history_index in range(history_count):
            donor_scores = patched.get(history_index, {})
            effects = []
            for donor_index, score_groups in sorted(donor_scores.items()):
                if len(score_groups) != len(data_groups):
                    raise ValueError("patched score groups do not match data groups")
                q_patch = expected_continuation_return(score_groups[group_index], temperature)
                effects.append((donor_index, q_full - q_patch, q_patch))
            if not effects:
                raise ValueError(f"history block {history_index} has no matched donors")
            values = [effect for _, effect, _ in effects]
            mean = sum(values) / len(values)
            boot = [sum(rng.choice(values) for _ in values) / len(values) for _ in range(bootstrap_samples)]
            labels.append(mean)
            records.append({
                "prefix_id": prefix_id, "history_index": history_index,
                "Q_memory_full": q_full, "memory_advantage": mean,
                "donor_count": len(values), "donors": [
                    {"history_index": donor, "Q_memory_patch": q_patch, "effect": effect}
                    for donor, effect, q_patch in effects
                ],
                "donor_bootstrap_ci95": [_percentile(boot, 0.025), _percentile(boot, 0.975)],
            })
        labels_by_prefix[prefix_id] = labels
    annotated = []
    for row in rows:
        item = copy.deepcopy(row)
        item["memory_advantages"] = labels_by_prefix[str(row["prefix_id"])]
        item["memory_advantage_source"] = "matched_latent_kv_patch_q"
        item.setdefault("prefix", {})["memory_advantages"] = list(item["memory_advantages"])
        annotated.append(item)
    values = [record["memory_advantage"] for record in records]
    return annotated, {
        "prefixes": len(data_groups), "history_blocks": len(records),
        "mean_memory_advantage": sum(values) / len(values) if values else None,
        "positive_blocks": sum(value > 0 for value in values),
        "negative_blocks": sum(value < 0 for value in values),
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--target", choices=("k", "v", "both"), default="k")
    parser.add_argument("--layers", nargs="+", type=int, default=list(range(12, 24)))
    parser.add_argument("--donor-indices", nargs="+", type=int, required=True)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--max-prefixes", type=int)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    rows = load_counterfactual_jsonl(args.data)
    groups = _groups(rows)
    if args.max_prefixes is not None:
        groups = groups[:args.max_prefixes]
        rows = [row for group in groups for row in group]
    history_count = len(groups[0][0].get("history_images", []))
    history_indices = list(range(history_count))
    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(args.model, dtype=torch.bfloat16, device_map=args.device).eval()
    full = _score_groups(model, processor, groups, args.device)
    ablator = attach_kv_block_ablator(model, target=args.target, layers=args.layers)
    patched: dict[int, dict[int, list[list[dict]]]] = {}
    try:
        for history_index in history_indices:
            donors = [index for index in args.donor_indices if index != history_index]
            if not donors: raise ValueError(f"no donor remains for history index {history_index}")
            patched[history_index] = {
                donor: _score_groups(model, processor, groups, args.device, kv_ablator=ablator,
                                     ablate_image_index=history_index, ablate_source_image_index=donor)
                for donor in donors
            }
    finally:
        ablator.close()
    annotated, summary = aggregate_matched_patch_credit(
        rows, full, patched, temperature=args.temperature,
        bootstrap_samples=args.bootstrap_samples,
    )
    write_counterfactual_jsonl(annotated, args.output)
    summary.update({"model": args.model, "data": args.data, "target": args.target, "layers": args.layers,
                    "donor_indices": args.donor_indices, "temperature": args.temperature})
    destination = Path(args.summary_output); destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if key != "records"}, indent=2))


if __name__ == "__main__":
    main()
