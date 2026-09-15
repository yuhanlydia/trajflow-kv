#!/usr/bin/env python3
"""Score same-prefix counterfactual GUI candidates with a Qwen VLM.

The script evaluates every candidate action under the exact same visual
prefix, using teacher-forced mean token log-probability.  It can score a
projector checkpoint and an optional baseline in one model process, which is
important for 3B VLMs on small GPUs.  With no checkpoint, the projector is
kept at its zero-residual initialization and acts as the base VLM.

The JSON output reports candidate top-1, critical-fork accuracy, per-family
breakdowns, and (when a baseline is supplied) score changes on non-critical
candidate rows.  It deliberately does not turn a higher score margin into an
online success claim: this is a controlled counterfactual ranking diagnostic.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch
from PIL import Image

from trajflow_kv.counterfactual import load_counterfactual_jsonl
from trajflow_kv.causal_ablation import vision_token_spans
from trajflow_kv.projector import attach_gated_kv_projectors, attach_kv_projectors
from trajflow_kv.train import _counterfactual_prompt


def _groups(rows: Iterable[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group rows by immutable prefix while preserving first-seen order."""
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        prefix_id = str(row.get("prefix_id", ""))
        if not prefix_id:
            raise ValueError("counterfactual rows require prefix_id")
        grouped.setdefault(prefix_id, []).append(row)
    return list(grouped.values())


def _path_from_row(row: dict[str, Any]) -> Path | None:
    image = row.get("image") or (row.get("prefix") or {}).get("image")
    if not image:
        return None
    path = Path(str(image))
    if path.exists():
        return path
    # JSONL files are often moved together with their image directory.  Try
    # the JSONL-relative spelling before giving a useful FileNotFoundError.
    data_path = row.get("_data_path")
    if data_path:
        candidate = Path(str(data_path)).parent / path
        if candidate.exists():
            return candidate
    return path


def _optimal_actions(group: list[dict[str, Any]]) -> set[str]:
    actions: set[str] = set()
    for row in group:
        actions.update(str(action) for action in row.get("optimal_actions", []))
    if actions:
        return actions
    # Text-only counterfactual rows do not carry an explicit optimal-action
    # annotation.  Their evaluator-provided Q is the oracle for this metric.
    best_q = max(float(row.get("Q", 0.0)) for row in group)
    return {str(row["action"]) for row in group if float(row.get("Q", 0.0)) == best_q}


def _is_critical_group(group: list[dict[str, Any]]) -> bool:
    return any(bool(row.get("critical_step")) or bool(row.get("is_critical_action")) for row in group)


def summarize_policy_scores(
    scored_groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Summarize scored candidate groups without requiring a VLM.

    Each row must contain ``score``.  The selected candidate is the maximum
    score, with stable first-seen tie breaking.  Candidate top-1 is measured
    against the evaluator's ``optimal_actions`` (or max-Q fallback).
    """
    prefix_results: list[dict[str, Any]] = []
    for group in scored_groups:
        if not group:
            continue
        selected = max(group, key=lambda row: float(row["score"]))
        optimal = _optimal_actions(group)
        critical = _is_critical_group(group)
        prefix_results.append({
            "prefix_id": str(group[0]["prefix_id"]),
            "task_family": str(group[0].get("task_family", "unknown")),
            "seed": group[0].get("seed"),
            "selected_action": str(selected["action"]),
            "optimal_actions": sorted(optimal),
            "selected_score": float(selected["score"]),
            "candidate_top1": str(selected["action"]) in optimal,
            "critical_step": critical,
            "critical_fork_correct": (str(selected["action"]) in optimal) if critical else None,
            "candidates": [
                {
                    "action": str(row["action"]),
                    "score": float(row["score"]),
                    "Q": float(row.get("Q", 0.0)),
                    "advantage": float(row.get("advantage", 0.0)),
                    "is_critical_action": bool(row.get("is_critical_action", False)),
                    **({"memory_gate_means": row["memory_gate_means"]} if "memory_gate_means" in row else {}),
                }
                for row in group
            ],
        })

    def _rate(items: list[dict[str, Any]], key: str) -> float | None:
        return (sum(bool(item[key]) for item in items) / len(items)) if items else None

    critical_rows = [item for item in prefix_results if item["critical_step"]]
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in prefix_results:
        by_family[item["task_family"]].append(item)
    family_summary = {}
    for family, items in sorted(by_family.items()):
        family_critical = [item for item in items if item["critical_step"]]
        family_summary[family] = {
            "prefixes": len(items),
            "candidate_top1_accuracy": _rate(items, "candidate_top1"),
            "critical_prefixes": len(family_critical),
            "critical_fork_accuracy": _rate(family_critical, "critical_fork_correct"),
        }
    return {
        "prefixes": len(prefix_results),
        "candidate_top1_accuracy": _rate(prefix_results, "candidate_top1"),
        "critical_prefixes": len(critical_rows),
        "critical_fork_accuracy": _rate(critical_rows, "critical_fork_correct"),
        "families": family_summary,
        "prefix_results": prefix_results,
    }


def summarize_memory_gate_values(prefix_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-history-block gates once per immutable prefix."""
    buckets: dict[tuple[str, bool, int], list[float]] = defaultdict(list)
    for result in prefix_results:
        candidates = result.get("candidates", [])
        if not candidates or "memory_gate_means" not in candidates[0]:
            continue
        for index, value in enumerate(candidates[0]["memory_gate_means"]):
            if value is not None:
                buckets[(result["task_family"], bool(result["critical_step"]), index)].append(float(value))
    return {
        f"{family}|{'critical' if critical else 'noncritical'}|history_{index}": {
            "prefixes": len(values),
            "mean_gate": sum(values) / len(values),
        }
        for (family, critical, index), values in sorted(buckets.items())
    }


def summarize_noncritical_score_change(
    target_groups: list[list[dict[str, Any]]],
    baseline_groups: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Compare target and baseline scores on non-critical candidate rows."""
    baseline = {
        (str(row["prefix_id"]), str(row["action"])): float(row["score"])
        for group in baseline_groups for row in group
    }
    deltas: list[dict[str, Any]] = []
    for group in target_groups:
        for row in group:
            if bool(row.get("critical_step")) or bool(row.get("is_critical_action")):
                continue
            key = (str(row["prefix_id"]), str(row["action"]))
            if key not in baseline:
                continue
            delta = float(row["score"]) - baseline[key]
            deltas.append({
                "prefix_id": key[0],
                "task_family": str(row.get("task_family", "unknown")),
                "action": key[1],
                "baseline_score": baseline[key],
                "target_score": float(row["score"]),
                "delta": delta,
            })
    values = [item["delta"] for item in deltas]
    by_family: dict[str, list[float]] = defaultdict(list)
    for item in deltas:
        by_family[item["task_family"]].append(item["delta"])
    return {
        "candidates": len(values),
        "mean_delta": sum(values) / len(values) if values else None,
        "mean_absolute_delta": sum(abs(value) for value in values) / len(values) if values else None,
        "families": {
            family: {
                "candidates": len(family_values),
                "mean_delta": sum(family_values) / len(family_values),
                "mean_absolute_delta": sum(abs(value) for value in family_values) / len(family_values),
            }
            for family, family_values in sorted(by_family.items())
        },
        "rows": deltas,
    }


def _load_projector_state(bundle: Any, checkpoint: str | None, zero_state: dict[str, torch.Tensor], device: str) -> None:
    if checkpoint:
        state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
        bundle.modules.load_state_dict(state)
    else:
        bundle.modules.load_state_dict(zero_state)


def _score_groups(
    model: Any,
    processor: Any,
    groups: list[list[dict[str, Any]]],
    device: str,
    drop_history_index: int | None = None,
    kv_ablator: Any | None = None,
    ablate_image_index: int = 0,
    ablate_source_image_index: int | None = None,
    memory_bundle: Any | None = None,
) -> list[list[dict[str, Any]]]:
    scored: list[list[dict[str, Any]]] = []
    with torch.inference_mode():
        for group in groups:
            scored_group: list[dict[str, Any]] = []
            for row in group:
                history_paths = []
                history_images = row.get("history_images", []) or (row.get("prefix") or {}).get("history_images", [])
                if drop_history_index is not None:
                    history_images = [
                        image for index, image in enumerate(history_images)
                        if index != drop_history_index
                    ]
                for history_image in history_images:
                    history_path = Path(str(history_image))
                    if not history_path.exists():
                        data_path = row.get("_data_path")
                        if data_path:
                            history_path = Path(str(data_path)).parent / history_path
                    if history_path.exists():
                        history_paths.append(history_path)
                image_path = _path_from_row(row)
                image = Image.open(image_path).convert("RGB") if image_path else None
                prompt = str(row["prompt"]) if row.get("prompt") else _counterfactual_prompt(row)
                content = []
                for history_path in history_paths:
                    content.append({"type": "image", "image": Image.open(history_path).convert("RGB")})
                if image is not None:
                    content.append({"type": "image", "image": image})
                content.append({"type": "text", "text": prompt})
                user = {"role": "user", "content": content}
                assistant = {"role": "assistant", "content": [{"type": "text", "text": str(row["action"])}]}
                prompt_batch = processor.apply_chat_template(
                    [user], tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt",
                )
                batch = processor.apply_chat_template(
                    [user, assistant], tokenize=True,
                    return_dict=True, return_tensors="pt",
                ).to(device)
                labels = batch["input_ids"].clone()
                labels[:, :prompt_batch["input_ids"].shape[1]] = -100
                if kv_ablator is not None:
                    kv_ablator.set_image(batch["input_ids"], ablate_image_index, ablate_source_image_index)
                if memory_bundle is not None:
                    memory_bundle.set_visual_memory_context(
                        batch["input_ids"], len(history_paths),
                        prompt_batch["input_ids"].shape[1] - 1,
                    )
                try:
                    output = model(**batch, labels=labels, use_cache=False)
                finally:
                    if kv_ablator is not None:
                        kv_ablator.clear()
                    if memory_bundle is not None:
                        memory_bundle.clear_context()
                memory_gate_means = None
                if memory_bundle is not None:
                    spans = vision_token_spans(batch["input_ids"])
                    gate_means = []
                    for start, end in spans[: len(history_paths)]:
                        values = [
                            module.last_gate[..., start:end, :].float().mean()
                            for module in memory_bundle.modules
                            if getattr(module, "last_gate", None) is not None
                        ]
                        gate_means.append(float(torch.stack(values).mean()) if values else None)
                    memory_gate_means = gate_means
                valid = (labels != -100).sum().clamp_min(1)
                score = -output.loss * valid.to(output.loss.dtype) / valid
                scored_group.append({
                    **row, "score": float(score),
                    **({"memory_gate_means": memory_gate_means} if memory_gate_means is not None else {}),
                })
            scored.append(scored_group)
    return scored


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="models/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--data", required=True, help="Counterfactual JSONL, usually visual delayed data")
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint", help="Target projector checkpoint; omit for zero-residual base")
    parser.add_argument("--baseline-checkpoint", help="Optional baseline projector checkpoint")
    parser.add_argument("--compare-zero", action="store_true", help="Compare against the zero-residual projector")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=float, default=8.0)
    parser.add_argument("--target", choices=("k", "v", "both"), default="v")
    parser.add_argument("--last-n-layers", type=int, default=8)
    parser.add_argument("--state-conditioned-gate", action="store_true")
    parser.add_argument("--gated-layers", nargs="+", type=int)
    parser.add_argument("--gate-rank", type=int, default=16)
    parser.add_argument("--signed-gate", action="store_true")
    parser.add_argument("--max-pixels", type=int, default=100352)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-prefixes", type=int)
    parser.add_argument(
        "--drop-history-index", type=int,
        help="Omit one prior screenshot from every prefix for causal memory ablation.",
    )
    args = parser.parse_args()

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(args.model, max_pixels=args.max_pixels)
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map=args.device
    ).eval()
    if args.state_conditioned_gate:
        bundle = attach_gated_kv_projectors(
            model, args.rank, args.alpha, args.target,
            layers=args.gated_layers, gate_rank=args.gate_rank, signed_gate=args.signed_gate,
        )
    else:
        bundle = attach_kv_projectors(
            model, args.rank, args.alpha, args.target, args.last_n_layers
        )
    zero_state = {name: value.detach().clone() for name, value in bundle.modules.state_dict().items()}
    rows = load_counterfactual_jsonl(args.data)
    for row in rows:
        row["_data_path"] = args.data
    groups = _groups(rows)
    if args.max_prefixes is not None:
        groups = groups[: args.max_prefixes]
    if not groups:
        raise ValueError("counterfactual data contains no prefix groups")

    def evaluate(checkpoint: str | None) -> dict[str, Any]:
        _load_projector_state(bundle, checkpoint, zero_state, args.device)
        return summarize_policy_scores(
            _score_groups(
                model, processor, groups, args.device, args.drop_history_index,
                memory_bundle=bundle if args.state_conditioned_gate else None,
            )
        )

    baseline = evaluate(args.baseline_checkpoint) if (args.baseline_checkpoint or args.compare_zero) else None
    target = evaluate(args.checkpoint)
    target_scored = target.pop("prefix_results")
    # Reconstruct the compact scored groups needed for the optional delta
    # calculation from the serialized per-prefix results.
    target_groups = [[
        {**candidate, "prefix_id": result["prefix_id"], "task_family": result["task_family"],
         "critical_step": result["critical_step"]}
        for candidate in result["candidates"]
    ] for result in target_scored]
    summary: dict[str, Any] = {
        "model": args.model,
        "data": args.data,
        "checkpoint": args.checkpoint,
        "baseline_checkpoint": args.baseline_checkpoint,
        "compare_zero": args.compare_zero,
        "rank": args.rank,
        "alpha": args.alpha,
        "target": args.target,
        "last_n_layers": args.last_n_layers,
        "drop_history_index": args.drop_history_index,
        "state_conditioned_gate": args.state_conditioned_gate,
        "gated_layers": args.gated_layers,
        "evaluation": target,
    }
    if args.state_conditioned_gate:
        summary["memory_gate_summary"] = summarize_memory_gate_values(target_scored)
    if baseline is not None:
        baseline_results = baseline.pop("prefix_results")
        baseline_groups = [[
            {**candidate, "prefix_id": result["prefix_id"], "task_family": result["task_family"],
             "critical_step": result["critical_step"]}
            for candidate in result["candidates"]
        ] for result in baseline_results]
        summary["baseline_evaluation"] = baseline
        summary["noncritical_score_change"] = summarize_noncritical_score_change(
            target_groups, baseline_groups
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    printable = {key: value for key, value in summary.items() if key not in {"evaluation", "baseline_evaluation"}}
    printable["evaluation"] = {key: value for key, value in target.items() if key != "families"}
    if baseline is not None:
        printable["baseline_evaluation"] = {key: value for key, value in baseline.items() if key != "families"}
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
