"""Losses and a tiny tabular runner for same-prefix TANGO experiments.

The functions in this module intentionally accept ordinary tensors, so they
can be used by the Qwen KV hook trainer as well as by the deterministic toy
benchmark.  ``global_return_loss`` is the existing episode-level baseline;
``tango_advantage_loss`` consumes state-conditioned counterfactual advantages.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .counterfactual import build_counterfactual_dataset, write_counterfactual_jsonl
from .objective import normalized_advantages, trajectory_policy_loss


def tango_advantage_loss(
    action_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    detach_advantages: bool = True,
) -> torch.Tensor:
    """Policy-gradient loss for candidate actions at identical prefixes."""
    if action_logprobs.shape != advantages.shape:
        raise ValueError("one action log-probability and advantage required per row")
    weights = advantages.detach() if detach_advantages else advantages
    return -(weights * action_logprobs).mean()


def global_return_loss(
    trajectory_logprob_sums: torch.Tensor,
    returns: torch.Tensor,
    task_ids: list[str],
) -> torch.Tensor:
    """Episode-return baseline retained for a direct TANGO comparison."""
    advantages = normalized_advantages(returns, task_ids)
    return trajectory_policy_loss(trajectory_logprob_sums, advantages)


def action_ce_loss(logits: torch.Tensor, action_indices: torch.Tensor) -> torch.Tensor:
    """Single-step action imitation baseline."""
    if logits.ndim != 2 or logits.shape[0] != action_indices.shape[0]:
        raise ValueError("logits must be [batch, actions] and labels [batch]")
    return F.cross_entropy(logits, action_indices)


def candidate_advantages_from_rows(rows: list[dict], device: str | torch.device = "cpu") -> torch.Tensor:
    """Extract serialized TANGO advantages for use in a minibatch."""
    return torch.tensor([float(row["advantage"]) for row in rows], dtype=torch.float32, device=device)


def counterfactual_objective_loss(
    action_logprobs: torch.Tensor,
    rows: list[dict],
    *,
    objective: str = "tango",
) -> torch.Tensor:
    """Compute one same-prefix candidate-group objective.

    ``action_logprobs`` must follow the candidate order in ``rows``.  This
    small tensor-only function is shared by the Qwen hook trainer and tests,
    keeping the policy objective independent from the tokenizer/model.  For
    the CE control, an explicit ``target_action``/``target_index`` in a row is
    honored; otherwise the highest-Q candidate is the oracle label supplied by
    the counterfactual evaluator.
    """
    if objective not in {"tango", "global_return", "ce"}:
        raise ValueError("objective must be tango, global_return, or ce")
    if action_logprobs.ndim != 1 or action_logprobs.shape[0] != len(rows):
        raise ValueError("one action log-probability required per candidate row")
    if not rows:
        raise ValueError("rows must not be empty")
    if objective == "tango":
        return tango_advantage_loss(
            action_logprobs,
            candidate_advantages_from_rows(rows, action_logprobs.device),
        )
    if objective == "global_return":
        returns = torch.tensor(
            [float(row["Q"]) for row in rows],
            dtype=action_logprobs.dtype,
            device=action_logprobs.device,
        )
        # A prefix is the task/state group for this objective.  The baseline
        # is deliberately computed over the same candidate set, while TANGO
        # consumes the evaluator's explicit Q-V advantage.
        advantages = normalized_advantages(returns, [str(rows[0].get("prefix_id", "prefix"))] * len(rows))
        return trajectory_policy_loss(action_logprobs, advantages)

    target = rows[0].get("target_index")
    if target is None and rows[0].get("target_action") is not None:
        target_action = str(rows[0]["target_action"])
        target = next(
            (index for index, row in enumerate(rows) if str(row["action"]) == target_action),
            None,
        )
    if target is None:
        target = max(range(len(rows)), key=lambda index: float(rows[index]["Q"]))
    target = int(target)
    if not 0 <= target < len(rows):
        raise ValueError("target_index is outside the candidate group")
    return action_ce_loss(action_logprobs[None, :], torch.tensor([target], device=action_logprobs.device))


class TabularPrefixPolicy(nn.Module):
    """Tiny prefix-conditioned policy used by the no-download CLI smoke run."""

    def __init__(self, num_prefixes: int, num_actions: int):
        super().__init__()
        self.logits = nn.Embedding(num_prefixes, num_actions)

    def forward(self, prefix_indices: torch.Tensor) -> torch.Tensor:
        return self.logits(prefix_indices)


def train_tabular_counterfactual(
    rows: list[dict],
    *,
    objective: str = "tango",
    epochs: int = 100,
    lr: float = 0.1,
    seed: int = 0,
) -> tuple[TabularPrefixPolicy, list[dict]]:
    """Train a tiny policy with TANGO, global-return, or CE objective.

    This runner is a deterministic contract test and a cheap diagnostic for
    the data protocol.  Real VLM training should use ``tango_advantage_loss``
    inside the existing K/V hook trainer.
    """
    if objective not in {"tango", "global_return", "ce"}:
        raise ValueError("objective must be tango, global_return, or ce")
    if not rows:
        raise ValueError("rows must not be empty")
    torch.manual_seed(seed)
    prefix_names = sorted({str(row["prefix_id"]) for row in rows})
    prefix_to_index = {name: i for i, name in enumerate(prefix_names)}
    actions = sorted({str(action) for row in rows for action in row["candidate_actions"]})
    action_to_index = {name: i for i, name in enumerate(actions)}
    model = TabularPrefixPolicy(len(prefix_names), len(actions))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    prefix_indices = torch.tensor([prefix_to_index[str(row["prefix_id"])] for row in rows])
    action_indices = torch.tensor([action_to_index[str(row["action"])] for row in rows])
    advantages = candidate_advantages_from_rows(rows)
    history: list[dict] = []
    for epoch in range(epochs):
        logits = model(prefix_indices)
        log_probs = logits.log_softmax(-1).gather(1, action_indices[:, None]).squeeze(1)
        if objective == "tango":
            loss = tango_advantage_loss(log_probs, advantages)
        elif objective == "ce":
            loss = action_ce_loss(logits, action_indices)
        else:
            # A row's Q is the return label for this candidate. Grouped by
            # prefix, this is the offline global-return control interface.
            returns = torch.tensor([float(row["Q"]) for row in rows])
            task_ids = [str(row["prefix_id"]) for row in rows]
            loss = global_return_loss(log_probs, returns, task_ids)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach())})
    return model, history


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/train the TANGO delayed-consequence toy benchmark")
    parser.add_argument("--output", type=Path, default=Path("data/toy/tango_counterfactual.jsonl"))
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=6)
    parser.add_argument("--aggregation", choices=("mean", "max"), default="mean")
    parser.add_argument("--objective", choices=("generate", "tango", "global_return", "ce"), default="generate")
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()
    rows = build_counterfactual_dataset(range(args.seeds), horizon=args.horizon, aggregation=args.aggregation)
    count = write_counterfactual_jsonl(rows, args.output)
    result = {"rows": count, "output": str(args.output), "families": sorted({r["task_family"] for r in rows})}
    if args.objective != "generate":
        model, history = train_tabular_counterfactual(rows, objective=args.objective, epochs=args.epochs)
        result["final_loss"] = history[-1]["loss"]
        result["prefixes"] = int(model.logits.num_embeddings)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
