from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
import yaml
from torch import nn
from torch.nn import functional as F

from .causal_ablation import vision_token_spans
from .data import load_jsonl
from .counterfactual import load_counterfactual_jsonl
from .objective import normalized_advantages, shuffle_within_tasks, trajectory_policy_loss
from .projector import attach_gated_kv_projectors, attach_kv_projectors
from .training_controls import controlled_returns, remove_prompt_history, select_trajectory_steps
from .tango_advantage import counterfactual_objective_loss


def memory_gate_supervision_loss(
    gates: torch.Tensor,
    advantages: torch.Tensor,
    *,
    scale: float,
    loss_type: str = "bce",
    signed_gate: bool = False,
    huber_beta: float = 0.1,
) -> torch.Tensor:
    """Distill causal memory credit into unsigned or signed gates."""
    if scale <= 0:
        raise ValueError("memory advantage scale must be positive")
    if gates.shape != advantages.shape:
        raise ValueError("memory advantage count does not match history gate count")
    if loss_type == "bce":
        targets = (advantages.clamp_min(0) / scale).clamp_max(1)
        return F.binary_cross_entropy(gates.float().clamp(1e-6, 1 - 1e-6), targets.float())
    if loss_type == "huber":
        if not signed_gate:
            raise ValueError("Huber signed credit supervision requires signed_gate=true")
        targets = (advantages / scale).clamp(-1, 1)
        return F.smooth_l1_loss(gates.float(), targets.float(), beta=huber_beta)
    raise ValueError("memory_loss must be bce or huber")


class ToyKVPolicy(nn.Module):
    def __init__(self, width: int = 16, actions: int = 5):
        super().__init__()
        self.k_proj = nn.Linear(width, width, bias=False)
        self.v_proj = nn.Linear(width, width, bias=False)
        self.head = nn.Linear(width, actions, bias=False)

    def forward(self, x):
        return self.head(torch.tanh(self.k_proj(x) + self.v_proj(x)))


def toy_batch(device: str):
    generator = torch.Generator(device=device).manual_seed(19)
    x = torch.randn(8, 16, generator=generator, device=device)
    actions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2], device=device)
    returns = torch.tensor([1., 0., 1., 0., 1., 0., 0., 1.], device=device)
    task_ids = ["a", "a", "b", "b", "c", "c", "d", "d"]
    return x, actions, returns, task_ids


def train_toy(cfg):
    device = cfg["device"]
    model = ToyKVPolicy().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    bundle = attach_kv_projectors(
        model,
        cfg["rank"],
        cfg["alpha"],
        cfg["target"],
        cfg.get("last_n_layers"),
    )
    optimizer = torch.optim.AdamW(bundle.modules.parameters(), lr=cfg["lr"])
    history = []
    for epoch in range(cfg["epochs"]):
        x, actions, returns, task_ids = toy_batch(device)
        logits = model(x)
        logprobs = logits.log_softmax(-1).gather(1, actions[:, None]).squeeze(1)
        advantages = normalized_advantages(returns, task_ids)
        pg = trajectory_policy_loss(logprobs, advantages)
        loss = pg + cfg["lambda_energy"] * bundle.energy() + cfg["lambda_orth"] * bundle.orthogonality_loss()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        history.append({"epoch": epoch, "loss": float(loss.detach()), "pg": float(pg.detach())})
    return model, bundle, history


def train_qwen(cfg):
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    from .qwen_policy import build_action_prompt

    device = cfg["device"]
    processor = AutoProcessor.from_pretrained(cfg["model_path"], max_pixels=cfg["max_pixels"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16, device_map=device
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    # Training never reuses generation caches. Keeping them would retain a
    # full extra KV copy and can exhaust a 16 GiB card before backward.
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    bundle = attach_kv_projectors(
        model,
        cfg["rank"],
        cfg["alpha"],
        cfg["target"],
        cfg.get("last_n_layers"),
    )
    projector_checkpoint = cfg.get("projector_checkpoint")
    if projector_checkpoint:
        bundle.modules.load_state_dict(
            torch.load(projector_checkpoint, map_location=device, weights_only=True)
        )
    if cfg.get("gate_only") and cfg.get("freeze_gate"):
        raise ValueError("gate_only and freeze_gate are mutually exclusive")
    if cfg.get("state_conditioned_gate"):
        for name, parameter in bundle.modules.named_parameters():
            is_gate = any(part in name for part in ("query_gate", "memory_gate", "gate_bias"))
            if cfg.get("gate_only") and not is_gate:
                parameter.requires_grad_(False)
            if cfg.get("freeze_gate") and is_gate:
                parameter.requires_grad_(False)
    if cfg.get("state_conditioned_gate") and cfg.get("gate_lr") is not None:
        gate_parameters, transport_parameters = [], []
        for name, parameter in bundle.modules.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(part in name for part in ("query_gate", "memory_gate", "gate_bias")):
                gate_parameters.append(parameter)
            else:
                transport_parameters.append(parameter)
        parameter_groups = []
        if transport_parameters:
            parameter_groups.append({"params": transport_parameters, "lr": float(cfg["lr"])})
        if gate_parameters:
            parameter_groups.append({"params": gate_parameters, "lr": float(cfg["gate_lr"])})
        optimizer = torch.optim.AdamW(parameter_groups)
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in bundle.modules.parameters() if parameter.requires_grad],
            lr=cfg["lr"],
        )
    trajectories = load_jsonl(cfg["data_path"])
    trajectories = trajectories[: cfg.get("max_trajectories", len(trajectories))]

    # Teacher-forced action log probabilities retain gradient through hooked K/V.
    history = [{
        "event": "setup",
        "hooked_modules": len(bundle.names),
        "trainable_parameters": sum(
            p.numel() for p in bundle.modules.parameters() if p.requires_grad
        ),
        "first_hook": bundle.names[0],
        "last_hook": bundle.names[-1],
        "projector_checkpoint": projector_checkpoint,
        "return_mode": cfg.get("return_mode", "observed"),
        "trajectory_horizon": cfg.get("trajectory_horizon", "full"),
        "step_selection": cfg.get("step_selection", "all"),
        "remove_history": cfg.get("remove_history", False),
    }]
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    for epoch in range(cfg["epochs"]):
        observed_returns = torch.tensor(
            [float(t["return"]) for t in trajectories], device=device
        )
        returns = observed_returns.clone()
        task_ids = [t["task_id"] for t in trajectories]
        return_mode = cfg.get("return_mode", "observed")
        if return_mode == "shuffle":
            returns = shuffle_within_tasks(returns, task_ids)
        else:
            returns = controlled_returns(
                observed_returns, return_mode, seed=cfg["seed"], epoch=epoch
            )
        advantages = normalized_advantages(returns, task_ids)
        if cfg.get("lambda_action", 0.0) == 0 and torch.count_nonzero(advantages) == 0:
            raise ValueError(
                "pure return training has zero advantages; collect mixed returns "
                "for at least one repeated task or set lambda_action > 0 for imitation"
            )
        optimizer.zero_grad(set_to_none=True)
        for index, trajectory in enumerate(trajectories):
            trajectory_logprob = torch.zeros((), device=device)
            selected_steps = select_trajectory_steps(
                trajectory["steps"], cfg.get("trajectory_horizon", "full"),
                cfg.get("step_selection", "all"),
            )
            for step in selected_steps:
                content = []
                loaded_image = None
                if step.get("image"):
                    loaded_image = Image.open(step["image"]).convert("RGB")
                    content.append({"type": "image", "image": loaded_image})
                inferred_size = loaded_image.size if loaded_image else tuple(step.get("screen_size", (1000, 1000)))
                step_history = [] if cfg.get("remove_history", False) else step.get("history", [])
                prompt = step.get("prompt") or build_action_prompt(
                    trajectory["instruction"], step_history, inferred_size
                )
                if cfg.get("remove_history", False) and step.get("prompt"):
                    prompt = remove_prompt_history(prompt)
                content.append({"type": "text", "text": prompt})
                messages = [{"role": "user", "content": content},
                            {"role": "assistant", "content": [
                                {"type": "text", "text": step["action"]}
                            ]}]
                prompt_messages = messages[:1]
                prompt_batch = processor.apply_chat_template(
                    prompt_messages, tokenize=True, add_generation_prompt=True,
                    return_dict=True, return_tensors="pt"
                )
                batch = processor.apply_chat_template(messages, tokenize=True, return_dict=True,
                                                       return_tensors="pt").to(device)
                labels = batch["input_ids"].clone()
                # Supervise the complete assistant span, including the action;
                # final-N masking is wrong because chat templates add EOS tokens.
                prompt_tokens = prompt_batch["input_ids"].shape[1]
                labels[:, :prompt_tokens] = -100
                output = model(**batch, labels=labels, use_cache=False)
                valid = (labels != -100).sum().clamp_min(1)
                trajectory_logprob = trajectory_logprob - output.loss * valid
            # AITW demos have no failure return; action NLL keeps that offline
            # warm-start useful. Set lambda_action=0 for pure online return RL.
            loss = -advantages[index].detach() * trajectory_logprob
            action_weight = cfg.get("lambda_action", 0.0)
            if cfg.get("positive_action_only", False) and observed_returns[index] <= 0:
                action_weight = 0.0
            loss = loss - action_weight * trajectory_logprob
            energy = bundle.energy()
            orthogonality = bundle.orthogonality_loss()
            loss = loss + cfg["lambda_energy"] * energy + cfg["lambda_orth"] * orthogonality
            (loss / cfg["gradient_accumulation_steps"]).backward()
            if (index + 1) % cfg["gradient_accumulation_steps"] == 0 or index + 1 == len(trajectories):
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
            metric = {"epoch": epoch, "trajectory": index, "loss": float(loss.detach()),
                      "return": float(returns[index]), "advantage": float(advantages[index]),
                      "energy": float(energy.detach()),
                      "orthogonality": float(orthogonality.detach())}
            if device.startswith("cuda"):
                metric["peak_gpu_gib"] = torch.cuda.max_memory_allocated() / 2**30
            history.append(metric)
    return model, bundle, history


def _counterfactual_prompt(row: dict) -> str:
    """Render a text-only same-prefix prompt for a counterfactual row.

    The toy benchmark intentionally has no screenshot dependency.  Real
    counterfactual collectors may provide ``prompt`` or an observation with a
    screen/accessibility description; both are accepted here.
    """
    if row.get("prompt"):
        return str(row["prompt"])
    prefix = row.get("prefix") or {}
    observation = prefix.get("observation") or {}
    screen = observation.get("screen", prefix.get("screen", ""))
    history = prefix.get("history", [])
    candidates = row.get("candidate_actions", [row.get("action")])
    return (
        "You are a GUI policy. Choose exactly one action from the candidate list "
        "and return only that action, with no explanation.\n"
        f"Task: {row.get('instruction', '')}\n"
        f"Observation: {screen}\n"
        f"History: {json.dumps(history, ensure_ascii=False)}\n"
        f"Candidates: {json.dumps([str(item) for item in candidates], ensure_ascii=False)}"
    )


def _counterfactual_groups(rows: list[dict]) -> list[list[dict]]:
    """Group rows by immutable prefix and put candidates in stable order."""
    groups: dict[str, list[dict]] = {}
    for row in rows:
        prefix_id = str(row.get("prefix_id", ""))
        if not prefix_id:
            raise ValueError("counterfactual rows require prefix_id")
        groups.setdefault(prefix_id, []).append(row)
    ordered: list[list[dict]] = []
    for prefix_id, group in groups.items():
        candidate_actions = group[0].get("candidate_actions")
        if candidate_actions is None:
            candidate_actions = [row["action"] for row in group]
        by_action = {str(row["action"]): row for row in group}
        missing = [str(action) for action in candidate_actions if str(action) not in by_action]
        if missing:
            raise ValueError(f"prefix {prefix_id} is missing candidate rows: {missing}")
        ordered.append([by_action[str(action)] for action in candidate_actions])
    return ordered


def train_qwen_counterfactual(cfg):
    """Train K/V projectors on same-prefix counterfactual candidate rows.

    This is intentionally a separate path from trajectory JSONL training.  It
    scores every legal candidate under the *same* text prefix, then applies
    the requested TANGO (Q-V), global-return, or oracle-CE objective.  Existing
    ``data_path`` training remains unchanged when ``counterfactual_data`` is
    absent.
    """
    from PIL import Image
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    objective = cfg.get("objective", "tango")
    if objective not in {"tango", "global_return", "ce"}:
        raise ValueError("objective must be tango, global_return, or ce")
    device = cfg["device"]
    processor = AutoProcessor.from_pretrained(cfg["model_path"], max_pixels=cfg["max_pixels"])
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        cfg["model_path"], dtype=torch.bfloat16, device_map=device
    )
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    if cfg.get("state_conditioned_gate"):
        bundle = attach_gated_kv_projectors(
            model,
            cfg["rank"],
            cfg["alpha"],
            cfg["target"],
            layers=cfg.get("gated_layers"),
            gate_rank=int(cfg.get("gate_rank", 16)),
            signed_gate=bool(cfg.get("signed_gate", False)),
        )
    else:
        bundle = attach_kv_projectors(
            model, cfg["rank"], cfg["alpha"], cfg["target"], cfg.get("last_n_layers")
        )
    projector_checkpoint = cfg.get("projector_checkpoint")
    if projector_checkpoint:
        bundle.modules.load_state_dict(
            torch.load(projector_checkpoint, map_location=device, weights_only=True)
        )
    if cfg.get("gate_only") and cfg.get("freeze_gate"):
        raise ValueError("gate_only and freeze_gate are mutually exclusive")
    if cfg.get("state_conditioned_gate"):
        for name, parameter in bundle.modules.named_parameters():
            is_gate = any(part in name for part in ("query_gate", "memory_gate", "gate_bias"))
            if cfg.get("gate_only") and not is_gate:
                parameter.requires_grad_(False)
            if cfg.get("freeze_gate") and is_gate:
                parameter.requires_grad_(False)
    if cfg.get("state_conditioned_gate") and cfg.get("gate_lr") is not None:
        gate_parameters, transport_parameters = [], []
        for name, parameter in bundle.modules.named_parameters():
            if not parameter.requires_grad:
                continue
            if any(part in name for part in ("query_gate", "memory_gate", "gate_bias")):
                gate_parameters.append(parameter)
            else:
                transport_parameters.append(parameter)
        parameter_groups = []
        if transport_parameters:
            parameter_groups.append({"params": transport_parameters, "lr": float(cfg["lr"])})
        if gate_parameters:
            parameter_groups.append({"params": gate_parameters, "lr": float(cfg["gate_lr"])})
        optimizer = torch.optim.AdamW(parameter_groups)
    else:
        optimizer = torch.optim.AdamW(
            [parameter for parameter in bundle.modules.parameters() if parameter.requires_grad],
            lr=cfg["lr"],
        )
    rows = load_counterfactual_jsonl(cfg["counterfactual_data"])
    groups = _counterfactual_groups(rows)
    if not groups:
        raise ValueError("counterfactual_data must contain at least one prefix")
    max_rows = cfg.get("max_counterfactual_rows")
    if max_rows is not None and max_rows < len(rows):
        # Never truncate inside a prefix: all candidates are needed to form a
        # valid same-prefix counterfactual advantage.
        selected_groups = []
        selected_rows = 0
        for group in groups:
            if selected_groups and selected_rows + len(group) > int(max_rows):
                break
            selected_groups.append(group)
            selected_rows += len(group)
        groups = selected_groups
        rows = [row for group in groups for row in group]
    accumulation = int(cfg.get("gradient_accumulation_steps", 1))
    history = [{
        "event": "setup",
        "objective": objective,
        "counterfactual_data": cfg["counterfactual_data"],
        "prefixes": len(groups),
        "rows": len(rows),
        "hooked_modules": len(bundle.names),
        "trainable_parameters": sum(
            p.numel() for p in bundle.modules.parameters() if p.requires_grad
        ),
        "projector_checkpoint": projector_checkpoint,
    }]
    optimizer.zero_grad(set_to_none=True)

    def action_score(row: dict, action: object) -> tuple[torch.Tensor, torch.Tensor | None]:
        content = []
        for history_image_path in row.get("history_images", []) or (row.get("prefix") or {}).get("history_images", []):
            history_image = Image.open(history_image_path).convert("RGB")
            content.append({"type": "image", "image": history_image})
        image_path = row.get("image") or (row.get("prefix") or {}).get("image")
        loaded_image = Image.open(image_path).convert("RGB") if image_path else None
        if loaded_image is not None:
            content.append({"type": "image", "image": loaded_image})
        content.append({"type": "text", "text": _counterfactual_prompt(row)})
        user = {"role": "user", "content": content}
        assistant = {"role": "assistant", "content": [{"type": "text", "text": str(action)}]}
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
        if cfg.get("state_conditioned_gate"):
            history_count = len(
                row.get("history_images", [])
                or (row.get("prefix") or {}).get("history_images", [])
            )
            bundle.set_visual_memory_context(
                batch["input_ids"], history_count, prompt_batch["input_ids"].shape[1] - 1
            )
        try:
            output = model(**batch, labels=labels, use_cache=False)
        finally:
            bundle.clear_context()
        memory_gates = None
        if cfg.get("state_conditioned_gate") and history_count:
            spans = vision_token_spans(batch["input_ids"])
            gate_values = []
            for start, end in spans[:history_count]:
                values = [
                    module.last_gate[..., start:end, :].float().mean()
                    for module in bundle.modules
                    if getattr(module, "last_gate", None) is not None
                ]
                gate_values.append(torch.stack(values).mean())
            memory_gates = torch.stack(gate_values) if gate_values else None
        valid = (labels != -100).sum().clamp_min(1)
        # Mean token log-prob is consistent with existing candidate ranking
        # and prevents longer JSON actions from winning by length alone.
        return -output.loss * valid.to(output.loss.dtype) / valid, memory_gates

    for epoch in range(int(cfg["epochs"])):
        for group_index, group in enumerate(groups):
            scored = [action_score(row, row["action"]) for row in group]
            scores = torch.stack([item[0] for item in scored])
            loss = counterfactual_objective_loss(scores, group, objective=objective)
            memory_loss = torch.zeros((), device=device)
            targets = group[0].get("memory_advantages", [])
            if cfg.get("state_conditioned_gate") and targets and scored[0][1] is not None:
                target_tensor = torch.tensor(targets, device=device, dtype=torch.float32)
                if target_tensor.shape != scored[0][1].shape:
                    raise ValueError("memory advantage count does not match history gate count")
                scale = float(cfg.get("memory_advantage_scale", 1.0))
                memory_loss_type = str(cfg.get("memory_loss", "bce"))
                memory_loss = memory_gate_supervision_loss(
                    scored[0][1], target_tensor, scale=scale,
                    loss_type=memory_loss_type,
                    signed_gate=bool(cfg.get("signed_gate", False)),
                    huber_beta=float(cfg.get("memory_huber_beta", 0.1)),
                )
                loss = loss + float(cfg.get("lambda_memory", 0.0)) * memory_loss
            energy = bundle.energy()
            orthogonality = bundle.orthogonality_loss()
            loss = loss + cfg.get("lambda_energy", 0.0) * energy
            loss = loss + cfg.get("lambda_orth", 0.0) * orthogonality
            (loss / accumulation).backward()
            if (group_index + 1) % accumulation == 0 or group_index + 1 == len(groups):
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            history.append({
                "epoch": epoch,
                "prefix": group[0].get("prefix_id"),
                "objective": objective,
                "loss": float(loss.detach()),
                "energy": float(energy.detach()),
                "orthogonality": float(orthogonality.detach()),
                "memory_loss": float(memory_loss.detach()),
            })
    return model, bundle, history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--data-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--projector-checkpoint")
    parser.add_argument("--counterfactual-data")
    parser.add_argument("--objective", choices=("tango", "global_return", "ce"))
    parser.add_argument("--no-projector-checkpoint", action="store_true")
    parser.add_argument("--target", choices=("k", "v", "both"))
    parser.add_argument("--rank", type=int)
    parser.add_argument("--alpha", type=float)
    parser.add_argument("--last-n-layers", type=int)
    parser.add_argument(
        "--return-mode", choices=("observed", "shuffle", "zero", "random", "sign_flip")
    )
    parser.add_argument(
        "--trajectory-horizon", default=None,
        help="Number of leading trajectory steps used by the objective, or 'full' (default).",
    )
    parser.add_argument(
        "--step-selection", choices=("all", "first", "final"), default=None,
        help="Use all, only the first, or only the final step inside the selected horizon.",
    )
    parser.add_argument("--remove-history", action="store_true")
    parser.add_argument("--lambda-action", type=float)
    parser.add_argument("--positive-action-only", action="store_true")
    parser.add_argument("--lambda-energy", type=float)
    parser.add_argument("--lambda-orth", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--max-pixels", type=int)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.max_trajectories is not None:
        cfg["max_trajectories"] = args.max_trajectories
    if args.data_path is not None:
        cfg["data_path"] = args.data_path
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.projector_checkpoint is not None:
        cfg["projector_checkpoint"] = args.projector_checkpoint
    if args.no_projector_checkpoint:
        cfg["projector_checkpoint"] = None
    if args.counterfactual_data is not None:
        cfg["counterfactual_data"] = args.counterfactual_data
    if args.positive_action_only:
        cfg["positive_action_only"] = True
    if args.remove_history:
        cfg["remove_history"] = True
    for key in (
        "return_mode", "lambda_action", "lambda_energy", "lambda_orth",
        "epochs", "max_pixels", "target", "rank", "alpha", "last_n_layers",
        "trajectory_horizon", "step_selection",
    ):
        value = getattr(args, key)
        if value is not None:
            cfg[key] = value
    if args.objective is not None:
        cfg["objective"] = args.objective
    random.seed(cfg["seed"]); torch.manual_seed(cfg["seed"])
    if cfg.get("counterfactual_data"):
        model, bundle, history = train_qwen_counterfactual(cfg)
    else:
        model, bundle, history = train_toy(cfg) if cfg["toy"] else train_qwen(cfg)
    output = Path(cfg["output_dir"]); output.mkdir(parents=True, exist_ok=True)
    torch.save(bundle.modules.state_dict(), output / "kv_projectors.pt")
    (output / "metrics.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(history[-1], indent=2))


if __name__ == "__main__":
    main()
