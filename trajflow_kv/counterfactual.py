"""Small, deterministic same-prefix counterfactual GUI benchmark.

The benchmark is deliberately environment-only: it has no model, browser, or
network dependency.  A task exposes several legal-looking actions at a state;
the evaluator exhaustively rolls out each action's possible continuations from
the *same immutable prefix*.  This makes it useful for testing credit
assignment before spending compute on AndroidWorld.

Each emitted row has the following important fields::

    prefix, action, continuations, Q, V, advantage

    ``Q`` is the expected terminal return under the enumerated continuation
    policy (``max`` is available as an optimistic diagnostic), and ``V`` is the uniform prior
over the candidates at that prefix.  The row is intentionally one candidate
per line so it can be consumed by either a preference learner or a policy
gradient learner without re-enumerating the environment.
"""

from __future__ import annotations

import copy
import hashlib
import itertools
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


Action = str


class DelayedConsequenceTask:
    """Interface implemented by the deterministic toy GUI tasks."""

    task_family: str = "delayed"
    instruction: str = ""

    def initial_state(self, seed: int) -> dict[str, Any]:
        raise NotImplementedError

    def available_actions(self, state: dict[str, Any]) -> tuple[Action, ...]:
        raise NotImplementedError

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def step(self, state: dict[str, Any], action: Action) -> tuple[dict[str, Any], bool]:
        raise NotImplementedError

    def terminal_return(self, state: dict[str, Any]) -> float:
        return float(state.get("terminal_return", 0.0))

    def clone(self, state: dict[str, Any]) -> dict[str, Any]:
        return copy.deepcopy(state)


class HiddenCueTask(DelayedConsequenceTask):
    """Remember a cue that disappears before the decision is made."""

    task_family = "hidden_cue"
    instruction = "Remember the color shown earlier, then select that color when it reappears."
    colors = ("red", "blue", "green")

    def initial_state(self, seed: int) -> dict[str, Any]:
        cue = self.colors[seed % len(self.colors)]
        return {"phase": "cue", "cue": cue, "selected": None, "history": []}

    def available_actions(self, state: dict[str, Any]) -> tuple[Action, ...]:
        phase = state["phase"]
        if phase == "cue":
            return ("reveal", "skip")
        if phase == "choose":
            return tuple(f"choose_{color}" for color in self.colors) + ("cancel",)
        if phase == "submit":
            return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["phase"] == "cue":
            text = f"A transient banner says: COLOR={state['cue']}"
        elif state["phase"] == "choose":
            # The cue is intentionally absent from this observation.
            text = "The banner disappeared. Choose the remembered color."
        elif state["phase"] == "submit":
            text = f"Selection is {state['selected']}. Press submit."
        else:
            text = "Task finished."
        return {"screen": text, "candidates": list(self.available_actions(state))}

    def step(self, state: dict[str, Any], action: Action) -> tuple[dict[str, Any], bool]:
        state = self.clone(state)
        state["history"].append(action)
        if state["phase"] == "cue":
            if action == "reveal":
                state["phase"] = "choose"
                return state, False
            state.update(phase="terminal", terminal_return=0.0)
            return state, True
        if state["phase"] == "choose":
            if action.startswith("choose_") and action[7:] in self.colors:
                state["selected"] = action[7:]
                state["phase"] = "submit"
                return state, False
            state.update(phase="terminal", terminal_return=0.0)
            return state, True
        if state["phase"] == "submit":
            state.update(
                phase="terminal",
                terminal_return=float(action == "submit" and state["selected"] == state["cue"]),
            )
            return state, True
        return state, True


class OrderDependencyTask(DelayedConsequenceTask):
    """Every action is locally legal, but only one order reaches success."""

    task_family = "order_dependency"
    instruction = "Complete the workflow in the required order: A, then C, then B."
    required = ("A", "C", "B")

    def initial_state(self, seed: int) -> dict[str, Any]:
        # The seed is recorded for paired evaluation, while the workflow is
        # fixed so prefixes and Q values are reproducible across methods.
        return {"position": 0, "seed": int(seed), "history": []}

    def available_actions(self, state: dict[str, Any]) -> tuple[Action, ...]:
        if state.get("terminal_return") is not None:
            return ()
        return ("A", "B", "C", "back")

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        position = int(state["position"])
        return {
            "screen": f"Workflow step {position + 1}/3 is ready; all controls look enabled.",
            "required_progress": position,
            "candidates": list(self.available_actions(state)),
        }

    def step(self, state: dict[str, Any], action: Action) -> tuple[dict[str, Any], bool]:
        state = self.clone(state)
        state["history"].append(action)
        if action == self.required[state["position"]]:
            state["position"] += 1
            if state["position"] == len(self.required):
                state["terminal_return"] = 1.0
                return state, True
            return state, False
        state["terminal_return"] = 0.0
        return state, True


class IrreversibleForkTask(DelayedConsequenceTask):
    """A plausible draft action causes a delayed, irreversible failure."""

    task_family = "irreversible_fork"
    instruction = "Submit the final report; do not save a draft or cancel."

    def initial_state(self, seed: int) -> dict[str, Any]:
        return {"phase": "fork", "seed": int(seed), "history": []}

    def available_actions(self, state: dict[str, Any]) -> tuple[Action, ...]:
        phase = state["phase"]
        if phase == "fork":
            return ("save_draft", "submit_final", "cancel")
        if phase == "confirm":
            return ("confirm", "back")
        if phase == "draft":
            return ("submit_final", "discard")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        messages = {
            "fork": "Report ready. Save Draft and Submit Final look equally available.",
            "confirm": "Final submission confirmation is open.",
            "draft": "Draft saved. Continue to finish the report.",
            "terminal": "Task finished.",
        }
        return {"screen": messages[state["phase"]], "candidates": list(self.available_actions(state))}

    def step(self, state: dict[str, Any], action: Action) -> tuple[dict[str, Any], bool]:
        state = self.clone(state)
        state["history"].append(action)
        phase = state["phase"]
        if phase == "fork" and action == "submit_final":
            state["phase"] = "confirm"
            return state, False
        if phase == "confirm" and action == "confirm":
            state.update(phase="terminal", terminal_return=1.0)
            return state, True
        if phase == "fork" and action == "save_draft":
            # Saving a draft looks locally valid.  Its consequence is only
            # exposed at the later submit step.
            state["phase"] = "draft"
            return state, False
        # Saving a draft is irreversible for this task: even a later submit
        # reaches a terminal failure.  This is the delayed consequence.
        state.update(phase="terminal", terminal_return=0.0)
        return state, True


def default_delayed_tasks() -> tuple[DelayedConsequenceTask, ...]:
    """Return the benchmark's three required task families."""
    return (HiddenCueTask(), OrderDependencyTask(), IrreversibleForkTask())


def _state_key(state: dict[str, Any]) -> str:
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def enumerate_continuations(
    task: DelayedConsequenceTask,
    state: dict[str, Any],
    action: Action,
    *,
    horizon: int = 6,
    max_continuations: int = 512,
) -> list[dict[str, Any]]:
    """Enumerate future action sequences from an immutable prefix.

    The returned action list excludes the candidate ``action`` itself.  Each
    continuation records the terminal return reached by that sequence.  A
    finite horizon is explicit in the schema, avoiding accidental infinite
    search if a future toy task adds a loop.
    """
    if horizon < 0:
        raise ValueError("horizon must be non-negative")
    next_state, done = task.step(task.clone(state), action)

    def visit(current: dict[str, Any], remaining: int, suffix: list[Action]) -> Iterable[dict[str, Any]]:
        if current.get("terminal_return") is not None or remaining == 0:
            yield {
                "actions": list(suffix),
                "return": task.terminal_return(current),
                "terminal": current.get("terminal_return") is not None,
            }
            return
        for candidate in task.available_actions(current):
            child, child_done = task.step(task.clone(current), candidate)
            yield from visit(child, remaining - 1, suffix + [candidate])

    if done:
        return [{"actions": [], "return": task.terminal_return(next_state), "terminal": True}]
    return list(itertools.islice(visit(next_state, horizon, []), max_continuations))


def _aggregate(values: list[float], aggregation: str) -> float:
    if not values:
        return 0.0
    if aggregation == "max":
        return max(values)
    if aggregation == "mean":
        return sum(values) / len(values)
    raise ValueError("aggregation must be 'max' or 'mean'")


def _prefix_id(task_family: str, seed: int, state: dict[str, Any]) -> str:
    digest = hashlib.sha1(_state_key(state).encode()).hexdigest()[:10]
    return f"{task_family}-s{seed}-{digest}"


def build_counterfactual_examples(
    task: DelayedConsequenceTask,
    seed: int,
    *,
    horizon: int = 6,
    aggregation: str = "max",
    max_prefixes: int = 128,
) -> list[dict[str, Any]]:
    """Create same-prefix candidate rows for one task/seed."""
    if horizon < 1:
        raise ValueError("horizon must be positive")
    initial = task.initial_state(seed)
    queue: list[tuple[dict[str, Any], int]] = [(initial, 0)]
    visited: set[str] = set()
    prefixes: list[tuple[dict[str, Any], int]] = []
    while queue and len(prefixes) < max_prefixes:
        state, depth = queue.pop(0)
        key = _state_key(state)
        if key in visited:
            continue
        visited.add(key)
        if state.get("terminal_return") is not None or depth >= horizon:
            continue
        prefixes.append((state, depth))
        for action in task.available_actions(state):
            child, _ = task.step(task.clone(state), action)
            queue.append((child, depth + 1))

    rows: list[dict[str, Any]] = []
    for state, depth in prefixes:
        candidates = list(task.available_actions(state))
        candidate_data: list[tuple[Action, list[dict[str, Any]], float]] = []
        for action in candidates:
            continuations = enumerate_continuations(task, state, action, horizon=horizon - depth)
            q_value = _aggregate([float(item["return"]) for item in continuations], aggregation)
            candidate_data.append((action, continuations, q_value))
        prior = 1.0 / len(candidate_data)
        value = sum(item[2] * prior for item in candidate_data)
        prefix = {
            "state": task.clone(state),
            "history": list(state.get("history", [])),
            "observation": task.observe(state),
            "step_index": depth,
        }
        prefix_id = _prefix_id(task.task_family, seed, state)
        for action_index, (action, continuations, q_value) in enumerate(candidate_data):
            rows.append(
                {
                    "schema_version": "tango.counterfactual.v1",
                    "benchmark": "tango_delayed_gui_toy",
                    "task_id": f"{task.task_family}-s{seed}",
                    "task_family": task.task_family,
                    "seed": int(seed),
                    "instruction": task.instruction,
                    "prefix_id": prefix_id,
                    "prefix": prefix,
                    "candidate_actions": candidates,
                    "action_index": action_index,
                    "action": action,
                    "continuations": continuations,
                    "Q": float(q_value),
                    "V": float(value),
                    "advantage": float(q_value - value),
                    "horizon": int(horizon),
                    "q_aggregation": aggregation,
                }
            )
    return rows


def build_counterfactual_dataset(
    seeds: Iterable[int] = range(10),
    *,
    tasks: Iterable[DelayedConsequenceTask] | None = None,
    horizon: int = 6,
    aggregation: str = "max",
) -> list[dict[str, Any]]:
    """Build rows for all default task families and the requested seeds."""
    selected_tasks = tuple(tasks) if tasks is not None else default_delayed_tasks()
    selected_seeds = tuple(int(seed) for seed in seeds)
    return [
        row
        for task in selected_tasks
        for seed in selected_seeds
        for row in build_counterfactual_examples(
            task, int(seed), horizon=horizon, aggregation=aggregation
        )
    ]


def write_counterfactual_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    """Write rows and return the number written."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def load_counterfactual_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {"prefix", "action", "continuations", "Q", "V", "advantage"}
    with Path(path).open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            missing = required - row.keys()
            if missing:
                raise ValueError(f"line {line_no}: missing {sorted(missing)}")
            rows.append(row)
    return rows
