"""Statistical tests for paired GUI policy evaluations."""
from __future__ import annotations

import math
import random
from collections import defaultdict


def exact_mcnemar_p(improved: int, regressed: int) -> float:
    discordant = improved + regressed
    if discordant == 0:
        return 1.0
    tail = min(improved, regressed)
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / 2**discordant
    return min(1.0, 2.0 * probability)


def clustered_interval(rows: list[dict], samples: int, seed: int) -> tuple[float, float]:
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[str(row["task_id"])].append(float(row["candidate"] > 0) - float(row["baseline"] > 0))
    keys = sorted(groups)
    if not keys:
        raise ValueError("no paired rows")
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        selected = [rng.choice(keys) for _ in keys]
        values = [value for key in selected for value in groups[key]]
        estimates.append(sum(values) / len(values))
    estimates.sort()
    lower = estimates[int(0.025 * (samples - 1))]
    upper = estimates[int(0.975 * (samples - 1))]
    return lower, upper


def summarize(rows: list[dict], samples: int = 10_000, seed: int = 7) -> dict:
    if not rows:
        raise ValueError("no paired rows")
    improved = sum(row["baseline"] <= 0 < row["candidate"] for row in rows)
    regressed = sum(row["candidate"] <= 0 < row["baseline"] for row in rows)
    baseline_rate = sum(row["baseline"] > 0 for row in rows) / len(rows)
    candidate_rate = sum(row["candidate"] > 0 for row in rows) / len(rows)
    low, high = clustered_interval(rows, samples, seed)
    return {
        "pairs": len(rows), "tasks": len({row["task_id"] for row in rows}),
        "baseline_success_rate": baseline_rate,
        "candidate_success_rate": candidate_rate,
        "paired_delta": candidate_rate - baseline_rate,
        "task_clustered_bootstrap_95_ci": [low, high],
        "improved": improved, "regressed": regressed,
        "ties": len(rows) - improved - regressed,
        "exact_mcnemar_p": exact_mcnemar_p(improved, regressed),
        "bootstrap_samples": samples, "seed": seed,
    }
