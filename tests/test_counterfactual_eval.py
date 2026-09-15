from scripts.evaluate_counterfactual_qwen import (
    summarize_noncritical_score_change,
    summarize_policy_scores,
)
import pytest


def _group():
    return [[
        {
            "prefix_id": "p0",
            "task_family": "distractor_credit",
            "action": "harmless",
            "score": 0.1,
            "Q": 0.5,
            "critical_step": False,
            "is_critical_action": False,
            "optimal_actions": ["harmless"],
        },
        {
            "prefix_id": "p0",
            "task_family": "distractor_credit",
            "action": "fork_bad",
            "score": 0.2,
            "Q": 0.0,
            "critical_step": False,
            "is_critical_action": False,
            "optimal_actions": ["harmless"],
        },
    ], [
        {
            "prefix_id": "p1",
            "task_family": "distractor_credit",
            "action": "choose_good",
            "score": 0.1,
            "Q": 1.0,
            "critical_step": True,
            "is_critical_action": True,
            "optimal_actions": ["choose_good"],
        },
        {
            "prefix_id": "p1",
            "task_family": "distractor_credit",
            "action": "choose_bad",
            "score": 0.2,
            "Q": 0.0,
            "critical_step": True,
            "is_critical_action": True,
            "optimal_actions": ["choose_good"],
        },
    ]]


def test_summary_reports_candidate_and_critical_metrics():
    summary = summarize_policy_scores(_group())
    assert summary["prefixes"] == 2
    assert summary["candidate_top1_accuracy"] == 0.0
    assert summary["critical_prefixes"] == 1
    assert summary["critical_fork_accuracy"] == 0.0
    assert summary["families"]["distractor_credit"]["prefixes"] == 2


def test_noncritical_change_excludes_critical_rows():
    target = _group()
    baseline = _group()
    target[0][0]["score"] = 0.3
    target[1][0]["score"] = 0.9
    change = summarize_noncritical_score_change(target, baseline)
    assert change["candidates"] == 2
    assert change["mean_delta"] == pytest.approx(0.1)
    assert change["mean_absolute_delta"] == pytest.approx(0.1)
