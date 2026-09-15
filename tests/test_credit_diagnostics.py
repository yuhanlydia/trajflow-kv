import json

from trajflow_kv.credit_diagnostics import (
    add_history_oracle,
    annotate_credit_ground_truth,
    balanced_memory_prefix_rows,
    critical_prefix_rows,
    group_counterfactual_rows,
    summarize_credit_rows,
)
from trajflow_kv.visual_delayed import DistractorCreditTask, HiddenMemoryTask, build_visual_counterfactual_dataset


def _rows():
    return build_visual_counterfactual_dataset(
        [0], output_dir="/tmp/tango-credit-diagnostic-images",
        tasks=[DistractorCreditTask(), HiddenMemoryTask()], horizon=8,
    )


def test_critical_filter_keeps_complete_candidate_groups():
    rows = _rows()
    all_groups = group_counterfactual_rows(rows)
    critical = critical_prefix_rows(rows)
    critical_groups = group_counterfactual_rows(critical)
    assert critical_groups
    assert all(any(row["critical_step"] for row in group) for group in critical_groups)
    assert all(len(group) == len(group[0]["candidate_actions"]) for group in critical_groups)
    assert len(critical) < len(rows)
    assert len(critical) == sum(len(group) for group in all_groups if any(row["critical_step"] for row in group))


def test_ground_truth_marks_critical_and_noncritical_rows():
    annotated = annotate_credit_ground_truth(_rows())
    assert all("ground_truth" in row for row in annotated)
    assert {row["ground_truth"]["credit_role"] for row in annotated} == {"critical", "noncritical"}
    hidden_choose = next(row for row in annotated if row["task_family"] == "hidden_memory" and row["prefix"]["state"]["phase"] == "choose")
    assert hidden_choose["ground_truth"]["memory_source"] == {
        "kind": "cue", "source_step": 0, "source_field": "cue"
    }


def test_history_oracle_reveals_only_the_delayed_choice_evidence():
    rows = _rows()
    transformed = add_history_oracle(rows)
    hidden_choose = next(row for row in transformed if row["task_family"] == "hidden_memory" and row["prefix"]["state"]["phase"] == "choose")
    assert "History oracle:" in hidden_choose["prompt"]
    assert "RED" in hidden_choose["prompt"]
    assert "History oracle:" in hidden_choose["prefix"]["observation"]["screen"]
    hidden_cue = next(row for row in transformed if row["task_family"] == "hidden_memory" and row["prefix"]["state"]["phase"] == "cue")
    assert "history_oracle" not in hidden_cue
    # Original rows are not mutated, so the non-oracle pilot remains a valid
    # comparison condition.
    original_choose = next(row for row in rows if row["task_family"] == "hidden_memory" and row["prefix"]["state"]["phase"] == "choose")
    assert "History oracle:" not in original_choose["prefix"]["observation"]["screen"]


def test_summary_reports_group_counts_and_memory_sources():
    summary = summarize_credit_rows(_rows())
    assert summary["complete_candidate_groups"] is True
    assert summary["critical_prefixes"] > 0
    assert summary["noncritical_prefixes"] > 0
    assert summary["critical_rows"] + summary["noncritical_rows"] == summary["rows"]
    assert summary["memory_source_prefixes"]["cue"] == 1


def test_balanced_memory_rows_pair_critical_and_zero_advantage_prefixes():
    rows = balanced_memory_prefix_rows(_rows())
    groups = group_counterfactual_rows(rows)
    critical = [group for group in groups if group[0]["critical_prefix"]]
    negatives = [group for group in groups if group[0]["noncritical_prefix"]]
    assert len(critical) == len(negatives)
    assert len(critical) >= 2
    assert all(group[0]["history_images"] for group in negatives)
    assert all(not any(group[0]["memory_advantages"]) for group in negatives)
