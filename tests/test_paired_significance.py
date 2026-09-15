from trajflow_kv.statistics import exact_mcnemar_p, summarize


def test_exact_mcnemar_is_two_sided():
    assert exact_mcnemar_p(0, 0) == 1.0
    assert exact_mcnemar_p(8, 0) == 0.0078125
    assert exact_mcnemar_p(4, 4) == 1.0


def test_clustered_summary_is_reproducible():
    rows = [
        {"task_id": "a", "baseline": 0, "candidate": 1},
        {"task_id": "a", "baseline": 0, "candidate": 1},
        {"task_id": "b", "baseline": 1, "candidate": 0},
        {"task_id": "b", "baseline": 1, "candidate": 1},
    ]
    result = summarize(rows, samples=200, seed=3)
    assert result["paired_delta"] == 0.25
    assert result["improved"] == 2
    assert result["regressed"] == 1
    assert result["tasks"] == 2
