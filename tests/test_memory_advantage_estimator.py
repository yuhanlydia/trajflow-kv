from scripts.estimate_memory_advantages_qwen import (
    annotate_memory_advantages,
    expected_continuation_return,
)


def _row(prefix, action, score, q):
    return {
        "prefix_id": prefix,
        "prefix": {},
        "action": action,
        "candidate_actions": ["a", "b"],
        "score": score,
        "Q": q,
        "history_images": ["cue.png"],
    }


def test_expected_continuation_return_uses_policy_scores():
    value = expected_continuation_return([_row("p", "a", 8, 1), _row("p", "b", -8, 0)])
    assert value > 0.999


def test_annotation_is_q_full_minus_q_drop_for_every_candidate_row():
    rows = [_row("p", "a", 0, 1), _row("p", "b", 0, 0)]
    full = [[_row("p", "a", 4, 1), _row("p", "b", 0, 0)]]
    dropped = {0: [[_row("p", "a", 0, 1), _row("p", "b", 0, 0)]]}
    annotated, summary = annotate_memory_advantages(rows, full, dropped)
    assert annotated[0]["memory_advantages"] == annotated[1]["memory_advantages"]
    assert annotated[0]["memory_advantages"][0] > 0.45
    assert summary["positive_blocks"] == 1
