from scripts.estimate_kv_memory_credit_qwen import aggregate_matched_patch_credit


def _row(prefix, action, score, q):
    return {"prefix_id": prefix, "prefix": {}, "action": action, "candidate_actions": ["a", "b"],
            "score": score, "Q": q, "history_images": ["one.png"]}


def test_multiple_donor_credit_is_averaged_and_annotated():
    rows = [_row("p", "a", 0, 1), _row("p", "b", 0, 0)]
    full = [[_row("p", "a", 4, 1), _row("p", "b", 0, 0)]]
    donor1 = [[_row("p", "a", 0, 1), _row("p", "b", 0, 0)]]
    donor2 = [[_row("p", "a", -4, 1), _row("p", "b", 0, 0)]]
    annotated, summary = aggregate_matched_patch_credit(rows, full, {0: {1: donor1, 2: donor2}}, bootstrap_samples=20)
    assert annotated[0]["memory_advantage_source"] == "matched_latent_kv_patch_q"
    assert annotated[0]["memory_advantages"] == annotated[1]["memory_advantages"]
    assert summary["records"][0]["donor_count"] == 2
    assert summary["records"][0]["memory_advantage"] > 0.45
