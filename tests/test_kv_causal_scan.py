from scripts.evaluate_kv_causal_scan import signed_score_effects


def _group(prefix, correct, wrong, label):
    return [
        {"prefix_id": prefix, "action": "correct", "score": correct, "optimal_actions": ["correct"], "memory_advantages": [label]},
        {"prefix_id": prefix, "action": "wrong", "score": wrong, "optimal_actions": ["correct"], "memory_advantages": [label]},
    ]


def test_signed_score_effects_distinguish_useful_and_harmful_memory():
    full = [_group("useful", -0.1, -1.0, 1.0), _group("stale", -0.8, -1.0, -1.0)]
    ablated = [_group("useful", -0.6, -1.0, 1.0), _group("stale", -0.2, -1.0, -1.0)]
    result = signed_score_effects(full, ablated, 0)
    assert result["memory_sign_accuracy"] == 1.0
    assert result["rows"][0]["correct_score_effect"] > 0
    assert result["rows"][1]["correct_score_effect"] < 0
