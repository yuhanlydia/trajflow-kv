from trajflow_kv.miniwob import (
    build_miniwob_prompt, candidate_target, clickable_candidates, exclude_acted_candidates,
)


def test_clickable_candidates_filters_and_deduplicates():
    observation = {
        "extra_element_properties": {
            "7": {"clickable": True, "visibility": 1, "bbox": [1, 2, 3, 4]},
            "8": {"clickable": False, "visibility": 1, "bbox": [5, 6, 7, 8]},
        },
        "axtree_object": {"nodes": [
            {"browsergym_id": "7", "role": {"value": "button"}, "name": {"value": "Save"}},
            {"browsergym_id": "8", "role": {"value": "button"}, "name": {"value": "Cancel"}},
        ]},
    }
    assert clickable_candidates(observation) == [{
        "bid": "7", "role": "button", "name": "Save", "bbox": [1.0, 2.0, 3.0, 4.0],
        "state": {},
    }]


def test_miniwob_prompt_and_target():
    candidate = {"bid": "7", "role": "button", "name": "Save", "state": {}}
    assert candidate_target(candidate) == "CLICK BID 7: [button] Save"
    assert "Task: Save the file" in build_miniwob_prompt("Save the file", [], [candidate])


def test_clickable_candidates_excludes_checked_checkbox():
    observation = {
        "extra_element_properties": {
            "4": {"clickable": True, "visibility": 1, "bbox": [1, 2, 3, 4]},
        },
        "axtree_object": {"nodes": [{
            "browsergym_id": "4", "role": {"value": "checkbox"},
            "name": {"value": "Done"},
            "properties": [{"name": "checked", "value": {"value": "true"}}],
        }]},
    }
    assert clickable_candidates(observation) == []


def test_exclude_acted_candidates_falls_back_if_exhausted():
    candidates = [{"bid": "1"}, {"bid": "2"}]
    assert exclude_acted_candidates(candidates, {"1"}) == [{"bid": "2"}]
    assert exclude_acted_candidates(candidates, {"1", "2"}) == candidates
