import xml.etree.ElementTree as ET

from trajflow_kv.mind2web import build_prompt, describe_candidate, format_target


def test_candidate_description_prefers_accessible_text():
    root = ET.fromstring('<html><button backend_node_id="7" aria-label="Save file">ignored</button></html>')
    assert describe_candidate(root, {"backend_node_id": "7", "tag": "button"}) == (
        "[button] Save file ignored"
    )


def test_mind2web_target_and_prompt():
    target = format_target("[input] Search", {"op": "TYPE", "value": "NFL"})
    assert target == "Element: [input] Search\nAction: TYPE\nValue: NFL"
    prompt = build_prompt("Find scores", ["clicked NFL"], ["[a] Scores", "[div] Ads"])
    assert "Find scores" in prompt and "clicked NFL" in prompt and "2. [div] Ads" in prompt
