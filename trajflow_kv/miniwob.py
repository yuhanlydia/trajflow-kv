from __future__ import annotations


def clickable_candidates(observation: dict) -> list[dict]:
    """Extract visible clickable BrowserGym elements with stable BIDs."""
    properties = observation["extra_element_properties"]
    rows = []
    for node in observation["axtree_object"]["nodes"]:
        bid = node.get("browsergym_id")
        prop = properties.get(str(bid)) if bid is not None else None
        if not prop or not prop.get("clickable") or not prop.get("bbox"):
            continue
        if float(prop.get("visibility", 0.0)) <= 0:
            continue
        name = str(node.get("name", {}).get("value", "")).strip() or "unlabeled"
        role = str(node.get("role", {}).get("value", "element"))
        state = {}
        for item in node.get("properties", []):
            if item.get("name") in {"checked", "selected", "expanded", "disabled"}:
                state[item["name"]] = item.get("value", {}).get("value")
        # A checked checkbox has already been acted on; removing it is the Web
        # analogue of the Android identical-action loop guard. Submit remains
        # available, and untouched checkbox options remain rankable.
        if role in {"checkbox", "radio"} and str(state.get("checked")).lower() == "true":
            continue
        x, y, width, height = prop["bbox"]
        rows.append({
            "bid": str(bid), "role": role, "name": name,
            "bbox": [float(x), float(y), float(width), float(height)],
            "state": state,
        })
    # BrowserGym can expose both a clickable container and clickable child at
    # the same box. Deduplicate descriptions without depending on DOM order.
    unique = {}
    for row in rows:
        key = (row["role"], row["name"], tuple(row["bbox"]), tuple(row["state"].items()))
        unique.setdefault(key, row)
    return sorted(unique.values(), key=lambda row: int(row["bid"]))


def exclude_acted_candidates(candidates: list[dict], acted_bids: set[str]) -> list[dict]:
    remaining = [row for row in candidates if row["bid"] not in acted_bids]
    return remaining or candidates


def build_miniwob_prompt(goal: str, history: list[str], candidates: list[dict]) -> str:
    choices = "\n".join(
        f"{index + 1}. {candidate_target(row)}"
        for index, row in enumerate(candidates)
    )
    return (
        "You control a web page. Select exactly one visible clickable element that advances "
        "the task. Do not invent an element.\n"
        f"Task: {goal}\nHistory: {history or ['None']}\nCandidates:\n{choices}\n"
        "Return exactly the selected candidate in the requested CLICK format."
    )


def candidate_target(candidate: dict) -> str:
    state = ""
    if candidate.get("state"):
        state = " " + " ".join(f"{key}={value}" for key, value in candidate["state"].items())
    geometry = ""
    if candidate.get("bbox"):
        x, y, width, height = candidate["bbox"]
        geometry = f" center=({x + width / 2:.1f},{y + height / 2:.1f})"
    return (
        f"CLICK BID {candidate['bid']}: [{candidate['role']}] "
        f"{candidate['name']}{state}{geometry}"
    )
