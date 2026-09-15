"""Small screenshot-backed delayed-consequence GUI benchmark.

This module turns the deterministic counterfactual tasks into actual visual
GUI observations without requiring a browser or an Android emulator.  The
state machine is executable through :class:`DelayedConsequenceTask.step`,
while :func:`build_visual_counterfactual_dataset` writes the existing
``tango.counterfactual.v1`` JSONL rows plus one deterministic PNG per prefix.

The benchmark is intentionally controlled: ``distractor_credit`` has two
harmless action stages around one hidden critical fork, and
``hidden_memory`` exposes a cue which disappears before the decision.  Each
row keeps an immutable same-prefix group and annotates the ground-truth
critical step so credit localization can be evaluated independently of
policy success.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .counterfactual import DelayedConsequenceTask, build_counterfactual_examples


_PALETTE = {
    "red": (214, 61, 61),
    "blue": (54, 103, 198),
    "green": (48, 157, 92),
    "amber": (202, 138, 4),
    "orange": (224, 132, 42),
}


def _font(size: int = 22) -> ImageFont.ImageFont:
    # Prefer a common container font so screenshots are legible to a VLM, but
    # retain the bitmap fallback for minimal CI images.
    for font_path in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
    ):
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size=size)
    return ImageFont.load_default()


def _draw_button(draw: ImageDraw.ImageDraw, xy: tuple[int, int, int, int], label: str, *, fill=(238, 242, 248)) -> None:
    draw.rounded_rectangle(xy, radius=10, fill=fill, outline=(60, 70, 85), width=2)
    left, top, right, bottom = xy
    bbox = draw.textbbox((0, 0), label, font=_font())
    draw.text(
        ((left + right - bbox[2]) / 2, (top + bottom - bbox[3]) / 2 - 2),
        label,
        fill=(25, 31, 40),
        font=_font(),
    )


class VisualDelayedTask(DelayedConsequenceTask):
    """Extension point for screenshot rendering and critical-step labels."""

    benchmark = "tango_visual_delayed_gui"

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return ()

    def memory_advantages(self, state: dict[str, Any], history_count: int) -> tuple[float, ...]:
        """Ground-truth memory-block advantage for controlled diagnostics."""
        return (0.0,) * history_count

    def memory_roles(self, state: dict[str, Any], history_count: int) -> tuple[str, ...]:
        """Semantic roles; these are not asserted to equal model-causal signs."""
        return ("irrelevant",) * history_count

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        raise NotImplementedError


class DistractorCreditTask(VisualDelayedTask):
    """Two irrelevant action stages surround a hidden, consequential fork.

    A transient target cue is shown on the first screen and disappears before
    the ``choose_a``/``choose_b`` fork.  ``click_x``/``skip_x`` and
    ``click_y``/``skip_y`` are deliberately return-equivalent distractors.
    """

    task_family = "distractor_credit"
    instruction = "Remember the target option, pass the harmless steps, choose it, then submit."
    options = ("A", "B")

    def initial_state(self, seed: int) -> dict[str, Any]:
        target = self.options[int(seed) % len(self.options)]
        return {"phase": "x", "target": target, "choice": None, "history": [], "seed": int(seed)}

    def available_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        phase = state["phase"]
        if phase == "x":
            return ("click_x", "skip_x")
        if phase == "fork":
            return ("choose_A", "choose_B")
        if phase == "y":
            return ("click_y", "skip_y")
        if phase == "submit":
            return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        phase = state["phase"]
        if phase == "x":
            screen = f"Temporary banner: TARGET OPTION = {state['target']}. Harmless step X."
        elif phase == "fork":
            screen = "Target banner disappeared. Choose the correct option, then continue."
        elif phase == "y":
            screen = "Harmless step Y. The selected option is not shown here."
        elif phase == "submit":
            screen = f"Ready to submit. Selected option: {state['choice']}."
        else:
            screen = "Task finished."
        return {"screen": screen, "candidates": list(self.available_actions(state))}

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return tuple(self.available_actions(state)) if state["phase"] == "fork" else ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "fork":
            return (f"choose_{state['target']}",)
        if state["phase"] == "submit" and state.get("choice") == state.get("target"):
            return ("submit",)
        return tuple(self.available_actions(state)) if state["phase"] in {"x", "y"} else ()

    def memory_advantages(self, state: dict[str, Any], history_count: int) -> tuple[float, ...]:
        values = [0.0] * history_count
        if state["phase"] == "fork" and values:
            values[0] = 1.0
        return tuple(values)

    def step(self, state: dict[str, Any], action: str) -> tuple[dict[str, Any], bool]:
        state = self.clone(state)
        state["history"].append(action)
        phase = state["phase"]
        if phase == "x" and action in self.available_actions(state):
            state["phase"] = "fork"
            return state, False
        if phase == "fork" and action in self.available_actions(state):
            state["choice"] = action.rsplit("_", 1)[-1]
            state["phase"] = "y"
            return state, False
        if phase == "y" and action in self.available_actions(state):
            state["phase"] = "submit"
            return state, False
        if phase == "submit":
            state.update(
                phase="terminal",
                terminal_return=float(action == "submit" and state.get("choice") == state.get("target")),
            )
            return state, True
        state.update(phase="terminal", terminal_return=0.0)
        return state, True

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        image = Image.new("RGB", (960, 600), (247, 249, 252))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 68), fill=(35, 48, 68))
        draw.text((28, 22), "TANGO delayed workflow", fill="white", font=_font())
        draw.text((28, 100), self.instruction, fill=(30, 38, 50), font=_font())
        phase = state["phase"]
        if phase == "x":
            draw.rounded_rectangle((40, 150, 920, 220), radius=12, fill=(255, 239, 192), outline=(206, 155, 39), width=2)
            draw.text((65, 177), f"Remember this: target option is {state['target']}", fill=(105, 72, 10), font=_font())
            labels = ("Click harmless X", "Skip harmless X")
        elif phase == "fork":
            draw.text((50, 168), "Choose the option required by the earlier target.", fill=(30, 38, 50), font=_font())
            labels = ("Choose A", "Choose B")
        elif phase == "y":
            draw.text((50, 168), "This step has no effect on the final result.", fill=(30, 38, 50), font=_font())
            labels = ("Click harmless Y", "Skip harmless Y")
        elif phase == "submit":
            draw.text((50, 168), f"Selected option: {state['choice']}", fill=(30, 38, 50), font=_font())
            labels = ("Submit", "Cancel")
        else:
            draw.text((50, 168), "Task finished.", fill=(30, 38, 50), font=_font())
            labels = ()
        for index, label in enumerate(labels):
            left = 60 + index * 430
            _draw_button(draw, (left, 275, left + 350, 355), label)
        draw.text((50, 470), f"Visible history length: {len(state.get('history', []))}", fill=(100, 108, 120), font=_font())
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")


class MultiCueBindingTask(VisualDelayedTask):
    """Bind two cues exposed on separate pages after an intervening distractor."""

    task_family = "multi_cue_binding"
    instruction = "Remember the color and symbol shown on separate pages, then select their exact pair."
    colors = ("red", "blue", "green", "amber")
    symbols = ("circle", "triangle", "square", "star")

    def initial_state(self, seed: int) -> dict[str, Any]:
        color = self.colors[int(seed) % len(self.colors)]
        symbol = self.symbols[(int(seed) * 3 + 1) % len(self.symbols)]
        wrong_color = self.colors[(self.colors.index(color) + 1) % len(self.colors)]
        wrong_symbol = self.symbols[(self.symbols.index(symbol) + 1) % len(self.symbols)]
        options = [f"choose_{color}_{symbol}", f"choose_{wrong_color}_{symbol}",
                   f"choose_{color}_{wrong_symbol}", f"choose_{wrong_color}_{wrong_symbol}"]
        shift = int(seed) % len(options); options = options[shift:] + options[:shift]
        return {"phase": "color_cue", "color": color, "symbol": symbol,
                "options": options, "choice": None, "history": [], "seed": int(seed)}

    def available_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] in {"color_cue", "symbol_cue"}: return ("continue",)
        if state["phase"] == "distractor": return ("acknowledge",)
        if state["phase"] == "choose": return tuple(state["options"])
        if state["phase"] == "submit": return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        text = {"color_cue": f"Color cue: {state['color']}",
                "symbol_cue": f"Symbol cue: {state['symbol']}",
                "distractor": "Neutral verification page; neither cue is shown.",
                "choose": "Select the exact color-symbol pair from memory.",
                "submit": f"Selected pair: {state['choice']}"}.get(state["phase"], "Task finished.")
        return {"screen": text, "candidates": list(self.available_actions(state))}

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return tuple(self.available_actions(state)) if state["phase"] == "choose" else ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "choose": return (f"choose_{state['color']}_{state['symbol']}",)
        if state["phase"] == "submit" and state.get("choice") == f"{state['color']}_{state['symbol']}": return ("submit",)
        return tuple(self.available_actions(state)) if state["phase"] in {"color_cue", "symbol_cue", "distractor"} else ()

    def memory_advantages(self, state: dict[str, Any], history_count: int) -> tuple[float, ...]:
        values = [0.0] * history_count
        if state["phase"] in {"symbol_cue", "distractor", "choose"} and values: values[0] = 1.0
        if state["phase"] in {"distractor", "choose"} and len(values) > 1: values[1] = 1.0
        return tuple(values)

    def step(self, state: dict[str, Any], action: str) -> tuple[dict[str, Any], bool]:
        state = self.clone(state); state["history"].append(action); phase = state["phase"]
        if phase == "color_cue" and action == "continue": state["phase"] = "symbol_cue"; return state, False
        if phase == "symbol_cue" and action == "continue": state["phase"] = "distractor"; return state, False
        if phase == "distractor" and action == "acknowledge": state["phase"] = "choose"; return state, False
        if phase == "choose" and action in self.available_actions(state):
            state["choice"] = action[len("choose_"):]; state["phase"] = "submit"; return state, False
        if phase == "submit":
            correct = state.get("choice") == f"{state['color']}_{state['symbol']}"
            state.update(phase="terminal", terminal_return=float(action == "submit" and correct)); return state, True
        state.update(phase="terminal", terminal_return=0.0); return state, True

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        image = Image.new("RGB", (960, 600), (247, 249, 252)); draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 68), fill=(35, 48, 68)); draw.text((28, 22), "TANGO multi-cue binding", fill="white", font=_font())
        phase = state["phase"]
        if phase == "color_cue":
            draw.rounded_rectangle((50, 130, 910, 235), radius=12, fill=_PALETTE[state["color"]])
            draw.text((75, 170), f"MEMORIZE COLOR: {state['color'].upper()}", fill="white", font=_font()); labels = ("Continue",)
        elif phase == "symbol_cue":
            draw.text((50, 170), f"MEMORIZE SYMBOL: {state['symbol'].upper()}", fill=(25, 31, 40), font=_font(28)); labels = ("Continue",)
        elif phase == "distractor":
            draw.text((50, 170), "Verification complete. The two cues are now hidden.", fill=(25, 31, 40), font=_font()); labels = ("Acknowledge",)
        elif phase == "choose":
            draw.text((50, 115), "Choose the remembered COLOR / SYMBOL pair", fill=(25, 31, 40), font=_font())
            labels = tuple(action[len("choose_"):].replace("_", " / ").upper() for action in state["options"])
        elif phase == "submit":
            draw.text((50, 170), f"Selected: {state['choice']}", fill=(25, 31, 40), font=_font()); labels = ("Submit", "Cancel")
        else: labels = ()
        for index, label in enumerate(labels):
            cols = 2 if len(labels) > 2 else len(labels); row, col = divmod(index, max(cols, 1)); left = 55 + col * 445; top = 270 + row * 105
            _draw_button(draw, (left, top, left + 405, top + 75), label)
        Path(path).parent.mkdir(parents=True, exist_ok=True); image.save(path, format="PNG")


class InterferenceUpdateTask(VisualDelayedTask):
    """A newer cue supersedes an old cue; stale memory should be suppressed."""

    task_family = "interference_update"
    instruction = "Remember the UPDATED color. Ignore the earlier obsolete color and later choose the update."
    colors = ("red", "blue", "green", "amber")

    def initial_state(self, seed: int) -> dict[str, Any]:
        old = self.colors[int(seed) % len(self.colors)]; new = self.colors[(int(seed) * 3 + 1) % len(self.colors)]
        if new == old: new = self.colors[(self.colors.index(old) + 1) % len(self.colors)]
        options = list(self.colors); shift = int(seed) % len(options); options = options[shift:] + options[:shift]
        return {"phase": "old_cue", "old": old, "new": new, "options": options,
                "choice": None, "history": [], "seed": int(seed)}

    def available_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] in {"old_cue", "new_cue"}: return ("continue",)
        if state["phase"] == "distractor": return ("acknowledge",)
        if state["phase"] == "choose": return tuple(f"choose_{color}" for color in state["options"])
        if state["phase"] == "submit": return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        text = {"old_cue": f"Initial color: {state['old']}", "new_cue": f"UPDATED color replaces it: {state['new']}",
                "distractor": "Neutral page. No color is visible.", "choose": "Choose the updated color, not the obsolete color.",
                "submit": f"Selected: {state['choice']}"}.get(state["phase"], "Task finished.")
        return {"screen": text, "candidates": list(self.available_actions(state))}

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return tuple(self.available_actions(state)) if state["phase"] == "choose" else ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "choose": return (f"choose_{state['new']}",)
        if state["phase"] == "submit" and state.get("choice") == state.get("new"): return ("submit",)
        return tuple(self.available_actions(state)) if state["phase"] in {"old_cue", "new_cue", "distractor"} else ()

    def memory_advantages(self, state: dict[str, Any], history_count: int) -> tuple[float, ...]:
        values = [0.0] * history_count
        if state["phase"] in {"distractor", "choose"} and len(values) > 1: values[0] = -1.0; values[1] = 1.0
        return tuple(values)

    def memory_roles(self, state: dict[str, Any], history_count: int) -> tuple[str, ...]:
        roles = ["irrelevant"] * history_count
        if state["phase"] in {"distractor", "choose"} and len(roles) > 1:
            roles[0], roles[1] = "stale", "useful"
        return tuple(roles)

    def step(self, state: dict[str, Any], action: str) -> tuple[dict[str, Any], bool]:
        state = self.clone(state); state["history"].append(action); phase = state["phase"]
        if phase == "old_cue" and action == "continue": state["phase"] = "new_cue"; return state, False
        if phase == "new_cue" and action == "continue": state["phase"] = "distractor"; return state, False
        if phase == "distractor" and action == "acknowledge": state["phase"] = "choose"; return state, False
        if phase == "choose" and action in self.available_actions(state): state["choice"] = action[len("choose_"):]; state["phase"] = "submit"; return state, False
        if phase == "submit": state.update(phase="terminal", terminal_return=float(action == "submit" and state.get("choice") == state.get("new"))); return state, True
        state.update(phase="terminal", terminal_return=0.0); return state, True

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        image = Image.new("RGB", (960, 600), (249, 249, 252)); draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 68), fill=(35, 48, 68)); draw.text((28, 22), "TANGO interference update", fill="white", font=_font())
        phase = state["phase"]
        if phase in {"old_cue", "new_cue"}:
            color = state["old"] if phase == "old_cue" else state["new"]
            title = "INITIAL (WILL BE REPLACED)" if phase == "old_cue" else "UPDATED — USE THIS"
            draw.rounded_rectangle((50, 130, 910, 235), radius=12, fill=_PALETTE[color]); draw.text((75, 170), f"{title}: {color.upper()}", fill="white", font=_font()); labels = ("Continue",)
        elif phase == "distractor": draw.text((50, 170), "Status page. Both color banners are hidden.", fill=(25, 31, 40), font=_font()); labels = ("Acknowledge",)
        elif phase == "choose": draw.text((50, 120), "Which color was the UPDATE?", fill=(25, 31, 40), font=_font()); labels = tuple(color.title() for color in state["options"])
        elif phase == "submit": draw.text((50, 170), f"Selected: {state['choice']}", fill=(25, 31, 40), font=_font()); labels = ("Submit", "Cancel")
        else: labels = ()
        for index, label in enumerate(labels):
            cols = 2 if len(labels) > 2 else len(labels); row, col = divmod(index, max(cols, 1)); left = 55 + col * 445; top = 270 + row * 105
            fill = _PALETTE.get(label.lower(), (238, 242, 248)) if phase == "choose" else (238, 242, 248); _draw_button(draw, (left, top, left + 405, top + 75), label, fill=fill)
        Path(path).parent.mkdir(parents=True, exist_ok=True); image.save(path, format="PNG")


class NonceVisualBindingTask(VisualDelayedTask):
    """Bind two render-only nonce codes and select a neutral positional action."""

    task_family = "nonce_visual_binding"
    instruction = "Remember both codes shown on separate pages. At the choice page, select the slot containing their exact ordered pair."
    left_codes = ("K7Q", "R4M", "B9X", "T2F", "P6N", "D3W", "H8L", "V5C")
    right_codes = ("Z2A", "G8P", "C5R", "N1Y", "W7D", "F4K", "M9B", "Q3V")

    def initial_state(self, seed: int) -> dict[str, Any]:
        seed = int(seed)
        left = self.left_codes[seed % len(self.left_codes)]
        right = self.right_codes[(seed * 5 + 3) % len(self.right_codes)]
        pairs = [(left, right)]
        for offset in range(1, 6):
            pairs.append((self.left_codes[(seed + offset) % 8], self.right_codes[(seed * 5 + 3 + 3 * offset) % 8]))
        shift = (seed * 3) % len(pairs)
        pairs = pairs[shift:] + pairs[:shift]
        return {"phase": "left", "left": left, "right": right, "pairs": pairs,
                "correct_slot": pairs.index((left, right)), "choice": None, "history": [], "seed": seed}

    def available_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] in {"left", "right"}: return ("continue",)
        if state["phase"] == "distractor": return ("acknowledge",)
        if state["phase"] == "choose": return tuple(f"select_slot_{i + 1}" for i in range(len(state["pairs"])))
        if state["phase"] == "submit": return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        text = {"left": "First code is visible only in the screenshot.", "right": "Second code is visible only in the screenshot.",
                "distractor": "The codes are hidden.", "choose": "Select the matching ordered pair by slot.",
                "submit": f"Selected slot: {state['choice']}"}.get(state["phase"], "Task finished.")
        return {"screen": text, "candidates": list(self.available_actions(state))}

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return tuple(self.available_actions(state)) if state["phase"] == "choose" else ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "choose": return (f"select_slot_{state['correct_slot'] + 1}",)
        if state["phase"] == "submit" and state.get("choice") == state.get("correct_slot"): return ("submit",)
        return tuple(self.available_actions(state)) if state["phase"] in {"left", "right", "distractor"} else ()

    def memory_advantages(self, state: dict[str, Any], history_count: int) -> tuple[float, ...]:
        values = [0.0] * history_count
        if state["phase"] in {"right", "distractor", "choose"} and values: values[0] = 1.0
        if state["phase"] in {"distractor", "choose"} and len(values) > 1: values[1] = 1.0
        return tuple(values)

    def memory_roles(self, state: dict[str, Any], history_count: int) -> tuple[str, ...]:
        roles = ["irrelevant"] * history_count
        if state["phase"] in {"right", "distractor", "choose"} and roles: roles[0] = "useful"
        if state["phase"] in {"distractor", "choose"} and len(roles) > 1: roles[1] = "useful"
        return tuple(roles)

    def step(self, state: dict[str, Any], action: str) -> tuple[dict[str, Any], bool]:
        state = self.clone(state); state["history"].append(action); phase = state["phase"]
        if phase == "left" and action == "continue": state["phase"] = "right"; return state, False
        if phase == "right" and action == "continue": state["phase"] = "distractor"; return state, False
        if phase == "distractor" and action == "acknowledge": state["phase"] = "choose"; return state, False
        if phase == "choose" and action in self.available_actions(state): state["choice"] = int(action.rsplit("_", 1)[1]) - 1; state["phase"] = "submit"; return state, False
        if phase == "submit": state.update(phase="terminal", terminal_return=float(action == "submit" and state.get("choice") == state.get("correct_slot"))); return state, True
        state.update(phase="terminal", terminal_return=0.0); return state, True

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        image = Image.new("RGB", (960, 600), (245, 247, 251)); draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 68), fill=(27, 39, 60)); draw.text((28, 22), "Secure pair verification", fill="white", font=_font())
        phase = state["phase"]
        if phase in {"left", "right"}:
            code = state[phase]; ordinal = "FIRST" if phase == "left" else "SECOND"
            draw.text((250, 150), f"MEMORIZE {ordinal} CODE", fill=(40, 48, 62), font=_font(24))
            draw.rounded_rectangle((265, 215, 695, 335), radius=15, fill=(225, 231, 243), outline=(60, 80, 115), width=3)
            draw.text((410, 255), code, fill=(15, 25, 45), font=_font(34)); labels = ("Continue",)
        elif phase == "distractor": draw.text((245, 210), "Session synchronization complete", fill=(40, 48, 62), font=_font()); labels = ("Acknowledge",)
        elif phase == "choose":
            draw.text((45, 92), "Select the slot containing FIRST CODE / SECOND CODE", fill=(40, 48, 62), font=_font())
            labels = tuple(f"SLOT {i + 1}:  {a} / {b}" for i, (a, b) in enumerate(state["pairs"]))
        elif phase == "submit": draw.text((300, 205), f"Selected slot {state['choice'] + 1}", fill=(40, 48, 62), font=_font()); labels = ("Submit", "Cancel")
        else: labels = ()
        for index, label in enumerate(labels):
            cols = 2 if len(labels) > 2 else max(len(labels), 1); row, col = divmod(index, cols); left = 35 + col * 465; top = (155 + row * 115) if phase == "choose" else 430
            _draw_button(draw, (left, top, left + 425, top + 82), label)
        Path(path).parent.mkdir(parents=True, exist_ok=True); image.save(path, format="PNG")


class InterferenceChainTask(VisualDelayedTask):
    """Multiple nonce updates interleaved with visually matched reference records."""

    task_family = "interference_chain"
    instruction = "Track the most recent UPDATE record. Ignore INITIAL and REFERENCE records, then select the slot containing the latest update code."
    codes = NonceVisualBindingTask.left_codes

    def __init__(self, template: str = "A"):
        if template not in {"A", "B", "C", "D"}: raise ValueError("template must be A, B, C, or D")
        self.template = template

    def initial_state(self, seed: int) -> dict[str, Any]:
        seed = int(seed)
        rng = random.Random(seed * 104729 + 17)
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        options = []
        while len(options) < 8:
            code = "".join(rng.choice(alphabet) for _ in range(3))
            if code not in options: options.append(code)
        ordered = [options[index] for index in (0, 5, 2, 7, 4)]
        entries = [
            {"role": "INITIAL", "code": ordered[0]}, {"role": "REFERENCE", "code": ordered[1]},
            {"role": "UPDATE", "code": ordered[2]}, {"role": "REFERENCE", "code": ordered[3]},
            {"role": "UPDATE", "code": ordered[4]},
        ]
        rng.shuffle(options)
        return {"phase": "record", "record_index": 0, "entries": entries, "options": options,
                "target": ordered[4], "choice": None, "history": [], "seed": seed, "template": self.template}

    def available_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "record": return ("continue",)
        if state["phase"] == "choose": return tuple(f"select_slot_{i + 1}" for i in range(len(state["options"])))
        if state["phase"] == "submit": return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        if state["phase"] == "record": text = "A record is visible in the screenshot. Track its role and code."
        elif state["phase"] == "choose": text = "Select the latest UPDATE code by slot."
        elif state["phase"] == "submit": text = f"Selected slot: {state['choice']}"
        else: text = "Task finished."
        return {"screen": text, "candidates": list(self.available_actions(state))}

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return tuple(self.available_actions(state)) if state["phase"] == "choose" else ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "choose": return (f"select_slot_{state['options'].index(state['target']) + 1}",)
        if state["phase"] == "submit" and state.get("choice") == state["options"].index(state["target"]): return ("submit",)
        return ("continue",) if state["phase"] == "record" else ()

    def memory_advantages(self, state: dict[str, Any], history_count: int) -> tuple[float, ...]:
        # Designer prior only; empirical causal signs come from matched KV patches.
        roles = self.memory_roles(state, history_count)
        return tuple(1.0 if role == "useful" else -1.0 if role == "stale" else 0.0 for role in roles)

    def memory_roles(self, state: dict[str, Any], history_count: int) -> tuple[str, ...]:
        if state["phase"] != "choose": return ("irrelevant",) * history_count
        canonical = ("stale", "irrelevant", "stale", "irrelevant", "useful")
        return canonical[:history_count]

    def step(self, state: dict[str, Any], action: str) -> tuple[dict[str, Any], bool]:
        state = self.clone(state); state["history"].append(action); phase = state["phase"]
        if phase == "record" and action == "continue":
            if state["record_index"] + 1 < len(state["entries"]): state["record_index"] += 1
            else: state["phase"] = "choose"
            return state, False
        if phase == "choose" and action in self.available_actions(state): state["choice"] = int(action.rsplit("_", 1)[1]) - 1; state["phase"] = "submit"; return state, False
        if phase == "submit": state.update(phase="terminal", terminal_return=float(action == "submit" and state.get("choice") == state["options"].index(state["target"]))); return state, True
        state.update(phase="terminal", terminal_return=0.0); return state, True

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        template = state.get("template", "A")
        backgrounds = {"A": (245, 247, 251), "B": (250, 246, 240), "C": (238, 246, 244), "D": (246, 241, 248)}
        headers = {"A": (27, 39, 60), "B": (70, 47, 34), "C": (24, 65, 60), "D": (64, 38, 70)}
        image = Image.new("RGB", (960, 600), backgrounds[template]); draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 68), fill=headers[template]); draw.text((28, 22), "Record audit", fill="white", font=_font())
        phase = state["phase"]
        if phase == "record":
            entry = state["entries"][state["record_index"]]
            card = (210, 135, 750, 340) if template in {"A", "B"} else (120, 125, 660, 330)
            draw.rounded_rectangle(card, radius=14, fill=(230, 234, 242), outline=(70, 82, 105), width=3)
            draw.text((card[0] + 40, card[1] + 45), f"RECORD TYPE: {entry['role']}", fill=(40, 48, 62), font=_font(25 if template in {"A", "C"} else 22))
            draw.text((card[0] + 190, card[1] + 120), entry["code"], fill=(15, 25, 45), font=_font(36 if template in {"A", "D"} else 31)); labels = ("Continue",)
        elif phase == "choose":
            draw.text((50, 90), "Choose the code from the most recent UPDATE record", fill=(40, 48, 62), font=_font())
            labels = tuple(f"SLOT {i + 1}: {code}" for i, code in enumerate(state["options"]))
        elif phase == "submit": draw.text((320, 210), f"Selected slot {state['choice'] + 1}", fill=(40, 48, 62), font=_font()); labels = ("Submit", "Cancel")
        else: labels = ()
        for index, label in enumerate(labels):
            cols = 2 if len(labels) > 2 else max(len(labels), 1); row, col = divmod(index, cols); left = 35 + col * 465; top = (125 + row * 100) if phase == "choose" else 445
            _draw_button(draw, (left, top, left + 425, top + 72), label)
        Path(path).parent.mkdir(parents=True, exist_ok=True); image.save(path, format="PNG")


class HiddenMemoryTask(VisualDelayedTask):
    """A cue, a harmless intervening screen, then a delayed visual choice."""

    task_family = "hidden_memory"
    instruction = "Remember the banner color, then select that color after it disappears."
    colors = ("red", "blue", "green")

    def initial_state(self, seed: int) -> dict[str, Any]:
        return {"phase": "cue", "cue": self.colors[int(seed) % len(self.colors)], "choice": None, "history": [], "seed": int(seed)}

    def available_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "cue":
            return ("continue",)
        if state["phase"] == "distractor":
            return ("acknowledge",)
        if state["phase"] == "choose":
            return tuple(f"choose_{color}" for color in self.colors)
        if state["phase"] == "submit":
            return ("submit", "cancel")
        return ()

    def observe(self, state: dict[str, Any]) -> dict[str, Any]:
        phase = state["phase"]
        if phase == "cue":
            screen = f"A banner shows the color {state['cue']}. Press continue."
        elif phase == "distractor":
            screen = "A neutral status screen contains no color cue. Press acknowledge."
        elif phase == "choose":
            screen = "The banner disappeared. Select the color you remember."
        elif phase == "submit":
            screen = f"Selected color: {state['choice']}. Press submit."
        else:
            screen = "Task finished."
        return {"screen": screen, "candidates": list(self.available_actions(state))}

    def critical_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        return tuple(self.available_actions(state)) if state["phase"] == "choose" else ()

    def optimal_actions(self, state: dict[str, Any]) -> tuple[str, ...]:
        if state["phase"] == "choose":
            return (f"choose_{state['cue']}",)
        if state["phase"] == "submit" and state.get("choice") == state.get("cue"):
            return ("submit",)
        return tuple(self.available_actions(state)) if state["phase"] in {"cue", "distractor"} else ()

    def memory_advantages(self, state: dict[str, Any], history_count: int) -> tuple[float, ...]:
        values = [0.0] * history_count
        # The cue is already useful on the harmless intervening screen because
        # it determines a later choice, even though the current action itself
        # is not critical.  This is precisely why action-critical labels are
        # not valid substitutes for memory advantage.
        if state["phase"] in {"distractor", "choose"} and values:
            values[0] = 1.0
        return tuple(values)

    def step(self, state: dict[str, Any], action: str) -> tuple[dict[str, Any], bool]:
        state = self.clone(state)
        state["history"].append(action)
        phase = state["phase"]
        if phase == "cue" and action == "continue":
            state["phase"] = "distractor"
            return state, False
        if phase == "distractor" and action == "acknowledge":
            state["phase"] = "choose"
            return state, False
        if phase == "choose" and action in self.available_actions(state):
            state["choice"] = action[len("choose_"):]
            state["phase"] = "submit"
            return state, False
        if phase == "submit":
            state.update(phase="terminal", terminal_return=float(action == "submit" and state.get("choice") == state.get("cue")))
            return state, True
        state.update(phase="terminal", terminal_return=0.0)
        return state, True

    def render_screenshot(self, state: dict[str, Any], path: str | Path) -> None:
        image = Image.new("RGB", (960, 600), (248, 250, 253))
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 960, 68), fill=(35, 48, 68))
        draw.text((28, 22), "TANGO hidden-memory task", fill="white", font=_font())
        draw.text((28, 100), self.instruction, fill=(30, 38, 50), font=_font())
        phase = state["phase"]
        if phase == "cue":
            color = _PALETTE[state["cue"]]
            draw.rounded_rectangle((50, 150, 910, 240), radius=12, fill=color)
            draw.text((75, 185), f"MEMORIZE: {state['cue'].upper()}", fill="white", font=_font())
            labels = ("Continue",)
        elif phase == "distractor":
            draw.text((50, 170), "Neutral status update — no color cue is visible.", fill=(30, 38, 50), font=_font())
            labels = ("Acknowledge",)
        elif phase == "choose":
            draw.text((50, 170), "The banner is gone. Which color was shown?", fill=(30, 38, 50), font=_font())
            labels = tuple(color.title() for color in self.colors)
        elif phase == "submit":
            draw.text((50, 170), f"Selected color: {state['choice']}", fill=(30, 38, 50), font=_font())
            labels = ("Submit", "Cancel")
        else:
            draw.text((50, 170), "Task finished.", fill=(30, 38, 50), font=_font())
            labels = ()
        width = 260 if len(labels) >= 3 else 350
        gap = 25
        for index, label in enumerate(labels):
            left = 55 + index * (width + gap)
            fill = _PALETTE.get(label.lower(), (238, 242, 248)) if phase == "choose" else (238, 242, 248)
            _draw_button(draw, (left, 275, left + width, 355), label, fill=fill)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path, format="PNG")


def visual_delayed_tasks() -> tuple[VisualDelayedTask, ...]:
    """Return the controlled visual families used by the pilot."""
    return (DistractorCreditTask(), HiddenMemoryTask(), MultiCueBindingTask(), InterferenceUpdateTask(), NonceVisualBindingTask(), InterferenceChainTask())


def visual_delayed_task_registry() -> dict[str, VisualDelayedTask]:
    """Return fresh task instances keyed by stable family name."""
    return {task.task_family: task for task in visual_delayed_tasks()}


def build_visual_counterfactual_dataset(
    seeds: Iterable[int] = range(500),
    *,
    output_dir: str | Path = "data/visual_delayed/images",
    tasks: Iterable[VisualDelayedTask] | None = None,
    horizon: int = 8,
    aggregation: str = "mean",
) -> list[dict[str, Any]]:
    """Build visual same-prefix rows and render one PNG for every prefix."""
    destination = Path(output_dir)
    selected_tasks = tuple(tasks) if tasks is not None else visual_delayed_tasks()
    rows: list[dict[str, Any]] = []
    # Prefixes are shared by all candidate rows.  Render the visual history
    # once per prefix and pass it explicitly to the VLM; otherwise a delayed
    # cue benchmark silently degenerates into a current-screenshot benchmark.
    rendered_histories: dict[str, list[str]] = {}
    rendered_states: dict[tuple[str, int, str], str] = {}
    for task in selected_tasks:
        for seed in tuple(int(item) for item in seeds):
            examples = build_counterfactual_examples(task, seed, horizon=horizon, aggregation=aggregation)
            for row in examples:
                image_path = destination / f"{task.task_family}_s{seed}_{row['prefix_id'].rsplit('-', 1)[-1]}.png"
                task.render_screenshot(row["prefix"]["state"], image_path)
                current_key = (task.task_family, seed, json.dumps(row["prefix"]["state"], sort_keys=True))
                rendered_states[current_key] = str(image_path)
                prefix_id = str(row["prefix_id"])
                history_images = rendered_histories.get(prefix_id)
                if history_images is None:
                    history_images = []
                    replay_state = task.initial_state(seed)
                    prefix_history = list(row["prefix"].get("history", []))
                    # Include observations that preceded the current decision,
                    # including the initial cue screenshot.  The current
                    # screenshot remains in ``image`` and is not duplicated.
                    if prefix_history:
                        state_key = (task.task_family, seed, json.dumps(replay_state, sort_keys=True))
                        cached = rendered_states.get(state_key)
                        if cached is None:
                            history_path = destination / f"{task.task_family}_s{seed}_state{len(rendered_states)}.png"
                            task.render_screenshot(replay_state, history_path)
                            cached = str(history_path)
                            rendered_states[state_key] = cached
                        history_images.append(cached)
                    for history_index, history_action in enumerate(prefix_history, start=1):
                        replay_state, _ = task.step(replay_state, str(history_action))
                        # The final replay state is the current screenshot and
                        # is deliberately kept out of history_images.
                        if history_index < len(prefix_history):
                            state_key = (task.task_family, seed, json.dumps(replay_state, sort_keys=True))
                            cached = rendered_states.get(state_key)
                            if cached is None:
                                history_path = destination / f"{task.task_family}_s{seed}_state{len(rendered_states)}.png"
                                task.render_screenshot(replay_state, history_path)
                                cached = str(history_path)
                                rendered_states[state_key] = cached
                            history_images.append(cached)
                    # The final replay state should agree with the immutable
                    # prefix state; fail loudly if a collector changes them.
                    if replay_state != row["prefix"]["state"]:
                        raise ValueError(f"history replay mismatch for {prefix_id}")
                    rendered_histories[prefix_id] = history_images
                observation = row["prefix"]["observation"]
                critical_actions = list(task.critical_actions(row["prefix"]["state"]))
                optimal_actions = list(task.optimal_actions(row["prefix"]["state"]))
                memory_advantages = list(
                    task.memory_advantages(row["prefix"]["state"], len(history_images))
                )
                memory_roles = list(task.memory_roles(row["prefix"]["state"], len(history_images)))
                row.update(
                    benchmark=task.benchmark,
                    visual=True,
                    image=str(image_path),
                    screenshot_path=str(image_path),
                    history_images=list(history_images),
                    memory_advantages=memory_advantages,
                    memory_advantage_source="designer_role_prior",
                    memory_roles=memory_roles,
                    critical_step=bool(critical_actions),
                    critical_actions=critical_actions,
                    optimal_actions=optimal_actions,
                    is_critical_action=row["action"] in critical_actions,
                    prompt=(
                        "You are a GUI policy. Inspect the screenshot and choose exactly one candidate action. "
                        f"Task: {row['instruction']} Candidates: {row['candidate_actions']}"
                    ),
                )
                row["prefix"]["image"] = str(image_path)
                row["prefix"]["screenshot_path"] = str(image_path)
                row["prefix"]["history_images"] = list(history_images)
                row["prefix"]["memory_advantages"] = memory_advantages
                row["prefix"]["memory_roles"] = memory_roles
                row["prefix"]["critical_step"] = bool(critical_actions)
                row["prefix"]["critical_actions"] = critical_actions
                row["prefix"]["optimal_actions"] = optimal_actions
                # The text observation remains available for non-vision
                # baselines; the screenshot is the primary VLM input.
                observation["screenshot_path"] = str(image_path)
                rows.append(row)
    return rows
