from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from .actions import InvalidAction, canonical_action, parse_action
from .projector import attach_kv_projectors


SYSTEM = """You control Android. Return exactly one valid JSON action and no prose.
Use double quotes around every key and string. Inspect the current screenshot and
take one action that advances the task. Do not claim completion until the visible
state proves the task is complete. Coordinates refer to the full screen resolution
given below, not the resized image tensor.
Valid examples:
{"action_type":"click","x":120,"y":300}
{"action_type":"scroll","direction":"down"}
{"action_type":"input_text","text":"hello"}
{"action_type":"open_app","app_name":"Contacts"}
{"action_type":"navigate_back"}
{"action_type":"status","goal_status":"complete"}
Coordinates may be absolute pixels or normalized floats in [0,1]."""


def action_signature(action: dict) -> tuple:
    """Signature for loop detection; ignore incidental generated coordinates."""
    action_type = action.get("action_type")
    if action_type in {"swipe", "scroll"}:
        return "pan", action.get("direction")
    return (action_type,)


def exclude_repeated_candidates(
    candidates: list[str], history: list[str], screen_size: tuple[int, int], limit: int
) -> list[str]:
    kept = []
    for candidate in candidates:
        candidate_action = canonical_action(parse_action(candidate, screen_size))
        repeated = 0
        for previous in reversed(history):
            if canonical_action(parse_action(previous, screen_size)) != candidate_action:
                break
            repeated += 1
        if repeated < limit:
            kept.append(candidate)
    return kept or candidates


def build_action_prompt(instruction: str, history: list[str], screen_size: tuple[int, int]) -> str:
    return (f"{SYSTEM}\nScreen size: {screen_size[0]}x{screen_size[1]}\n"
            f"Task: {instruction}\nHistory: {json.dumps(history)}")


class QwenKVPolicy:
    def __init__(self, model_path: str, checkpoint: str | None = None, *, rank: int = 8,
                 alpha: float = 0.1, target: str = "both", device: str = "cuda",
                 max_pixels: int = 401408, temperature: float = 0.7,
                 max_new_tokens: int = 128,
                 last_n_layers: int | None = None,
                 candidate_mode: str | None = None,
                 max_identical_actions: int | None = None,
                 max_identical_candidates: int | None = None,
                 state_router_checkpoint: str | None = None):
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        self.processor = AutoProcessor.from_pretrained(model_path, max_pixels=max_pixels)
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, dtype=torch.bfloat16, device_map=device
        ).eval()
        self.bundle = attach_kv_projectors(
            self.model, rank, alpha, target, last_n_layers
        )
        if checkpoint:
            state = torch.load(Path(checkpoint), map_location=device, weights_only=True)
            self.bundle.modules.load_state_dict(state)
        self.device = device
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.candidate_mode = candidate_mode
        self.max_identical_actions = max_identical_actions
        self.max_identical_candidates = (
            max_identical_candidates
            if max_identical_candidates is not None
            else max_identical_actions
        )
        if state_router_checkpoint:
            from .state_router import StateRouter
            self.state_router = StateRouter.load(state_router_checkpoint)
        else:
            self.state_router = None

    def _system_candidates(self, instruction: str) -> list[str]:
        lower = instruction.lower()
        common = [
            canonical_action({"action_type": "swipe", "direction": "down"}),
            canonical_action({"action_type": "navigate_back"}),
            canonical_action({"action_type": "wait"}),
        ]
        if "wifi" in lower or "wi-fi" in lower:
            return common + [
                canonical_action({"action_type": "click", "x": 260, "y": 200}),
                canonical_action({"action_type": "click", "x": 260, "y": 340}),
                canonical_action({"action_type": "click", "x": 870, "y": 920}),
            ]
        if "bluetooth" in lower:
            return common + [
                canonical_action({"action_type": "click", "x": 780, "y": 200}),
                canonical_action({"action_type": "click", "x": 780, "y": 340}),
            ]
        return common

    @torch.inference_mode()
    def _rank_candidates(
        self, instruction: str, image: Image.Image, history: list[str],
        screen_size: tuple[int, int], candidates: list[str]
    ) -> str:
        prompt = build_action_prompt(instruction, history, screen_size)
        scores = []
        for candidate in candidates:
            user = {"role": "user", "content": [
                {"type": "image", "image": image}, {"type": "text", "text": prompt}
            ]}
            prompt_batch = self.processor.apply_chat_template(
                [user], tokenize=True, add_generation_prompt=True,
                return_dict=True, return_tensors="pt"
            )
            batch = self.processor.apply_chat_template(
                [user, {"role": "assistant", "content": [
                    {"type": "text", "text": candidate}
                ]}],
                tokenize=True, return_dict=True, return_tensors="pt"
            ).to(self.device)
            labels = batch["input_ids"].clone()
            labels[:, :prompt_batch["input_ids"].shape[1]] = -100
            output = self.model(**batch, labels=labels, use_cache=False)
            # Length-normalize so short actions (notably ``wait``/``swipe``)
            # do not win solely because they contain fewer JSON tokens.
            scores.append(-float(output.loss))
        score_tensor = torch.tensor(scores)
        if self.temperature > 0:
            probabilities = torch.softmax(score_tensor / self.temperature, dim=0)
            index = int(torch.multinomial(probabilities, 1))
        else:
            index = int(score_tensor.argmax())
        return candidates[index]

    @torch.inference_mode()
    def _generate(self, instruction: str, image: Image.Image, history: list[str],
                  screen_size: tuple[int, int]) -> str:
        prompt = build_action_prompt(instruction, history, screen_size)
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image}, {"type": "text", "text": prompt}
        ]}]
        batch = self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt"
        ).to(self.device)
        sampling = ({"do_sample": True, "temperature": self.temperature, "top_p": 0.9}
                    if self.temperature > 0 else {"do_sample": False, "temperature": None, "top_p": None})
        generated = self.model.generate(**batch, max_new_tokens=self.max_new_tokens, **sampling)
        suffix = generated[0, batch["input_ids"].shape[1]:]
        return self.processor.decode(suffix, skip_special_tokens=True).strip()

    @torch.inference_mode()
    def act(self, instruction: str, image: Image.Image, history: list[str],
            screen_size: tuple[int, int]) -> str:
        candidates = self._system_candidates(instruction)
        if self.candidate_mode == "system":
            return self._rank_candidates(
                instruction, image, history, screen_size, candidates,
            )
        if self.candidate_mode == "system_hierarchical":
            proposal = None
            if self.max_identical_candidates and history:
                try:
                    candidates = exclude_repeated_candidates(
                        candidates, history, screen_size, self.max_identical_candidates
                    )
                except InvalidAction:
                    pass
            try:
                if self.state_router is not None:
                    proposed_type = self.state_router.predict(
                        image, len(history), instruction
                    )
                else:
                    proposal = self._generate(instruction, image, history, screen_size)
                    proposed_type = parse_action(proposal, screen_size)["action_type"]
                same_type = [
                    candidate for candidate in candidates
                    if (
                        action_signature(json.loads(candidate))[0] == proposed_type
                        or json.loads(candidate)["action_type"] == proposed_type
                    )
                ]
            except InvalidAction:
                same_type = []
            if self.max_identical_actions and history and proposal is not None:
                try:
                    proposed_action = parse_action(proposal, screen_size)
                    proposed_signature = action_signature(proposed_action)
                    repeated = 0
                    for previous in reversed(history):
                        if action_signature(parse_action(previous, screen_size)) != proposed_signature:
                            break
                        repeated += 1
                    if proposed_signature[0] == "pan" and repeated >= self.max_identical_actions:
                        candidates = [
                            item for item in candidates
                            if action_signature(json.loads(item)) != proposed_signature
                        ]
                        same_type = [
                            item for item in same_type
                            if action_signature(json.loads(item)) != proposed_signature
                        ]
                except InvalidAction:
                    pass
            return self._rank_candidates(
                instruction, image, history, screen_size, same_type or candidates,
            )
        return self._generate(instruction, image, history, screen_size)
