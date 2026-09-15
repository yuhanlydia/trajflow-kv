from __future__ import annotations

from typing import Any

import numpy as np
import requests


class AndroidWorldHTTPClient:
    """Small client matching AndroidWorld's official FastAPI server."""

    def __init__(self, base_url: str = "http://localhost:5000", timeout: float = 60.0,
                 session: Any = None):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = session or requests.Session()
        self._screen_size: tuple[int, int] | None = None

    def _request(self, method: str, path: str, **kwargs):
        response = self.session.request(
            method, f"{self.base_url}{path}", timeout=self.timeout, **kwargs
        )
        response.raise_for_status()
        return response.json()

    def health(self) -> bool:
        try:
            return self._request("GET", "/health").get("status") == "success"
        except requests.RequestException:
            return False

    def reset(self, go_home: bool = True) -> None:
        self._request("POST", "/reset", params={"go_home": str(go_home).lower()})

    def screenshot(self, wait_to_stabilize: bool = True) -> np.ndarray:
        payload = self._request("GET", "/screenshot", params={
            "wait_to_stabilize": str(wait_to_stabilize).lower()
        })
        pixels = np.asarray(payload["pixels"], dtype=np.uint8)
        if pixels.ndim != 3 or pixels.shape[2] not in (3, 4):
            raise ValueError(f"invalid screenshot shape: {pixels.shape}")
        self._screen_size = (pixels.shape[1], pixels.shape[0])
        return pixels[:, :, :3]

    @property
    def screen_size(self) -> tuple[int, int]:
        if self._screen_size is None:
            self.screenshot()
        assert self._screen_size is not None
        return self._screen_size

    def execute(self, action: dict[str, Any]) -> None:
        self._request("POST", "/execute_action", json=action)

    def initialize_task(self, task_type: str, task_idx: int) -> None:
        self._request("POST", "/task/initialize", params={"task_type": task_type, "task_idx": task_idx})

    def tear_down_task(self, task_type: str, task_idx: int) -> None:
        self._request("POST", "/task/tear_down", params={"task_type": task_type, "task_idx": task_idx})

    def goal(self, task_type: str, task_idx: int) -> str:
        return str(self._request("GET", "/task/goal", params={
            "task_type": task_type, "task_idx": task_idx
        })["goal"])

    def score(self, task_type: str, task_idx: int) -> float:
        return float(self._request("GET", "/task/score", params={
            "task_type": task_type, "task_idx": task_idx
        })["score"])

