from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

from .androidworld_http import AndroidWorldHTTPClient


ANR_MARKERS = ("isn't responding", "is not responding", "application error")


def adb_output(adb: str, *args: str, timeout: float = 20.0) -> str:
    result = subprocess.run(
        [adb, "-e", "shell", *args], capture_output=True, text=True,
        timeout=timeout, check=False,
    )
    return (result.stdout + result.stderr).strip()


def inspect_androidworld(
    server_url: str,
    adb: str = "/opt/android/platform-tools/adb",
    require_kvm: bool = False,
    timeout: float = 60.0,
) -> dict:
    client = AndroidWorldHTTPClient(server_url, timeout=timeout)
    failures = []
    try:
        pixels = client.screenshot(wait_to_stabilize=False)
        server_health = client.health()
    except Exception as exc:  # preflight must report an absent guest, not crash
        pixels = np.zeros((0, 0, 3), dtype=np.uint8)
        server_health = False
        failures.append(f"AndroidWorld server unavailable: {type(exc).__name__}")
    activity = adb_output(adb, "service", "check", "activity")
    focus = adb_output(adb, "dumpsys", "window", "windows")
    system_pid = adb_output(adb, "pidof", "system_server")
    lowered_focus = focus.lower()
    result = {
        "server_health": server_health,
        "screen_shape": list(pixels.shape),
        "screen_mean": float(np.mean(pixels)) if pixels.size else None,
        "screen_std": float(np.std(pixels)) if pixels.size else None,
        "activity_service": activity,
        "system_server_pid": system_pid,
        "anr_dialog_visible": any(marker in lowered_focus for marker in ANR_MARKERS),
        "kvm_available": Path("/dev/kvm").exists(),
        "require_kvm": require_kvm,
    }
    if not result["server_health"]:
        if not any(item.startswith("AndroidWorld server unavailable") for item in failures):
            failures.append("server health endpoint failed")
    if pixels.size == 0 or result["screen_std"] < 1.0:
        failures.append("screenshot is blank or near-constant")
    if "found" not in activity.lower() or not system_pid.isdigit():
        failures.append("Android activity/system_server is unavailable")
    if result["anr_dialog_visible"]:
        failures.append("ANR/application-error dialog is visible")
    if require_kvm and not result["kvm_available"]:
        failures.append("/dev/kvm is unavailable")
    result["failures"] = failures
    result["ok"] = not failures
    return result
