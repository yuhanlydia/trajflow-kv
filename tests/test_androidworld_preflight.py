from unittest.mock import patch

import numpy as np

from trajflow_kv.androidworld_preflight import inspect_androidworld


class FakeClient:
    def __init__(self, *_args, **_kwargs):
        pass

    def health(self):
        return True

    def screenshot(self, wait_to_stabilize=False):  # noqa: ARG002
        image = np.zeros((8, 8, 3), dtype=np.uint8)
        image[0, 0] = 255
        return image


@patch("trajflow_kv.androidworld_preflight.Path.exists", return_value=True)
@patch("trajflow_kv.androidworld_preflight.adb_output")
@patch("trajflow_kv.androidworld_preflight.AndroidWorldHTTPClient", FakeClient)
def test_preflight_accepts_healthy_host(adb, _exists):
    adb.side_effect = ["Service activity: found", "mCurrentFocus=Launcher", "609"]
    result = inspect_androidworld("http://test", require_kvm=True)
    assert result["ok"] and not result["failures"]


@patch("trajflow_kv.androidworld_preflight.Path.exists", return_value=False)
@patch("trajflow_kv.androidworld_preflight.adb_output")
@patch("trajflow_kv.androidworld_preflight.AndroidWorldHTTPClient", FakeClient)
def test_preflight_rejects_anr_and_missing_kvm(adb, _exists):
    adb.side_effect = ["Service activity: found", "Process system isn't responding", "609"]
    result = inspect_androidworld("http://test", require_kvm=True)
    assert not result["ok"]
    assert "ANR/application-error dialog is visible" in result["failures"]
    assert "/dev/kvm is unavailable" in result["failures"]
