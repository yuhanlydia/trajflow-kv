import numpy as np

from trajflow_kv.androidworld_http import AndroidWorldHTTPClient


class Response:
    def __init__(self, payload): self.payload = payload
    def raise_for_status(self): pass
    def json(self): return self.payload


class Session:
    def __init__(self): self.calls = []
    def request(self, method, url, timeout, **kwargs):
        self.calls.append((method, url, kwargs))
        if url.endswith("/health"): return Response({"status": "success"})
        if url.endswith("/screenshot"): return Response({"pixels": np.zeros((3, 5, 3), dtype=int).tolist()})
        if url.endswith("/task/goal"): return Response({"goal": "Set alarm"})
        if url.endswith("/task/score"): return Response({"score": 1})
        return Response({"status": "success"})


def test_official_http_protocol():
    session = Session(); client = AndroidWorldHTTPClient("http://aw:5000/", session=session)
    assert client.health()
    assert client.screenshot().shape == (3, 5, 3)
    assert client.screen_size == (5, 3)
    client.initialize_task("Clock", 0)
    client.execute({"action_type": "wait"})
    assert client.goal("Clock", 0) == "Set alarm"
    assert client.score("Clock", 0) == 1.0
    assert any(call[1].endswith("/execute_action") for call in session.calls)
