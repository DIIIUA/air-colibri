"""CPU checks for the LAN server: request limits, auth, stop, preemption and disconnect detection.

A fake engine stands in for the GPU model; it polls `cancel_check` the way the real engine does
(every prefill chunk and decode step) and counts how many generations hold "VRAM" at once.
"""
import http.client
import json
import socket
import threading
import time

import pytest

from aircolibri_cli.lan_server import AirColibriLANServer
from aircolibri_cli.unified_runtime import GenerationCancelled


class FakeEngine:
    model_id = "fake-model"

    def __init__(self, prefill_steps=0, decode_steps=200, step_s=0.01):
        self.prefill_steps = prefill_steps
        self.decode_steps = decode_steps
        self.step_s = step_s
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.started = 0
        self.cleanups = 0
        self.cancel_phases = []
        self.last_metrics = {}
        self.first_token = threading.Event()

    def status_snapshot(self):
        return {"model_id": self.model_id, "gpu_name": "fake", "vram_mode": "normal", "mtp_enabled": False}

    def reset_session(self):
        pass

    def stream_chat(self, messages, max_new_tokens=16, enable_thinking=True, use_session_cache=False,
                    stop_on_eos=True, cancel_check=None):
        with self.lock:
            self.active += 1
            self.started += 1
            self.max_active = max(self.max_active, self.active)
        try:
            for _ in range(self.prefill_steps):
                if cancel_check and cancel_check():
                    self.cancel_phases.append("prefill")
                    raise GenerationCancelled("prefill")
                time.sleep(self.step_s)
            for i in range(min(self.decode_steps, max_new_tokens)):
                if cancel_check and cancel_check():
                    self.cancel_phases.append("decode")
                    raise GenerationCancelled("decode")
                time.sleep(self.step_s)
                self.first_token.set()
                yield {"type": "delta", "phase": "answer", "text": f"t{i} "}
            self.last_metrics = {"finish_reason": "length", "prompt_tokens": 3, "generated_tokens": max_new_tokens}
            yield {"type": "done", "thinking_text": "", "answer_text": "ok", "metrics": self.last_metrics}
        finally:
            with self.lock:
                self.active -= 1
                self.cleanups += 1


@pytest.fixture
def server(tmp_path):
    servers = []

    def make(engine=None, **kw):
        eng = engine or FakeEngine()
        srv = AirColibriLANServer(eng, host="127.0.0.1", http_port=0, enable_beacon=False,
                                  log_file=str(tmp_path / "logs" / "server.log"), **kw)
        srv.start_background()
        servers.append(srv)
        return srv, eng

    yield make
    for s in servers:
        s.stop()


def _post(port, path, body=None, headers=None, raw=None, timeout=10):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    data = raw if raw is not None else (json.dumps(body).encode() if body is not None else b"")
    h = {"Content-Type": "application/json", "Content-Length": str(len(data))}
    h.update(headers or {})
    conn.request("POST", path, body=data, headers=h)
    resp = conn.getresponse()
    payload = resp.read()
    conn.close()
    return resp.status, payload


def _start_stream(port, max_tokens=200):
    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "stream": True, "max_tokens": max_tokens}).encode()
    sock.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    )
    return sock


def _wait(cond, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_oversized_body_is_rejected_without_reading_it(server):
    srv, eng = server(max_request_bytes=1024)
    sock = socket.create_connection(("127.0.0.1", srv.http_port), timeout=5)
    sock.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Length: 50000000\r\n\r\n")
    reply = sock.recv(4096).decode("utf-8", "replace")
    sock.close()
    assert reply.startswith("HTTP/1.1 413")
    assert eng.started == 0


def test_bad_requests_get_clear_errors(server):
    srv, _ = server()
    assert _post(srv.http_port, "/v1/chat/completions", raw=b"{not json")[0] == 400
    assert _post(srv.http_port, "/v1/chat/completions", body={"messages": []})[0] == 400
    assert _post(srv.http_port, "/v1/chat/completions", body={"messages": [{"role": "user", "content": "x"}], "max_tokens": "abc"})[0] == 400


def test_api_key_is_enforced_and_non_ascii_keys_do_not_crash(server):
    srv, eng = server(api_key="sekret", require_auth=True)
    body = {"messages": [{"role": "user", "content": "x"}], "max_tokens": 2}
    assert _post(srv.http_port, "/v1/chat/completions", body=body)[0] == 401
    raw = json.dumps(body).encode()
    sock = socket.create_connection(("127.0.0.1", srv.http_port), timeout=5)
    sock.sendall(
        b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nAuthorization: Bearer "
        + "ключ".encode("utf-8") + f"\r\nContent-Length: {len(raw)}\r\n\r\n".encode() + raw
    )
    assert sock.recv(4096).startswith(b"HTTP/1.1 401")
    sock.close()
    assert eng.started == 0
    status, payload = _post(srv.http_port, "/v1/chat/completions", body=body, headers={"Authorization": "Bearer sekret"})
    assert status == 200 and json.loads(payload)["choices"][0]["message"]["content"] == "ok"


def test_stop_endpoint_cancels_running_generation(server):
    srv, eng = server(engine=FakeEngine(decode_steps=1000))
    sock = _start_stream(srv.http_port, max_tokens=1000)
    assert eng.first_token.wait(5)
    t0 = time.time()
    status, payload = _post(srv.http_port, "/v1/chat/stop")
    assert status == 200 and json.loads(payload)["stopped"] is True
    assert _wait(lambda: eng.cleanups == 1, 2.0), "generation was not cleaned up after stop"
    assert time.time() - t0 < 1.0
    data = b""
    while b"[DONE]" not in data:
        chunk = sock.recv(65536)
        if not chunk:
            break
        data += chunk
    sock.close()
    assert b"aircolibri_cancelled" in data


def test_new_request_preempts_old_one_and_never_overlaps(server):
    srv, eng = server(engine=FakeEngine(decode_steps=1000))
    first = _start_stream(srv.http_port, max_tokens=1000)
    assert eng.first_token.wait(5)
    status, payload = _post(srv.http_port, "/v1/chat/completions",
                            body={"messages": [{"role": "user", "content": "again"}], "max_tokens": 3})
    first.close()
    assert status == 200
    assert eng.max_active == 1, "two generations held VRAM at the same time"
    assert eng.cancel_phases and eng.cancel_phases[0] == "decode"
    assert srv.coordinator.snapshot()["preemptions"] >= 1


def test_client_disconnect_is_detected_during_prefill(server):
    srv, eng = server(engine=FakeEngine(prefill_steps=1000, step_s=0.01))
    sock = _start_stream(srv.http_port)
    assert _wait(lambda: eng.active == 1, 5.0)
    t0 = time.time()
    sock.close()  # ForgeMind's call.cancel(): FIN while the server is still in prefill
    assert _wait(lambda: eng.cleanups == 1, 3.0), "server did not notice the closed socket during prefill"
    assert time.time() - t0 < 1.5
    assert eng.cancel_phases == ["prefill"]


def test_server_log_is_written(server, tmp_path):
    srv, _ = server()
    _post(srv.http_port, "/v1/chat/completions", body={"messages": [{"role": "user", "content": "x"}], "max_tokens": 2})
    srv.stop()
    log = (tmp_path / "logs" / "server.log").read_text(encoding="utf-8")
    assert "server started" in log and " start " in log and " end ok " in log
