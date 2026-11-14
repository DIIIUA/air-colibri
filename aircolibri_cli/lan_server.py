"""Lightweight Zero-Dependency LAN Server + UDP Self-Broadcaster for AirColibri.

Provides:
- Automatic LAN IPv4 discovery (`detect_lan_ips`) + clear console Connection Passport
- UDP LAN Beacon (`UDPLANBeacon` on port 8788): broadcasts JSON presence every 3s to
  `<subnet>.255` + `255.255.255.255` and replies immediately to `AIRCOLIBRI_DISCOVER?`
- `discover_lan_servers()` client scanner for `aircolibri discover`
- Multi-threaded `ThreadingHTTPServer` (`0.0.0.0:8787`) serving:
  - `GET /` — Built-in zero-CDN dark-mode LAN Web Chat UI (works offline on any phone/PC!)
  - `GET /api/info` & `GET /health` — Live node telemetry + connection credentials
  - `GET /v1/models` — OpenAI-compatible model listing
  - `POST /v1/chat/completions` — OpenAI-compatible JSON & `stream: true` SSE
    (`delta.reasoning_content` for `<think>` + `delta.content` for final answer)
  - `POST /v1/chat/stop` — stop the running generation immediately
- One generation at a time: a new request (or a closed client socket, detected even during
  prefill and `<think>` buffering) cancels the running one and waits for its VRAM cleanup
- Rotating server log (`logs/server.log`), request body limit (413), optional Bearer API key
"""
from __future__ import annotations

from contextlib import contextmanager
import hmac
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import ipaddress
import json
import logging
import logging.handlers
from pathlib import Path
import select
import socket
import threading
import time
import traceback
from typing import Any
import uuid


DISCOVER_MAGIC = b"AIRCOLIBRI_DISCOVER?"
DEFAULT_HTTP_PORT = 8787
DEFAULT_BEACON_PORT = 8788


def detect_lan_ips() -> tuple[str, list[str]]:
    """Detects primary route LAN IPv4 and all non-loopback IPv4 interfaces on the host."""
    primary = "127.0.0.1"
    for probe_target in ("192.168.1.1", "10.255.255.255", "8.8.8.8"):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                s.connect((probe_target, 80))
                cand = s.getsockname()[0]
                if cand and not cand.startswith("127.") and not cand.startswith("169.254."):
                    primary = cand
                    break
        except OSError:
            continue

    found: list[str] = []
    if primary != "127.0.0.1":
        found.append(primary)
    try:
        hostname = socket.gethostname()
        _, _, ips = socket.gethostbyname_ex(hostname)
        for ip in ips:
            if ip and not ip.startswith("127.") and not ip.startswith("169.254.") and ip not in found:
                found.append(ip)
    except OSError:
        pass

    # Prefer physical RFC1918 192.168.x.x / 10.x.x.x as primary if primary fell on virtual 172.16-31.x.x
    def _rank_ip(ip_s: str) -> int:
        if ip_s.startswith("192.168."):
            return 0
        if ip_s.startswith("10."):
            return 1
        return 2

    if found:
        found_sorted = sorted(found, key=lambda x: (0 if x == primary and _rank_ip(x) <= 1 else 1, _rank_ip(x)))
        primary = found_sorted[0]
        return primary, found_sorted
    return "127.0.0.1", ["127.0.0.1"]


def _subnet_broadcast_targets(lan_ips: list[str]) -> list[str]:
    targets = ["255.255.255.255"]
    for ip_s in lan_ips:
        try:
            ip_obj = ipaddress.IPv4Address(ip_s)
            if ip_obj.is_loopback:
                continue
            parts = ip_s.split(".")
            bcast24 = f"{parts[0]}.{parts[1]}.{parts[2]}.255"
            if bcast24 not in targets:
                targets.append(bcast24)
        except ValueError:
            continue
    return targets


class UDPLANBeacon:
    """Broadcasts server presence over UDP and answers `AIRCOLIBRI_DISCOVER?` queries on the LAN."""

    def __init__(
        self,
        payload_fn,
        lan_ips: list[str],
        beacon_port: int = DEFAULT_BEACON_PORT,
        interval_s: float = 3.0,
    ):
        self.payload_fn = payload_fn
        self.lan_ips = lan_ips
        self.beacon_port = int(beacon_port)
        self.interval_s = float(interval_s)
        self._stop = threading.Event()
        self._sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self.packets_sent = 0
        self.discoveries_answered = 0

    def start(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", self.beacon_port))
        sock.settimeout(0.5)
        self._sock = sock
        self._thread = threading.Thread(target=self._loop, name="aircolibri-lan-beacon", daemon=True)
        self._thread.start()

    def _encode_packet(self) -> bytes:
        data = self.payload_fn()
        return json.dumps(data, ensure_ascii=False).encode("utf-8")

    def _loop(self) -> None:
        targets = _subnet_broadcast_targets(self.lan_ips)
        next_bcast = 0.0
        assert self._sock is not None
        while not self._stop.is_set():
            now = time.monotonic()
            if now >= next_bcast:
                pkt = self._encode_packet()
                for tgt in targets:
                    try:
                        self._sock.sendto(pkt, (tgt, self.beacon_port))
                        self.packets_sent += 1
                    except OSError:
                        pass
                next_bcast = now + self.interval_s

            try:
                msg, addr = self._sock.recvfrom(4096)
                if msg.strip().startswith(DISCOVER_MAGIC):
                    pkt = self._encode_packet()
                    self._sock.sendto(pkt, addr)
                    self.discoveries_answered += 1
            except socket.timeout:
                continue
            except OSError:
                break

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except OSError:
                pass
            self._sock = None
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None


def discover_lan_servers(timeout_s: float = 2.0, beacon_port: int = DEFAULT_BEACON_PORT) -> list[dict[str, Any]]:
    """Actively probes the LAN via UDP broadcast + listens for AirColibri beacons."""
    _, lan_ips = detect_lan_ips()
    targets = _subnet_broadcast_targets(lan_ips)
    targets.append("127.0.0.1")

    discovered: dict[str, dict[str, Any]] = {}
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP) as probe_sock:
        probe_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        probe_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe_sock.bind(("0.0.0.0", 0))
        probe_sock.settimeout(0.25)

        deadline = time.monotonic() + float(timeout_s)
        sent_rounds = 0
        while time.monotonic() < deadline:
            if sent_rounds < 2:
                for tgt in targets:
                    try:
                        probe_sock.sendto(DISCOVER_MAGIC, (tgt, int(beacon_port)))
                    except OSError:
                        pass
                sent_rounds += 1
            try:
                raw, addr = probe_sock.recvfrom(8192)
                info = json.loads(raw.decode("utf-8", errors="replace"))
                if info.get("service") == "aircolibri":
                    key = f"{info.get('lan_ip', addr[0])}:{info.get('http_port', DEFAULT_HTTP_PORT)}"
                    info["responder_ip"] = addr[0]
                    discovered[key] = info
            except (socket.timeout, json.JSONDecodeError):
                continue
            except OSError:
                break

    return list(discovered.values())


def format_connection_passport(passport: dict[str, Any]) -> str:
    """Formats a crisp, copy-paste-ready LAN Connection Passport for terminal output."""
    lan_url = passport["lan_web_url"]
    local_url = passport["local_web_url"]
    oai_url = passport["openai_base_url"]
    model_id = passport["model_id"]
    api_key = passport["api_key"]
    auth_mode = "REQUIRED (Bearer)" if passport.get("require_auth") else "Optional (accepts any Bearer key)"
    extra_ips = [ip for ip in passport.get("all_lan_ips", []) if ip != passport["lan_ip"]]
    extra_line = f"\n  Alt LAN IPs     : {', '.join(f'http://{ip}:{passport['http_port']}' for ip in extra_ips)}" if extra_ips else ""

    return (
        "\n"
        "================================================================================\n"
        "  AIRCOLIBRI LAN NODE — READY FOR CONNECTIONS\n"
        "================================================================================\n"
        f"  Node Hostname   : {passport['hostname']} ({passport['gpu_name']}, mode={passport['vram_mode']})\n"
        f"  Model ID        : {model_id}\n"
        f"  Local Browser   : {local_url}\n"
        f"  LAN Web Chat    : {lan_url}{extra_line}\n"
        f"  OpenAI Base URL : {oai_url}\n"
        f"  API Key         : {api_key}  [{auth_mode}]\n"
        f"  LAN UDP Beacon  : UDP :{passport['beacon_port']} (broadcast every 3s + `aircolibri discover`)\n"
        "--------------------------------------------------------------------------------\n"
        "  1) Find from any PC on the same Wi-Fi / LAN:\n"
        "       aircolibri.cmd discover\n\n"
        "  2) Quick curl test (streaming SSE):\n"
        f'       curl -N {oai_url}/chat/completions -H "Content-Type: application/json" '
        f'-H "Authorization: Bearer {api_key}" '
        f'-d "{{\\"model\\":\\"{model_id}\\",\\"stream\\":true,\\"messages\\":[{{\\"role\\":\\"user\\",\\"content\\":\\"Привет!\\"}}]}}"\n\n'
        "  3) Connect Cursor / Continue / OpenWebUI / Python `openai`:\n"
        f'       client = OpenAI(base_url="{oai_url}", api_key="{api_key}")\n'
        "--------------------------------------------------------------------------------\n"
        "  Note (Windows Firewall): if other LAN devices cannot reach port "
        f"{passport['http_port']}, allow TCP {passport['http_port']} & UDP {passport['beacon_port']}:\n"
        f'       netsh advfirewall firewall add rule name="AirColibri LAN" dir=in action=allow protocol=TCP localport={passport["http_port"]}\n'
        "================================================================================\n"
    )


WEB_UI_HTML = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AirColibri LAN Chat</title>
<style>
  :root { --bg:#0b0f17; --panel:#131a26; --border:#233044; --txt:#e6edf7; --muted:#8b9bb4; --acc:#38bdf8; --ok:#34d399; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif; background:var(--bg); color:var(--txt); display:flex; flex-direction:column; height:100vh; }
  header { padding:12px 18px; background:var(--panel); border-bottom:1px solid var(--border); display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:10px; }
  .brand { font-weight:700; font-size:16px; display:flex; align-items:center; gap:8px; }
  .dot { width:9px; height:9px; border-radius:50%; background:var(--ok); display:inline-block; box-shadow:0 0 8px var(--ok); }
  .badges { display:flex; flex-wrap:wrap; gap:8px; font-size:12px; color:var(--muted); }
  .badge { background:#1a2436; border:1px solid var(--border); padding:4px 9px; border-radius:999px; }
  main { flex:1; overflow-y:auto; padding:18px; display:flex; flex-direction:column; gap:14px; max-width:940px; width:100%; margin:0 auto; }
  .msg { padding:13px 16px; border-radius:12px; line-height:1.55; white-space:pre-wrap; word-break:break-word; }
  .user { align-self:flex-end; background:#1e3a5f; border:1px solid #2d5488; max-width:82%; }
  .assistant { align-self:flex-start; background:var(--panel); border:1px solid var(--border); width:100%; }
  details.think { margin-bottom:10px; background:#0e1420; border:1px dashed #2b3b55; border-radius:8px; padding:8px 12px; color:var(--muted); font-size:13px; }
  details.think summary { cursor:pointer; color:var(--acc); font-weight:600; user-select:none; }
  .meta-foot { margin-top:8px; font-size:11.5px; color:var(--muted); }
  footer { padding:12px 18px; background:var(--panel); border-top:1px solid var(--border); }
  .bar { max-width:940px; margin:0 auto; display:flex; gap:10px; align-items:center; }
  textarea { flex:1; background:#0d131f; color:var(--txt); border:1px solid var(--border); border-radius:10px; padding:10px 12px; font-size:14px; resize:none; height:48px; font-family:inherit; }
  button { background:var(--acc); color:#06121e; border:none; border-radius:10px; padding:0 18px; height:48px; font-weight:700; cursor:pointer; }
  button:disabled { opacity:0.5; cursor:not-allowed; }
  label.chk { font-size:12.5px; color:var(--muted); display:flex; align-items:center; gap:5px; cursor:pointer; }
</style>
</head>
<body>
<header>
  <div class="brand"><span class="dot"></span> AirColibri LAN Node</div>
  <div class="badges" id="badges">Загрузка паспорта узла...</div>
</header>
<main id="chat"></main>
<footer>
  <div class="bar">
    <label class="chk"><input type="checkbox" id="thinkChk" checked> &lt;think&gt;</label>
    <textarea id="inp" placeholder="Напиши запрос модели в локальной сети... (Enter — отправить, Shift+Enter — перенос)"></textarea>
    <button id="sendBtn">Отправить</button>
  </div>
</footer>
<script>
const chat = document.getElementById('chat');
const inp = document.getElementById('inp');
const sendBtn = document.getElementById('sendBtn');
const badges = document.getElementById('badges');
const thinkChk = document.getElementById('thinkChk');
let historyMsgs = [];

async function refreshInfo() {
  try {
    const r = await fetch('/api/info');
    const d = await r.json();
    const st = d.engine || {};
    const lm = st.last_metrics || {};
    const tps = lm.decode_tok_per_s ? `${lm.decode_tok_per_s} tok/s` : 'ready';
    badges.innerHTML = `
      <span class="badge">Model: ${d.model_id}</span>
      <span class="badge">GPU: ${d.gpu_name} (${d.vram_mode})</span>
      <span class="badge">VRAM: ${st.allocated_vram_mib || 0} MiB</span>
      <span class="badge">Speed: ${tps}</span>
      <span class="badge">OpenAI API: ${d.openai_base_url}</span>`;
  } catch (e) {}
}
refreshInfo();

let activeCtrl = null;
inp.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); if (!activeCtrl) sendPrompt(); }
});
sendBtn.onclick = () => { if (activeCtrl) stopGeneration(); else sendPrompt(); };

function stopGeneration() {
  if (activeCtrl) activeCtrl.abort();
  fetch('/v1/chat/stop', {method: 'POST'}).catch(() => {});
}

async function sendPrompt() {
  const text = inp.value.trim();
  if (!text || activeCtrl) return;
  inp.value = '';
  activeCtrl = new AbortController();
  sendBtn.textContent = 'Стоп';
  historyMsgs.push({role: 'user', content: text});

  const uDiv = document.createElement('div');
  uDiv.className = 'msg user';
  uDiv.textContent = text;
  chat.appendChild(uDiv);

  const aDiv = document.createElement('div');
  aDiv.className = 'msg assistant';
  const details = document.createElement('details');
  details.className = 'think';
  details.open = true;
  details.style.display = 'none';
  const summary = document.createElement('summary');
  summary.textContent = 'Размышление (<think>)...';
  const thinkBody = document.createElement('div');
  details.appendChild(summary);
  details.appendChild(thinkBody);
  const ansBody = document.createElement('div');
  const metaFoot = document.createElement('div');
  metaFoot.className = 'meta-foot';
  aDiv.appendChild(details);
  aDiv.appendChild(ansBody);
  aDiv.appendChild(metaFoot);
  chat.appendChild(aDiv);
  chat.scrollTop = chat.scrollHeight;

  let fullAns = '';
  try {
    const resp = await fetch('/v1/chat/completions', {
      method: 'POST',
      signal: activeCtrl.signal,
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        messages: historyMsgs,
        stream: true,
        max_tokens: 1024,
        enable_thinking: thinkChk.checked
      })
    });
    const reader = resp.body.getReader();
    const dec = new TextDecoder();
    let buf = '';
    while (true) {
      const {value, done} = await reader.read();
      if (done) break;
      buf += dec.decode(value, {stream: true});
      const lines = buf.split('\n');
      buf = lines.pop();
      for (const ln of lines) {
        if (!ln.startsWith('data: ')) continue;
        const payload = ln.slice(6).trim();
        if (payload === '[DONE]') continue;
        const j = JSON.parse(payload);
        const delta = (j.choices && j.choices[0] && j.choices[0].delta) || {};
        if (delta.reasoning_content) {
          details.style.display = 'block';
          thinkBody.textContent += delta.reasoning_content;
        }
        if (delta.content) {
          if (details.open) details.open = false;
          summary.textContent = 'Показать размышление (<think>)';
          fullAns += delta.content;
          ansBody.textContent = fullAns;
        }
        if (j.aircolibri_metrics) {
          const m = j.aircolibri_metrics;
          const acc = (m.spec_gate && m.spec_gate.overall_accept_rate_pct) || 0;
          metaFoot.textContent = `${m.decode_tok_per_s} tok/s | ${m.generated_tokens} tok | MTP accept: ${acc}% | Peak VRAM: ${m.peak_vram_mib} MiB`;
        }
        chat.scrollTop = chat.scrollHeight;
      }
    }
    historyMsgs.push({role: 'assistant', content: fullAns});
  } catch (err) {
    if (err.name === 'AbortError') {
      metaFoot.textContent = 'Остановлено';
      historyMsgs.push({role: 'assistant', content: fullAns});
    } else {
      ansBody.textContent = 'Ошибка соединения: ' + err;
    }
  } finally {
    activeCtrl = null;
    sendBtn.textContent = 'Отправить';
    refreshInfo();
  }
}
</script>
</body>
</html>
"""


DEFAULT_MAX_REQUEST_BYTES = 8 * 1024 * 1024
DEFAULT_LOG_FILE = "logs/server.log"
MAX_TOKENS_LIMIT = 32768


class SafeRotatingFileHandler(logging.handlers.RotatingFileHandler):
    """Rotating log that keeps writing when Windows refuses the rename (WinError 32: file in use)."""

    def doRollover(self) -> None:  # noqa: N802 (stdlib name)
        try:
            super().doRollover()
        except OSError:
            if self.stream is None:
                self.stream = self._open()


def setup_server_logger(log_file: str | Path | None) -> logging.Logger:
    """Creates the `aircolibri.server` logger writing to a rotating UTF-8 file (10 MB x 5)."""
    log = logging.getLogger("aircolibri.server")
    log.setLevel(logging.INFO)
    log.propagate = False
    for h in list(log.handlers):
        log.removeHandler(h)
        h.close()
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(threadName)s] %(message)s")
    if log_file:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = SafeRotatingFileHandler(path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8", delay=True)
        fh.setFormatter(fmt)
        log.addHandler(fh)
    eh = logging.StreamHandler()
    eh.setLevel(logging.WARNING)
    eh.setFormatter(fmt)
    log.addHandler(eh)
    return log


def make_disconnect_probe(sock: socket.socket, interval_s: float = 0.05):
    """Returns a cheap callable that reports True once the client closed or reset the connection.

    The request body has been read completely, so a readable socket whose ``MSG_PEEK`` returns
    ``b""`` means the peer sent FIN (ForgeMind's ``call.cancel()``, a closed browser tab, ...).
    It runs during prefill and ``<think>`` buffering, when nothing is written to the socket.
    """
    state = {"last": 0.0, "closed": False}

    def probe() -> bool:
        if state["closed"]:
            return True
        now = time.monotonic()
        if now - state["last"] < interval_s:
            return False
        state["last"] = now
        try:
            readable, _, errored = select.select([sock], [], [sock], 0)
            if errored:
                state["closed"] = True
            elif readable and sock.recv(1, socket.MSG_PEEK) == b"":
                state["closed"] = True
        except (OSError, ValueError):
            state["closed"] = True
        return state["closed"]

    return probe


class GenerationTicket:
    """One request's claim on the GPU; its ``cancel_event`` is polled by the engine."""

    def __init__(self, seq: int, request_id: str):
        self.seq = seq
        self.request_id = request_id
        self.cancel_event = threading.Event()
        self.cancel_reason = ""

    def cancel(self, reason: str) -> None:
        if not self.cancel_event.is_set():
            self.cancel_reason = reason
            self.cancel_event.set()


class ServerGenerationCoordinator:
    """Single-flight GPU access: at most one generation runs and holds VRAM at any time.

    With ``preempt=True`` (default) a new request cancels the running one and every older queued
    one, then waits until the old generation has finished its ``finally`` cleanup before it
    allocates anything. This is what keeps "Stop + send again" from stacking two KV caches.
    """

    def __init__(self, preempt: bool = True):
        self.preempt = bool(preempt)
        self._state_lock = threading.Lock()
        self._gpu_lock = threading.Lock()
        self._seq = 0
        self._active: GenerationTicket | None = None
        self._waiting: dict[int, GenerationTicket] = {}
        self.preemptions = 0
        self.completed = 0

    def new_ticket(self, request_id: str) -> GenerationTicket:
        with self._state_lock:
            self._seq += 1
            ticket = GenerationTicket(self._seq, request_id)
            if self.preempt:
                for other in [self._active, *self._waiting.values()]:
                    if other is not None and not other.cancel_event.is_set():
                        other.cancel(f"preempted by newer request {request_id}")
                        self.preemptions += 1
            self._waiting[ticket.seq] = ticket
            return ticket

    @contextmanager
    def gpu(self, ticket: GenerationTicket):
        """Holds the GPU for ``ticket``; raises ``GenerationCancelled`` if it is cancelled while queued."""
        from .unified_runtime import GenerationCancelled

        try:
            while not self._gpu_lock.acquire(timeout=0.05):
                if ticket.cancel_event.is_set():
                    raise GenerationCancelled("queue", ticket.cancel_reason)
        finally:
            with self._state_lock:
                self._waiting.pop(ticket.seq, None)
        try:
            if ticket.cancel_event.is_set():
                raise GenerationCancelled("queue", ticket.cancel_reason)
            with self._state_lock:
                self._active = ticket
            yield ticket
        finally:
            with self._state_lock:
                if self._active is ticket:
                    self._active = None
                self.completed += 1
            self._gpu_lock.release()

    def stop_current(self, reason: str = "stop request") -> bool:
        with self._state_lock:
            targets = [t for t in [self._active, *self._waiting.values()] if t is not None]
        for t in targets:
            t.cancel(reason)
        return bool(targets)

    def snapshot(self) -> dict[str, Any]:
        with self._state_lock:
            return {
                "busy": self._active is not None,
                "active_request": self._active.request_id if self._active else None,
                "queued": len(self._waiting),
                "preemptions": self.preemptions,
                "finished": self.completed,
                "preempt_mode": self.preempt,
            }


class RequestError(Exception):
    def __init__(self, status: int, message: str, err_type: str = "invalid_request_error"):
        super().__init__(message)
        self.status = status
        self.message = message
        self.err_type = err_type


class AirColibriLANServer:
    """Self-contained LAN HTTP/SSE server + UDP LAN broadcaster wrapping `AirColibriUnifiedEngine`."""

    def __init__(
        self,
        engine: Any,
        host: str = "0.0.0.0",
        http_port: int = DEFAULT_HTTP_PORT,
        beacon_port: int = DEFAULT_BEACON_PORT,
        api_key: str = "colibri-local",
        require_auth: bool = False,
        enable_beacon: bool = True,
        max_request_bytes: int = DEFAULT_MAX_REQUEST_BYTES,
        log_file: str | Path | None = DEFAULT_LOG_FILE,
        preempt: bool = True,
        body_timeout_s: float = 30.0,
    ):
        self.engine = engine
        self.host = host
        self.http_port = int(http_port)
        self.beacon_port = int(beacon_port)
        self.api_key = api_key
        self.require_auth = bool(require_auth)
        self.enable_beacon = bool(enable_beacon)
        self.max_request_bytes = int(max_request_bytes)
        self.body_timeout_s = float(body_timeout_s)
        self.log_file = str(log_file) if log_file else None
        self.log = setup_server_logger(self.log_file)
        self.coordinator = ServerGenerationCoordinator(preempt=preempt)
        self.hostname = socket.gethostname()
        self.lan_ip, self.all_lan_ips = detect_lan_ips()
        self.started_at = int(time.time())
        self._shutdown = threading.Event()
        self._httpd: ThreadingHTTPServer | None = None
        self._http_thread: threading.Thread | None = None
        self.beacon: UDPLANBeacon | None = None

    def passport(self) -> dict[str, Any]:
        st = self.engine.status_snapshot()
        return {
            "service": "aircolibri",
            "version": "1.0.0",
            "hostname": self.hostname,
            "lan_ip": self.lan_ip,
            "all_lan_ips": self.all_lan_ips,
            "http_port": self.http_port,
            "beacon_port": self.beacon_port,
            "local_web_url": f"http://127.0.0.1:{self.http_port}",
            "lan_web_url": f"http://{self.lan_ip}:{self.http_port}",
            "openai_base_url": f"http://{self.lan_ip}:{self.http_port}/v1",
            "model_id": st["model_id"],
            "gpu_name": st["gpu_name"],
            "vram_mode": st["vram_mode"],
            "mtp_enabled": st["mtp_enabled"],
            "api_key": self.api_key if not self.require_auth else "***protected***",
            "require_auth": self.require_auth,
            "auth_required": self.require_auth,
            "started_at": self.started_at,
        }

    def full_passport_for_console(self) -> dict[str, Any]:
        p = self.passport()
        p["api_key"] = self.api_key
        return p

    def _build_handler(self):
        srv = self
        from .unified_runtime import GenerationCancelled

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt: str, *args: Any) -> None:
                srv.log.info("%s %s", self.address_string(), fmt % args)

            def log_error(self, fmt: str, *args: Any) -> None:
                srv.log.warning("%s %s", self.address_string(), fmt % args)

            def _send_cors(self) -> None:
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
                self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

            def _send_json(self, obj: dict[str, Any], status: int = HTTPStatus.OK, close: bool = False) -> None:
                raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
                self.send_response(status)
                self._send_cors()
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                if close:
                    self.send_header("Connection", "close")
                    self.close_connection = True
                self.end_headers()
                try:
                    self.wfile.write(raw)
                except OSError:
                    pass

            def _send_error(self, status: int, message: str, err_type: str = "invalid_request_error", close: bool = False) -> None:
                self._send_json({"error": {"message": message, "type": err_type}}, status, close=close)

            def _check_auth(self) -> bool:
                if not srv.require_auth:
                    return True
                auth = self.headers.get("Authorization", "")
                expected = f"Bearer {srv.api_key}".encode("utf-8")
                if hmac.compare_digest(auth.encode("utf-8", errors="replace"), expected):
                    return True
                self._send_error(HTTPStatus.UNAUTHORIZED, "Invalid Bearer API key", "auth_error", close=True)
                return False

            def _read_json_body(self) -> dict[str, Any]:
                te = (self.headers.get("Transfer-Encoding") or "").strip().lower()
                if te and te != "identity":
                    raise RequestError(HTTPStatus.LENGTH_REQUIRED, "Chunked request bodies are not supported; send Content-Length")
                raw_len = self.headers.get("Content-Length")
                if raw_len is None or raw_len.strip() == "":
                    return {}
                try:
                    length = int(raw_len)
                except ValueError:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length") from None
                if length < 0:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
                if length > srv.max_request_bytes:
                    raise RequestError(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"Request body {length} bytes exceeds the limit of {srv.max_request_bytes} bytes",
                    )
                if length == 0:
                    return {}
                self.connection.settimeout(srv.body_timeout_s)
                try:
                    raw = self.rfile.read(length)
                finally:
                    self.connection.settimeout(None)
                if len(raw) != length:
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Request body shorter than Content-Length")
                try:
                    body = json.loads(raw.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                    raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid JSON body") from None
                if not isinstance(body, dict):
                    raise RequestError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
                return body

            def do_OPTIONS(self) -> None:
                self.send_response(HTTPStatus.NO_CONTENT)
                self._send_cors()
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self) -> None:
                path = self.path.split("?", 1)[0]
                if path in ("/", "/index.html"):
                    raw = WEB_UI_HTML.encode("utf-8")
                    self.send_response(HTTPStatus.OK)
                    self._send_cors()
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(raw)))
                    self.end_headers()
                    self.wfile.write(raw)
                    return

                if path in ("/health", "/api/info"):
                    info = srv.passport()
                    info["engine"] = srv.engine.status_snapshot()
                    info["generation"] = srv.coordinator.snapshot()
                    info["max_request_bytes"] = srv.max_request_bytes
                    if srv.beacon is not None:
                        info["beacon_stats"] = {
                            "packets_sent": srv.beacon.packets_sent,
                            "discoveries_answered": srv.beacon.discoveries_answered,
                        }
                    self._send_json(info)
                    return

                if path == "/v1/models":
                    if not self._check_auth():
                        return
                    self._send_json(
                        {
                            "object": "list",
                            "data": [
                                {
                                    "id": srv.engine.model_id,
                                    "object": "model",
                                    "created": srv.started_at,
                                    "owned_by": "aircolibri-lan",
                                }
                            ],
                        }
                    )
                    return

                self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {path}", "not_found")

            def do_POST(self) -> None:
                path = self.path.split("?", 1)[0]
                if path not in ("/v1/chat/completions", "/v1/chat/stop", "/api/stop", "/api/reset"):
                    self._send_error(HTTPStatus.NOT_FOUND, f"Unknown endpoint {path}", "not_found", close=True)
                    return
                if not self._check_auth():
                    return
                try:
                    body = self._read_json_body()
                except RequestError as exc:
                    srv.log.warning("%s %s rejected: %s %s", self.address_string(), path, int(exc.status), exc.message)
                    self._send_error(exc.status, exc.message, exc.err_type, close=True)
                    return
                except (TimeoutError, OSError) as exc:
                    srv.log.warning("%s %s body read failed: %r", self.address_string(), path, exc)
                    self.close_connection = True
                    return

                if path in ("/v1/chat/stop", "/api/stop"):
                    stopped = srv.coordinator.stop_current("stop request from client")
                    srv.log.info("stop request from %s: stopped=%s", self.address_string(), stopped)
                    self._send_json({"ok": True, "stopped": stopped})
                    return

                if path == "/api/reset":
                    ticket = srv.coordinator.new_ticket("reset")
                    try:
                        with srv.coordinator.gpu(ticket):
                            srv.engine.reset_session()
                    except GenerationCancelled:
                        pass
                    self._send_json({"ok": True})
                    return

                self._chat_completions(body)

            def _chat_completions(self, body: dict[str, Any]) -> None:
                messages = body.get("messages")
                if not isinstance(messages, list) or not messages or not all(isinstance(m, dict) and "role" in m for m in messages):
                    self._send_error(HTTPStatus.BAD_REQUEST, "`messages` must be a non-empty array of {role, content}")
                    return
                for m in messages:
                    if isinstance(m.get("content"), list):
                        m["content"] = "".join(
                            str(part.get("text", "")) for part in m["content"] if isinstance(part, dict)
                        )
                try:
                    max_tokens = int(body.get("max_tokens") or body.get("max_completion_tokens") or 1024)
                except (TypeError, ValueError):
                    self._send_error(HTTPStatus.BAD_REQUEST, "`max_tokens` must be an integer")
                    return
                max_tokens = max(1, min(MAX_TOKENS_LIMIT, max_tokens))
                stream = bool(body.get("stream", False))
                enable_thinking = bool(body.get("enable_thinking", True))
                use_session = bool(body.get("use_session_cache", True))
                req_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
                created_ts = int(time.time())
                model_id = srv.engine.model_id
                probe = make_disconnect_probe(self.connection)
                ticket = srv.coordinator.new_ticket(req_id)
                t0 = time.perf_counter()
                srv.log.info(
                    "%s start %s stream=%s messages=%d max_tokens=%d", req_id, self.address_string(), stream,
                    len(messages), max_tokens,
                )

                def cancel_check() -> bool:
                    if ticket.cancel_event.is_set() or srv._shutdown.is_set():
                        return True
                    if probe():
                        ticket.cancel("client disconnected")
                        return True
                    return False

                def chunk(delta: dict[str, Any], finish: str | None = None, **extra: Any) -> dict[str, Any]:
                    obj = {
                        "id": req_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_id,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                    }
                    obj.update(extra)
                    return obj

                headers_sent = False

                def _write_sse(payload_obj: dict[str, Any]) -> bool:
                    line = f"data: {json.dumps(payload_obj, ensure_ascii=False)}\n\n".encode("utf-8")
                    try:
                        self.wfile.write(line)
                        self.wfile.flush()
                        return True
                    except OSError:
                        ticket.cancel("client disconnected (write failed)")
                        return False

                outcome = "ok"
                gen = None
                try:
                    with srv.coordinator.gpu(ticket):
                        gen = srv.engine.stream_chat(
                            messages,
                            max_new_tokens=max_tokens,
                            enable_thinking=enable_thinking,
                            use_session_cache=use_session,
                            stop_on_eos=True,
                            cancel_check=cancel_check,
                        )
                        if not stream:
                            done_ev = None
                            try:
                                for ev in gen:
                                    if ev["type"] == "done":
                                        done_ev = ev
                            finally:
                                gen.close()
                            assert done_ev is not None
                            m = done_ev["metrics"]
                            msg_out: dict[str, Any] = {"role": "assistant", "content": done_ev["answer_text"]}
                            if done_ev["thinking_text"]:
                                msg_out["reasoning_content"] = done_ev["thinking_text"]
                            self._send_json(
                                {
                                    "id": req_id,
                                    "object": "chat.completion",
                                    "created": created_ts,
                                    "model": model_id,
                                    "choices": [{"index": 0, "message": msg_out, "finish_reason": m["finish_reason"]}],
                                    "usage": {
                                        "prompt_tokens": m["prompt_tokens"],
                                        "completion_tokens": m["generated_tokens"],
                                        "total_tokens": m["prompt_tokens"] + m["generated_tokens"],
                                    },
                                    "aircolibri_metrics": m,
                                }
                            )
                            return

                        self.send_response(HTTPStatus.OK)
                        self._send_cors()
                        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
                        self.send_header("Cache-Control", "no-cache")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        self.close_connection = True
                        headers_sent = True
                        if not _write_sse(chunk({"role": "assistant"})):
                            return
                        try:
                            for ev in gen:
                                if ev["type"] == "delta":
                                    field = "reasoning_content" if ev["phase"] == "thinking" else "content"
                                    if not _write_sse(chunk({field: ev["text"]})):
                                        break
                                elif ev["type"] == "done":
                                    m = ev["metrics"]
                                    _write_sse(chunk({}, m["finish_reason"], aircolibri_metrics=m))
                        finally:
                            gen.close()
                        try:
                            self.wfile.write(b"data: [DONE]\n\n")
                            self.wfile.flush()
                        except OSError:
                            pass
                except GenerationCancelled as exc:
                    outcome = f"cancelled ({exc.phase}: {ticket.cancel_reason or exc.reason})"
                    if headers_sent:
                        if _write_sse(chunk({}, "stop", aircolibri_cancelled=True)):
                            try:
                                self.wfile.write(b"data: [DONE]\n\n")
                                self.wfile.flush()
                            except OSError:
                                pass
                    elif not probe():
                        self._send_error(499, "Generation cancelled: " + (ticket.cancel_reason or exc.reason), "cancelled", close=True)
                except Exception as exc:  # engine failures (CUDA OOM, ...) must never leave the server wedged
                    outcome = f"error: {exc!r}"
                    srv.log.error("%s failed: %r\n%s", req_id, exc, traceback.format_exc())
                    if headers_sent:
                        _write_sse({"error": {"message": str(exc), "type": "server_error"}})
                    else:
                        self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, f"Generation failed: {exc}", "server_error", close=True)
                finally:
                    if gen is not None:
                        gen.close()
                    m = (getattr(srv.engine, "last_metrics", {}) or {}) if outcome == "ok" else {}
                    srv.log.info(
                        "%s end %s in %.2fs | prompt=%s gen=%s prefill_ms=%s tok/s=%s peak_vram_mib=%s alloc_mib=%.1f",
                        req_id, outcome, time.perf_counter() - t0, m.get("prompt_tokens", "-"), m.get("generated_tokens", "-"),
                        m.get("prefill_ms", "-"), m.get("decode_tok_per_s", "-"), m.get("peak_vram_mib", "-"),
                        _allocated_mib(),
                    )

        return Handler

    def start_background(self) -> None:
        handler_cls = self._build_handler()
        self._httpd = ThreadingHTTPServer((self.host, self.http_port), handler_cls)
        self._httpd.daemon_threads = True
        self.http_port = int(self._httpd.server_address[1])
        if self.enable_beacon:
            self.beacon = UDPLANBeacon(
                payload_fn=self.passport,
                lan_ips=self.all_lan_ips,
                beacon_port=self.beacon_port,
            )
            self.beacon.start()
        self._http_thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.25}, name="aircolibri-lan-http", daemon=True
        )
        self._http_thread.start()
        self.log.info(
            "server started on %s:%s (log=%s, max_request_bytes=%s, auth=%s, preempt=%s)",
            self.host, self.http_port, self.log_file, self.max_request_bytes, self.require_auth, self.coordinator.preempt,
        )

    def serve_forever(self) -> None:
        self.start_background()
        print(format_connection_passport(self.full_passport_for_console()), flush=True)
        if self.log_file:
            print(f"  Server log: {Path(self.log_file).resolve()}", flush=True)
        try:
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            print("\n[AirColibri] Shutting down LAN server...", flush=True)
        finally:
            self.stop()

    def stop(self) -> None:
        self._shutdown.set()
        self.coordinator.stop_current("server shutdown")
        if self.beacon is not None:
            self.beacon.stop()
            self.beacon = None
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None
        if self._http_thread is not None:
            self._http_thread.join(timeout=2.0)
            self._http_thread = None
        self.log.info("server stopped")


def _allocated_mib() -> float:
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / (1024 ** 2)
    except Exception:
        pass
    return 0.0
