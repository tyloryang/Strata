#!/usr/bin/env python3
"""Authenticated API and NPU metrics gateway for the Strata CANN engine."""

import collections
import hmac
import ipaddress
import json
import os
import re
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

UPSTREAM = os.environ.get("STRATA_API_UPSTREAM", "http://127.0.0.1:18790")
CLIENT_IP = ipaddress.ip_address(os.environ["STRATA_API_CLIENT_IP"])
CERT = os.environ["STRATA_API_CERT"]
TLS_KEY = os.environ["STRATA_API_KEY_FILE"]
TOKEN_FILE = os.environ["STRATA_API_TOKEN_FILE"]
MONITOR_HTML = os.environ.get("STRATA_API_MONITOR_HTML", "/data/strata-npu-vector-v4-20261106/monitor.html")
MODEL_NAME = "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS"
ENGINE_START = time.time()


def _npu_smi_snapshot():
    try:
        proc = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=4, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return [], str(exc)
    if proc.returncode != 0:
        return [], proc.stderr.strip() or f"npu-smi exited {proc.returncode}"

    devices = {}
    lines = proc.stdout.splitlines()
    header = re.compile(r"^\|\s*(\d+)\s+(910\S+)\s+\|\s*([^|]+)\|\s*([\d.]+)\s+([\d.]+)")
    data = re.compile(r"^\|\s*(\d+)\s+\|\s*([0-9A-Fa-f:.]+)\s*\|\s*([^|]+)\|")
    process = re.compile(r"^\|\s*(\d+)\s+\d+\s*\|\s*\d+\s*\|\s*([^|]+)\|\s*(\d+)\s*\|")

    for i, line in enumerate(lines):
        match = header.match(line)
        if not match:
            continue
        device_id = int(match.group(1))
        item = {
            "device": device_id,
            "name": match.group(2),
            "health": match.group(3).strip(),
            "power_w": float(match.group(4)),
            "temperature_c": float(match.group(5)),
            "aicore_percent": None,
            "hbm_used_mb": None,
            "hbm_total_mb": None,
            "strata_process_mb": 0,
        }
        for candidate in lines[i + 1:i + 5]:
            row = data.match(candidate)
            if row:
                values = re.findall(r"\d+(?:\.\d+)?", row.group(3))
                if len(values) >= 5:
                    item["aicore_percent"] = float(values[0])
                    item["hbm_used_mb"] = int(float(values[-2]))
                    item["hbm_total_mb"] = int(float(values[-1]))
                break
        devices[device_id] = item

    for line in lines:
        match = process.match(line)
        if match and "model_seq" in match.group(2):
            device_id = int(match.group(1))
            if device_id in devices:
                devices[device_id]["strata_process_mb"] += int(match.group(3))

    for item in devices.values():
        used, total = item["hbm_used_mb"], item["hbm_total_mb"]
        item["hbm_percent"] = round(100 * used / total, 2) if used is not None and total else None
    return [devices[k] for k in sorted(devices)], None if devices else "npu-smi returned no Ascend device rows"


def _engine_pid():
    try:
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            try:
                argv = open(f"/proc/{entry}/cmdline", "rb").read().split(b"\0")
            except OSError:
                continue
            if argv and os.path.basename(argv[0].decode("utf-8", "replace")) == "model_seq-grouped":
                return int(entry)
    except OSError:
        pass
    return None


class HardwareSampler:
    def __init__(self):
        self.lock = threading.Lock()
        self.at = 0.0
        self.devices = []
        self.error = "sampling has not run"

    def snapshot(self):
        with self.lock:
            now = time.monotonic()
            if now - self.at >= 1.0:
                self.devices, self.error = _npu_smi_snapshot()
                self.at = now
            return self.devices, self.error


class RequestMetrics:
    def __init__(self):
        self.lock = threading.Lock()
        self.hardware = HardwareSampler()
        self.total = self.successes = self.errors = 0
        self.active = self.prompt_tokens = self.completion_tokens = 0
        self.total_wall_s = 0.0
        self.recent = collections.deque(maxlen=50)

    def begin(self):
        with self.lock:
            self.active += 1

    def finish(self, path, status, wall_s, body):
        try:
            payload = json.loads(body) if body else {}
        except (ValueError, TypeError):
            payload = {}
        usage = payload.get("usage") or {}
        prompt = int(usage.get("prompt_tokens", 0) or 0)
        completion = int(usage.get("completion_tokens", 0) or 0)
        strata = payload.get("strata") or {}
        service_s = strata.get("seconds")
        denominator = float(service_s) if isinstance(service_s, (float, int)) and service_s > 0 else wall_s
        row = {
            "id": payload.get("id", ""),
            "path": path,
            "time": time.time(),
            "status": int(status),
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "wall_s": round(wall_s, 3),
            "end_to_end_tok_s": round(completion / denominator, 2) if denominator > 0 and completion else None,
        }
        with self.lock:
            self.active = max(0, self.active - 1)
            self.total += 1
            self.successes += int(200 <= status < 300)
            self.errors += int(status < 200 or status >= 300)
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_wall_s += wall_s
            self.recent.appendleft(row)

    def snapshot(self):
        devices, device_error = self.hardware.snapshot()
        with self.lock:
            req = {
                "total": self.total,
                "successes": self.successes,
                "errors": self.errors,
                "active": self.active,
                "prompt_tokens": self.prompt_tokens,
                "completion_tokens": self.completion_tokens,
                "average_wall_s": round(self.total_wall_s / self.total, 3) if self.total else None,
                "recent": list(self.recent),
            }
        pid = _engine_pid()
        return {
            "timestamp": time.time(),
            "engine": "strata-cann",
            "model": MODEL_NAME,
            "engine_state": "generating" if req["active"] else "loaded / idle" if pid else "not loaded",
            "engine_pid": pid,
            "uptime_s": round(time.time() - ENGINE_START, 1),
            "requests": req,
            "devices": devices,
            "device_error": device_error,
        }


METRICS = RequestMetrics()


def _prometheus(snapshot):
    lines = [
        "# HELP strata_requests_total Inference requests seen by the CANN gateway.",
        "# TYPE strata_requests_total counter",
        f"strata_requests_total {snapshot['requests']['total']}",
        "# HELP strata_request_errors_total Failed inference requests.",
        "# TYPE strata_request_errors_total counter",
        f"strata_request_errors_total {snapshot['requests']['errors']}",
        "# HELP strata_requests_active Current inference requests in flight.",
        "# TYPE strata_requests_active gauge",
        f"strata_requests_active {snapshot['requests']['active']}",
        "# HELP strata_prompt_tokens_total Prompt tokens served since gateway start.",
        "# TYPE strata_prompt_tokens_total counter",
        f"strata_prompt_tokens_total {snapshot['requests']['prompt_tokens']}",
        "# HELP strata_completion_tokens_total Completion tokens served since gateway start.",
        "# TYPE strata_completion_tokens_total counter",
        f"strata_completion_tokens_total {snapshot['requests']['completion_tokens']}",
    ]
    for d in snapshot["devices"]:
        label = f'device="{d["device"]}"'
        for name, help_text, value in (
            ("strata_npu_aicore_utilization_percent", "Ascend AICore utilization percent.", d["aicore_percent"]),
            ("strata_npu_hbm_used_bytes", "Ascend HBM used bytes.", None if d["hbm_used_mb"] is None else d["hbm_used_mb"] * 1048576),
            ("strata_npu_hbm_total_bytes", "Ascend HBM total bytes.", None if d["hbm_total_mb"] is None else d["hbm_total_mb"] * 1048576),
            ("strata_npu_temperature_celsius", "Ascend device temperature Celsius.", d["temperature_c"]),
            ("strata_npu_power_watts", "Ascend device power watts.", d["power_w"]),
            ("strata_npu_strata_process_memory_bytes", "HBM used by the Strata model process.", d["strata_process_mb"] * 1048576),
        ):
            if value is not None:
                lines.extend((f"# HELP {name} {help_text}", f"# TYPE {name} gauge", f"{name}{{{label}}} {value}"))
    return "\n".join(lines) + "\n"


class Gateway(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):
        print("%s %s" % (self.address_string(), fmt % args), flush=True)

    def _reply(self, status, body, content_type="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _source_allowed(self):
        allowed_ip = self.server.allowed_client_ip
        if allowed_ip is None:
            return True
        try:
            allowed = ipaddress.ip_address(self.client_address[0]) == allowed_ip
        except ValueError:
            allowed = False
        if not allowed:
            self._reply(403, b'{"error":{"message":"source IP is not allowed"}}')
        return allowed

    def _authorized(self):
        try:
            with open(TOKEN_FILE, "r", encoding="ascii") as token_file:
                expected = "Bearer " + token_file.read().strip()
        except OSError:
            self._reply(503, b'{"error":{"message":"API authentication is unavailable"}}')
            return False
        if not hmac.compare_digest(self.headers.get("Authorization", ""), expected):
            self._reply(401, b'{"error":{"message":"invalid API key"}}')
            return False
        return self._source_allowed()

    def _monitor_page(self):
        if not self._source_allowed():
            return
        try:
            with open(MONITOR_HTML, "rb") as page:
                self._reply(200, page.read(), "text/html; charset=utf-8")
        except OSError:
            self._reply(503, b"Monitor page is not installed", "text/plain; charset=utf-8")

    def _metrics(self):
        if not self._authorized():
            return
        snapshot = METRICS.snapshot()
        query = parse_qs(urlsplit(self.path).query)
        accept = self.headers.get("Accept", "")
        prometheus = urlsplit(self.path).path.endswith("/prometheus") or query.get("format") == ["prometheus"] or "text/plain" in accept
        if prometheus:
            self._reply(200, _prometheus(snapshot).encode(), "text/plain; version=0.0.4; charset=utf-8")
        else:
            self._reply(200, json.dumps(snapshot, separators=(",", ":")).encode(), "application/json; charset=utf-8")

    def _forward(self):
        if not self._authorized():
            return
        length = int(self.headers.get("Content-Length", "0"))
        if length > 1_048_576:
            self._reply(413, b'{"error":{"message":"request body too large"}}')
            return
        body = self.rfile.read(length) if length else None
        path = urlsplit(self.path).path
        counted = path in ("/v1/completions", "/v1/chat/completions")
        if counted:
            METRICS.begin()
        started = time.perf_counter()
        status, response_body, content_type = 502, b"", "application/json"
        request = Request(UPSTREAM + self.path, data=body,
                          headers={"Content-Type": self.headers.get("Content-Type", "application/json")},
                          method=self.command)
        try:
            with urlopen(request, timeout=180) as response:
                status, response_body = response.status, response.read()
                content_type = response.headers.get("Content-Type", content_type)
        except HTTPError as exc:
            status, response_body = exc.code, exc.read()
            content_type = exc.headers.get("Content-Type", content_type)
        except (URLError, TimeoutError) as exc:
            response_body = json.dumps({"error": {"message": str(exc)}}).encode()
        finally:
            if counted:
                METRICS.finish(path, status, time.perf_counter() - started, response_body)
        self._reply(status, response_body, content_type)

    def do_GET(self):
        path = urlsplit(self.path).path
        if path in ("/", "/monitor", "/monitor/"):
            self._monitor_page()
        elif path in ("/metrics", "/metrics/prometheus"):
            self._metrics()
        elif path in ("/health", "/health/", "/v1/models"):
            self._forward()
        else:
            self._reply(404, b'{"error":{"message":"not found"}}')

    def do_POST(self):
        if urlsplit(self.path).path not in ("/v1/completions", "/v1/chat/completions"):
            self._reply(404, b'{"error":{"message":"not found"}}')
            return
        self._forward()


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 32

    def __init__(self, address, handler, allowed_client_ip=None):
        self.allowed_client_ip = allowed_client_ip
        super().__init__(address, handler)


internal = GatewayServer(("127.0.0.1", 18791), Gateway)
threading.Thread(target=internal.serve_forever, daemon=True).start()
server = GatewayServer(("0.0.0.0", 443), Gateway, CLIENT_IP)
context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
context.minimum_version = ssl.TLSVersion.TLSv1_2
context.load_cert_chain(CERT, TLS_KEY)
server.socket = context.wrap_socket(server.socket, server_side=True, do_handshake_on_connect=False)
print("monitor: /monitor; metrics: /metrics and /metrics/prometheus; bearer auth enabled", flush=True)
server.serve_forever()
