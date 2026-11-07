# -*- coding: utf-8 -*-
"""Benchmark honesty: chunks are not tokens.

A local fake OpenAI-compatible server streams SSE so the tests never touch a
real provider and never invent throughput numbers.
"""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vramcalc.bench import run_bench
from vramcalc.errors import BenchmarkError


class _Handler(BaseHTTPRequestHandler):
    mode = "usage"  # "usage" | "no_usage" | "http_500"
    seen_payload = {}

    def log_message(self, format, *args):  # noqa: A002 - silence the test output
        return

    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        _Handler.seen_payload = body
        if _Handler.mode == "http_500":
            payload = b'{"error": {"message": "boom"}}'
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        assert body.get("stream") is True
        assert body.get("stream_options", {}).get("include_usage") is True, "must request real token usage"

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # a real server has a non-zero inter-token interval; without it a
        # sub-millisecond window would be honestly reported as unmeasurable
        for i in range(5):
            chunk = {"choices": [{"delta": {"content": f"tok{i} "}}]}
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
            time.sleep(0.005)
        final = {"choices": [], "usage": {"prompt_tokens": 11, "completion_tokens": 5, "total_tokens": 16}}
        if _Handler.mode == "no_usage":
            final.pop("usage")
        self.wfile.write(f"data: {json.dumps(final)}\n\n".encode())
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def test_usage_reported_means_tokens_per_second(server):
    _Handler.mode = "usage"
    res = run_bench(server, "test-model", max_tokens=16)
    assert res["completion_tokens"] == 5
    assert res["stream_chunks"] == 5
    assert res["tps"] is not None and res["tps"] > 0
    assert res["tps_basis"].startswith("provider_reported_usage")
    assert "NOT tokens/s" in res["chunk_rate_label"]


def test_missing_usage_never_fabricates_tps(server):
    _Handler.mode = "no_usage"
    res = run_bench(server, "test-model", max_tokens=16)
    assert res["tps"] is None
    assert res["completion_tokens"] is None
    assert "not tokens" in res["tps_unavailable_reason"]
    assert res["stream_chunks"] == 5
    assert res["chunk_rate_per_s"] is not None
    assert "NOT tokens/s" in res["chunk_rate_label"]


def test_request_asks_for_usage_and_streaming(server):
    _Handler.mode = "usage"
    run_bench(server, "test-model")
    assert _Handler.seen_payload["stream"] is True
    assert _Handler.seen_payload["stream_options"] == {"include_usage": True}


def test_http_error_is_reported_not_swallowed(server):
    _Handler.mode = "http_500"
    with pytest.raises(BenchmarkError) as exc:
        run_bench(server, "test-model")
    assert exc.value.code == "benchmark_http_error"


@pytest.mark.parametrize("url", ["", "ftp://x", "example.com", None])
def test_invalid_url_is_refused(url):
    with pytest.raises(BenchmarkError) as exc:
        run_bench(url, "m")
    assert exc.value.code == "invalid_url"


@pytest.mark.parametrize("tokens", [0, -1, 100000, "10", 1.5])
def test_invalid_max_tokens_is_refused(server, tokens):
    with pytest.raises(BenchmarkError) as exc:
        run_bench(server, "m", max_tokens=tokens)
    assert exc.value.code == "invalid_max_tokens"


def test_unreachable_host_is_reported():
    with pytest.raises(BenchmarkError) as exc:
        run_bench("http://127.0.0.1:9/v1", "m", timeout=2)
    assert exc.value.code in {"benchmark_unreachable", "benchmark_http_error"}


def test_sample_output_is_truncated(server):
    _Handler.mode = "usage"
    res = run_bench(server, "m", max_tokens=8)
    assert len(res["sample_output"]) <= 165
    assert res["sample_output"].startswith("tok0")