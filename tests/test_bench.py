# -*- coding: utf-8 -*-
"""Benchmark honesty: the network probe is DISABLED in this release.

The previous implementation streamed from an arbitrary user-supplied URL with
``urllib`` (following redirects, forwarding a bearer token), computed tokens/s
from ``(N-1)/first_to_last_chunk`` and treated incomplete SSE as success. The
release review scoped bench out of 2.0.0, so the module must now refuse with a
structured ``benchmark_disabled`` error and must not be able to touch the
network at all.

These tests prove the *absence* of a probe:

* the CLI exits 2 with a machine-readable ``benchmark_disabled`` code;
* a live local server observes ZERO requests (no network reachable);
* the module carries no HTTP/socket transport and cannot be re-enabled by
  accident without failing here.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from vramcalc.bench import DEFAULT_PROMPT, run_bench
from vramcalc.errors import BenchmarkError

ROOT = pathlib.Path(__file__).resolve().parents[1]


class _CountingHandler(BaseHTTPRequestHandler):
    requests = 0

    def log_message(self, *args):  # noqa: A002 - silence the test output
        return

    def do_POST(self):  # noqa: N802
        _CountingHandler.requests += 1
        self.send_response(200)
        self.end_headers()

    def do_GET(self):  # noqa: N802
        _CountingHandler.requests += 1
        self.send_response(200)
        self.end_headers()


@pytest.fixture()
def server():
    _CountingHandler.requests = 0
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _CountingHandler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/v1"
    finally:
        httpd.shutdown()
        httpd.server_close()


def run(*args):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, "-m", "vramcalc", *args],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_run_bench_refuses_offline_without_touching_the_network():
    with pytest.raises(BenchmarkError) as exc:
        run_bench("http://127.0.0.1:9/v1", "m")
    assert exc.value.code == "benchmark_disabled"
    assert "disabled" in exc.value.message.lower()
    # a migration command must be offered instead of a probe
    assert exc.value.details.get("migration")


def test_run_bench_still_accepts_the_legacy_signature_for_migration():
    # tolerated old invocation shapes must raise benchmark_disabled, not TypeError
    with pytest.raises(BenchmarkError) as exc:
        run_bench(base_url="http://127.0.0.1:8000/v1", model="local-model", prompt=DEFAULT_PROMPT, max_tokens=120)
    assert exc.value.code == "benchmark_disabled"


def test_bench_cli_receives_zero_requests_and_exits_two(server):
    res = run("bench", "--url", server, "--model", "local-model", "--json")
    assert res.returncode == 2, res.stdout + res.stderr
    payload = json.loads(res.stdout)
    assert payload["error"] == "benchmark_disabled"
    assert payload["details"].get("migration")
    # the decisive evidence: the endpoint saw nothing at all
    assert _CountingHandler.requests == 0, f"bench reached the network: {_CountingHandler.requests} requests"


def test_bench_cli_text_error_names_the_migration():
    res = run("bench", "--url", "http://127.0.0.1:8000/v1", "--model", "m")
    assert res.returncode == 2
    assert "benchmark_disabled" in res.stderr
    assert "--model" in res.stderr


def test_bench_module_has_no_network_transport():
    import vramcalc.bench as bench

    for forbidden in ("urllib", "socket", "http", "requests"):
        assert not hasattr(bench, forbidden), f"bench must not expose {forbidden}"
    source = (ROOT / "vramcalc" / "bench.py").read_text(encoding="utf-8")
    for token in ("urlopen", "urllib.request", "socket.socket", "http.client"):
        assert token not in source, f"bench source still carries transport: {token}"


def test_bench_help_says_the_probe_is_disabled():
    res = run("bench", "--help")
    assert res.returncode == 0
    assert "disabled" in res.stdout.lower()
