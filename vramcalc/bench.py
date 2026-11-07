# -*- coding: utf-8 -*-
"""Honest streaming benchmark.

The previous version of this tool counted Server-Sent-Event *chunks* and
reported them as tokens per second. A chunk can hold one token, half a token or
ten tokens, so that number was not a token rate.

This module therefore reports two clearly different things:

* ``tps`` - token throughput, available **only** when the server reports
  ``usage.completion_tokens`` (requested via ``stream_options.include_usage``).
* ``chunk_rate_per_s`` - the raw stream chunk rate, explicitly labelled as
  chunks/s and never presented as tokens/s.

When the server reports no usage, ``tps`` is ``None`` with a reason. No number
is invented.
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .errors import BenchmarkError

DEFAULT_PROMPT = "Explain how a KV cache works in three short bullet points."


def _iter_sse(resp, deadline_s: float, t_start: float):
    for raw in resp:
        if time.perf_counter() - t_start > deadline_s:
            raise BenchmarkError(f"benchmark exceeded {deadline_s}s deadline", code="benchmark_timeout")
        line = raw.decode("utf-8", errors="ignore").strip()
        if not line or not line.startswith("data:"):
            continue
        body = line[5:].strip()
        if body == "[DONE]":
            return
        try:
            yield json.loads(body)
        except json.JSONDecodeError:
            continue


def run_bench(
    base_url: str,
    model: str,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 120,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict:
    if not base_url or not base_url.lower().startswith(("http://", "https://")):
        raise BenchmarkError(
            f"bench needs an http(s) base URL, got {base_url!r}", code="invalid_url"
        )
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or not 1 <= max_tokens <= 8192:
        raise BenchmarkError(
            f"max tokens must be an integer in 1..8192, got {max_tokens!r}", code="invalid_max_tokens"
        )
    url = f"{base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.2,
        "stream": True,
        # ask the server to tell us the real token count instead of guessing it
        "stream_options": {"include_usage": True},
    }
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)

    t_start = time.perf_counter()
    first_chunk_t = None
    last_chunk_t = None
    chunks = 0
    text_parts: list = []
    usage: dict | None = None

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - user supplied URL
            for obj in _iter_sse(resp, timeout, t_start):
                t_now = time.perf_counter()
                if isinstance(obj.get("usage"), dict):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}
                piece = delta.get("content")
                if piece:
                    chunks += 1
                    if first_chunk_t is None:
                        first_chunk_t = t_now
                    last_chunk_t = t_now
                    text_parts.append(piece)
    except urllib.error.HTTPError as exc:
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
        except Exception:
            pass
        raise BenchmarkError(
            f"endpoint returned HTTP {exc.code}: {detail or exc.reason}",
            code="benchmark_http_error",
            details={"status": exc.code},
        ) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BenchmarkError(f"could not reach {url}: {exc}", code="benchmark_unreachable") from exc

    t_end = time.perf_counter()
    emission_window = (last_chunk_t - first_chunk_t) if (first_chunk_t and last_chunk_t) else 0.0
    completion_tokens = None
    if usage:
        raw = usage.get("completion_tokens")
        if isinstance(raw, int) and not isinstance(raw, bool) and raw > 0:
            completion_tokens = raw

    tps = None
    tps_basis = None
    tps_reason = None
    if completion_tokens is not None and emission_window > 0.001:
        tps = round((completion_tokens - 1) / emission_window, 2)
        tps_basis = "provider_reported_usage.completion_tokens over the decode window"
    elif completion_tokens is None:
        tps_reason = (
            "server did not report usage.completion_tokens; stream chunks are not tokens, so no "
            "tokens/s value is claimed"
        )
    else:
        tps_reason = "generation finished in under 1 ms of decode window; sample too short to time"

    chunk_rate = round(chunks / emission_window, 2) if chunks > 1 and emission_window > 0.001 else None

    return {
        "model": model,
        "url": url,
        "streaming": True,
        "ttft_ms": round((first_chunk_t - t_start) * 1000, 1) if first_chunk_t else None,
        "total_time_s": round(t_end - t_start, 3),
        "stream_chunks": chunks,
        "completion_tokens": completion_tokens,
        "tps": tps,
        "tps_basis": tps_basis,
        "tps_unavailable_reason": tps_reason,
        "chunk_rate_per_s": chunk_rate,
        "chunk_rate_label": "stream chunks/s - NOT tokens/s",
        "usage": usage,
        "sample_output": ("".join(text_parts)[:160] + "...") if text_parts else "",
        "warnings": [
            "Chunks in an SSE stream are not tokens; chunk_rate_per_s must not be quoted as tokens/s.",
            "Network latency, server queueing and provider-side batching are not controlled here; "
            "compare runs only under identical conditions.",
        ],
    }
