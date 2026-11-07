# -*- coding: utf-8 -*-
"""Streaming benchmark - TEMPORARILY DISABLED in this release.

The 2.0.0 release is a config-grounded *calculator*. The old streaming probe was
removed after a release-safety review found it was not safe enough to ship:

* it streamed from an arbitrary user-supplied URL with ``urllib`` and followed
  redirects while forwarding the ``Authorization`` header to the redirect target;
* it reported tokens/s as ``(completion_tokens - 1) / (last_chunk - first_chunk)``,
  which is not a token rate and ignores the real request/decode window;
* a truncated or malformed SSE stream (no ``[DONE]``, no usage) still returned
  "success", and the overall deadline only checked between chunks.

A correct bounded local probe needs a redesign (pinned endpoint, no redirect
following, explicit stream terminator, wall-clock deadline), which is out of
scope here. This module therefore contains **no network transport at all**:
``run_bench`` refuses immediately with a structured ``benchmark_disabled`` error
and a migration hint, so an old invocation fails loudly instead of measuring
anything.

Nothing in this module claims a working tokens/s figure.
"""
from __future__ import annotations

from .errors import BenchmarkError

#: Kept so the CLI parser still accepts ``--prompt`` with its old default.
DEFAULT_PROMPT = "Explain how a KV cache works in three short bullet points."

DISABLED_MESSAGE = (
    "bench is temporarily disabled in this release: the old streaming probe was removed after a "
    "release-safety review (redirect/auth forwarding, chunk-vs-token maths, incomplete-SSE success "
    "and a weak overall deadline). No network request is made. Use the server's own metrics for "
    "throughput, or run the memory ledger instead: "
    "`python -m vram_calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192`."
)

#: Printed with the refusal so existing scripts have a concrete next step.
MIGRATION = "python -m vram_calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192"


def run_bench(
    base_url: str | None = None,
    model: str | None = None,
    prompt: str = DEFAULT_PROMPT,
    max_tokens: int = 120,
    api_key: str | None = None,
    timeout: float = 120.0,
) -> dict:
    """Always refuse. Kept only so legacy invocations get a migration hint.

    The signature is intentionally the old one: a caller that still passes
    ``base_url``/``model``/``max_tokens`` gets ``benchmark_disabled`` instead of a
    ``TypeError``, which is the useful migration signal.
    """
    raise BenchmarkError(
        DISABLED_MESSAGE,
        code="benchmark_disabled",
        details={
            "migration": MIGRATION,
            "removed_in": "2.0.0",
            "reason": "no safe bounded probe in this release; a redesign is required",
        },
    )
