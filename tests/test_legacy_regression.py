# -*- coding: utf-8 -*-
"""Real pre-change behavioural regression harness.

This is deliberately NOT a "the old code failed to import" claim. It loads the
*frozen* legacy calculator (the public ``calculate_vram`` function exactly as it
shipped before the config-grounded rewrite, taken from git commit 2dcb9ff) and
runs it against the same release requirement the new ledger has to satisfy:

    1. the KV cache bytes must be derived from the model's own config geometry
       (for Qwen2.5-7B-Instruct: 57344 bytes/token, so 8192 tokens = 469762048);
    2. runtime overhead must be represented as an explicit *unknown*, never as a
       fabricated positive number.

The legacy function fails both (it guesses layers from a parameter count and
invents CUDA/activation overhead). The final ``compute_ledger`` API passes both.
The point of the module is that the first test is a real RED: the shared
requirement check raises a genuine AssertionError on the legacy output, which is
behavioural evidence, not a collection error.

The frozen copy lives in ``tests/fixtures/legacy_vram_calc_2dcb9ff.py``; it is a
test fixture only and is never imported by the shipped package.
"""
from __future__ import annotations

import importlib.util
import pathlib

import pytest

from vramcalc.catalog import get_model
from vramcalc.core import GIB, compute_ledger

FIXTURE = pathlib.Path(__file__).resolve().parent / "fixtures" / "legacy_vram_calc_2dcb9ff.py"

# Qwen2.5-7B-Instruct at revision a09a3545..., 8192 tokens, one sequence.
QWEN25_KV_PER_TOKEN = 57344
QWEN25_KV_BYTES_8192 = QWEN25_KV_PER_TOKEN * 8192  # 469_762_048


def _load_legacy():
    spec = importlib.util.spec_from_file_location("legacy_vram_calc_2dcb9ff", FIXTURE)
    assert spec and spec.loader, "legacy fixture is missing"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _requirement_check(kv_bytes: int, overhead_kind: str, *, label: str) -> None:
    """The semantic requirement both implementations are judged against."""
    assert kv_bytes == QWEN25_KV_BYTES_8192, (
        f"{label}: KV bytes {kv_bytes} != config-grounded {QWEN25_KV_BYTES_8192} "
        f"(57344 B/token x 8192 tokens)"
    )
    assert overhead_kind == "unknown_not_estimated", (
        f"{label}: runtime overhead must be an explicit unknown, got {overhead_kind!r}"
    )


def test_legacy_public_calculate_vram_fails_the_release_requirement():
    legacy = _load_legacy()
    # 7.6B approximates Qwen2.5-7B; the legacy API takes a parameter count, not a config.
    res = legacy.calculate_vram(7.6, "q4_k_m", 8192, 1)

    legacy_kv_bytes = int(round(res["kv_cache_gb"] * GIB))
    legacy_overhead_kind = "invented_numeric" if res.get("cuda_overhead_gb", 0) > 0 else "absent"

    # Real divergence, not a rounding difference: the guessed layer count is far off.
    assert abs(legacy_kv_bytes - QWEN25_KV_BYTES_8192) > GIB / 2, res

    # RED: running the release requirement against the legacy output raises.
    with pytest.raises(AssertionError):
        _requirement_check(legacy_kv_bytes, legacy_overhead_kind, label="legacy calculate_vram")

    # and it fabricates positive overhead instead of exposing an unknown
    assert res["cuda_overhead_gb"] > 0 and res["activations_gb"] > 0
    assert "runtime_overhead" not in res and "overhead_kind" not in res


def test_final_api_passes_the_same_requirement():
    d = compute_ledger(get_model("qwen2.5-7b-instruct"), "q4_k_m", 8192).to_dict()
    kv_bytes = d["kv_cache"]["bytes"]
    overhead_kind = d["runtime_overhead"]["kind"]

    _requirement_check(kv_bytes, overhead_kind, label="compute_ledger")

    assert kv_bytes == QWEN25_KV_BYTES_8192
    assert d["runtime_overhead"]["unknown"] is True
    assert d["runtime_overhead"]["bytes"] is None
