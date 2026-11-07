# -*- coding: utf-8 -*-
"""Ledger arithmetic, units and honesty guarantees."""
from __future__ import annotations

import pytest

from vramcalc.catalog import get_model
from vramcalc.core import GIB, compute_ledger
from vramcalc.errors import InvalidParameter, UnknownModel, UnsupportedQuant

# Byte sizes verified against the official Hugging Face file listings at the
# pinned revisions (see scripts/fetch_model_metadata.py).
QWEN25_Q4_K_M_BYTES = 4_683_073_632
QWEN25_BF16_BYTES = 15_231_271_888
QWEN3_BF16_BYTES = 16_381_516_776


def test_qwen25_q4km_8192_ledger_is_exact(qwen25):
    led = compute_ledger(qwen25, "q4_k_m", 8192)
    d = led.to_dict()
    assert led.weights_bytes == QWEN25_Q4_K_M_BYTES
    assert led.weights_estimated is False
    assert led.kv_bytes_per_token == 57344
    assert led.kv_bytes_total == 469_762_048
    assert d["subtotal"]["bytes"] == QWEN25_Q4_K_M_BYTES + 469_762_048
    assert d["ledger_total"]["bytes"] == d["subtotal"]["bytes"]
    assert d["weights"]["gib"] == round(QWEN25_Q4_K_M_BYTES / GIB, 4)
    assert d["unit"].startswith("GiB")


def test_runtime_overhead_is_unknown_not_invented(qwen25):
    d = compute_ledger(qwen25, "q4_k_m", 8192).to_dict()
    oh = d["runtime_overhead"]
    # unknown must be explicit and must NOT masquerade as a measured zero
    assert oh["bytes"] is None
    assert oh["gib"] is None
    assert oh["unknown"] is True
    assert oh["kind"] == "unknown_not_estimated"
    assert oh["label"] is None
    assert "NOT estimated" in oh["note"]
    # the subtotal stays useful and the total still equals weights+KV
    assert d["subtotal"]["bytes"] > 0
    assert d["subtotal"]["bytes"] == d["weights"]["bytes"] + d["kv_cache"]["bytes"]
    assert d["ledger_total"]["bytes"] == d["subtotal"]["bytes"]
    assert d["ledger_total"]["unknown_overhead_excluded"] is True
    assert d["ledger_total"]["includes_manual_overhead_only"] is True


def test_unknown_overhead_is_not_confused_with_an_explicit_zero_bound(qwen25):
    implicit = compute_ledger(qwen25, "q4_k_m", 8192).to_dict()
    explicit = compute_ledger(qwen25, "q4_k_m", 8192, manual_overhead_gib=0).to_dict()
    assert implicit["runtime_overhead"]["unknown"] is True
    assert implicit["runtime_overhead"]["bytes"] is None
    assert explicit["runtime_overhead"]["unknown"] is False
    assert explicit["runtime_overhead"]["bytes"] == 0
    assert explicit["runtime_overhead"]["gib"] == 0.0
    assert explicit["runtime_overhead"]["kind"] == "manual_bound"
    assert "manual bound" in explicit["runtime_overhead"]["label"].lower()


def test_manual_overhead_is_labelled_and_added(qwen25):
    led = compute_ledger(qwen25, "q4_k_m", 8192, manual_overhead_gib=1.5)
    d = led.to_dict()
    assert d["runtime_overhead"]["bytes"] == int(1.5 * GIB) == 1_610_612_736
    assert d["runtime_overhead"]["kind"] == "manual_bound"
    assert d["runtime_overhead"]["unknown"] is False
    assert "manual bound" in d["runtime_overhead"]["label"]
    assert d["ledger_total"]["bytes"] == d["subtotal"]["bytes"] + 1_610_612_736
    assert d["ledger_total"]["unknown_overhead_excluded"] is False


def test_never_claims_measured_vram_or_fit(qwen25):
    d = compute_ledger(qwen25, "q4_k_m", 8192).to_dict()
    assert d["ledger_total"]["is_measured_vram"] is False
    assert d["ledger_total"]["is_fit_guarantee"] is False
    assert d["verification"]["gpu_runtime_checked"] is False
    assert "not a measured" in d["verification"]["statement"].lower()


def test_qwen3_bf16_32k_matches_config_geometry(qwen3):
    led = compute_ledger(qwen3, "bf16", 32768)
    assert led.weights_bytes == QWEN3_BF16_BYTES
    assert led.kv_bytes_per_token == 147456
    assert led.kv_bytes_total == 147456 * 32768 == 4_831_838_208
    assert led.geometry.head_dim == 128
    assert "147456" in " ".join(led.to_dict()["notes"])


def test_batch_multiplies_only_kv(qwen25):
    one = compute_ledger(qwen25, "q4_k_m", 4096, batch=1)
    four = compute_ledger(qwen25, "q4_k_m", 4096, batch=4)
    assert four.weights_bytes == one.weights_bytes
    assert four.kv_bytes_total == one.kv_bytes_total * 4
    assert four.subtotal_bytes == one.weights_bytes + one.kv_bytes_total * 4


def test_weights_do_not_depend_on_context(qwen25):
    a = compute_ledger(qwen25, "q8_0", 1024)
    b = compute_ledger(qwen25, "q8_0", 32768)
    assert a.weights_bytes == b.weights_bytes == qwen25["weights"]["q8_0"]["bytes"]


@pytest.mark.parametrize("ctx", [0, -1, 1.5, "8192", None, True])
def test_invalid_context_is_refused(qwen25, ctx):
    with pytest.raises(InvalidParameter):
        compute_ledger(qwen25, "q4_k_m", ctx)


@pytest.mark.parametrize("batch", [0, -3, 4097, 2.5, None])
def test_invalid_batch_is_refused(qwen25, batch):
    with pytest.raises(InvalidParameter):
        compute_ledger(qwen25, "q4_k_m", 8192, batch=batch)


def test_context_above_model_max_is_refused(qwen25):
    with pytest.raises(InvalidParameter) as exc:
        compute_ledger(qwen25, "q4_k_m", 32769)
    assert exc.value.code == "context_exceeds_model_max"
    assert "YaRN" in exc.value.message or "rope" in exc.value.message.lower()


def test_context_equal_to_model_max_is_accepted(qwen25):
    assert compute_ledger(qwen25, "q4_k_m", 32768).context == 32768


def test_absurd_context_batch_product_is_refused(qwen25):
    with pytest.raises(InvalidParameter) as exc:
        compute_ledger(qwen25, "q4_k_m", 32768, batch=4096)
    assert exc.value.code == "positions_too_large"


@pytest.mark.parametrize("overhead", [float("nan"), float("inf"), -1, 5000, "1.5"])
def test_invalid_manual_overhead_is_refused(qwen25, overhead):
    with pytest.raises(InvalidParameter):
        compute_ledger(qwen25, "q4_k_m", 8192, manual_overhead_gib=overhead)


def test_unknown_quant_lists_published_ones(qwen3):
    with pytest.raises(UnsupportedQuant) as exc:
        compute_ledger(qwen3, "q2_k", 8192)
    assert exc.value.code == "unsupported_quant"
    assert "q4_k_m" in exc.value.message
    assert "allow-estimate" in exc.value.message
    assert exc.value.details["published"] == [
        "bf16",
        "fp16",
        "q8_0",
        "q6_k",
        "q5_k_m",
        "q5_0",
        "q4_k_m",
    ]


def test_estimate_is_opt_in_and_labelled(qwen3):
    led = compute_ledger(qwen3, "q2_k", 8192, allow_estimate=True)
    d = led.to_dict()
    assert led.weights_estimated is True
    assert d["weights"]["source"] == "estimated_bytes_per_param"
    assert any("ESTIMATE" in n for n in d["notes"])
    assert d["weights"]["bytes"] > 0


def test_unknown_model_is_refused():
    with pytest.raises(UnknownModel) as exc:
        get_model("llama-3-70b")
    assert "qwen2.5-7b-instruct" in exc.value.message


def test_limits_are_present_and_specific(qwen25):
    d = compute_ledger(qwen25, "q4_k_m", 8192).to_dict()
    text = " ".join(d["limits"]).lower()
    for word in ("overhead", "fit guarantee", "sliding", "fp16", "offload", "max_position_embeddings"):
        assert word in text
    assert len(d["limits"]) >= 6


def test_notes_explain_head_dim_and_kv_math(qwen25):
    d = compute_ledger(qwen25, "q4_k_m", 8192).to_dict()
    joined = " ".join(d["notes"])
    assert "head_dim" in joined
    assert "57344" in joined


def test_source_is_carried_into_result(qwen25):
    d = compute_ledger(qwen25, "q4_k_m", 8192).to_dict()
    assert d["source"]["repo"] == "Qwen/Qwen2.5-7B-Instruct"
    assert d["source"]["revision"] == qwen25["source"]["revision"]
    assert d["source"]["revision_url"].endswith(qwen25["source"]["revision"])
    assert d["source"]["license"] == "apache-2.0"