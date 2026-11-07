# -*- coding: utf-8 -*-
"""Fixture integrity: the shipped catalogue must be re-verifiable against the
official metadata it claims to come from."""
from __future__ import annotations

import re

import pytest

from vramcalc.catalog import (
    _approx_parameters,
    estimated_quants,
    get_model,
    list_models,
    published_quants,
)
from vramcalc.core import geometry_from_config, kv_bytes_per_token

HEX40 = re.compile(r"^[0-9a-f]{40}$")
HEX64 = re.compile(r"^[0-9a-f]{64}$")

# Sizes read from huggingface.co/api/models/.../tree/<revision> on 2026-11-07.
EXPECTED_WEIGHTS = {
    "qwen2.5-7b-instruct": {
        "bf16": 15_231_271_888,
        "fp16": 15_231_271_888,
        "q4_k_m": 4_683_073_632,
        "q5_k_m": 5_444_831_360,
        "q6_k": 6_254_198_880,
        "q8_0": 8_098_525_408,
        "q3_k_m": 3_808_391_072,
        "q2_k": 3_015_940_000,
        "q4_0": 4_431_390_848,
        "q5_0": 5_315_176_576,
    },
    "qwen3-8b": {
        "bf16": 16_381_516_776,
        "fp16": 16_381_516_776,
        "q4_k_m": 5_027_783_488,
        "q5_k_m": 5_851_112_224,
        "q6_k": 6_725_899_040,
        "q8_0": 8_709_518_112,
        "q5_0": 5_720_761_152,
    },
}


def test_catalogue_has_the_two_required_models(catalog_raw):
    assert [m["id"] for m in catalog_raw["models"]] == ["qwen2.5-7b-instruct", "qwen3-8b"]
    assert catalog_raw["schema_version"] == 2
    assert catalog_raw["data_source"] == "huggingface_hub_public_api"


@pytest.mark.parametrize("model_id", ["qwen2.5-7b-instruct", "qwen3-8b"])
def test_every_fixture_is_revision_pinned(model_id):
    m = get_model(model_id)
    assert HEX40.match(m["source"]["revision"]), "revision must be a 40-char commit sha"
    assert HEX40.match(m["source"]["gguf_revision"])
    assert m["source"]["revision"] in m["source"]["revision_url"]
    assert m["source"]["revision"] in m["source"]["config_url"]
    assert HEX64.match(m["source"]["config_sha256"])
    assert m["license"] == "apache-2.0"


@pytest.mark.parametrize("model_id", EXPECTED_WEIGHTS)
def test_weight_bytes_match_official_file_listings(model_id):
    m = get_model(model_id)
    for quant, expected in EXPECTED_WEIGHTS[model_id].items():
        assert m["weights"][quant]["bytes"] == expected, f"{model_id} {quant}"


@pytest.mark.parametrize("model_id", ["qwen2.5-7b-instruct", "qwen3-8b"])
def test_weight_entries_are_internally_consistent(model_id):
    m = get_model(model_id)
    for quant, entry in m["weights"].items():
        assert entry["bytes"] > 0
        assert entry["files"], f"{quant} must list the source files"
        assert sum(f["bytes"] for f in entry["files"]) == entry["bytes"], quant
        if quant.startswith("q"):
            assert entry["source"] == "official_gguf"
            assert HEX40.match(entry["gguf_revision"])
            assert all(f["path"].endswith(".gguf") for f in entry["files"])
        else:
            assert entry["source"].startswith("official_safetensors")
            assert all(f["path"].endswith(".safetensors") for f in entry["files"])


@pytest.mark.parametrize("model_id", ["qwen2.5-7b-instruct", "qwen3-8b"])
def test_fp16_and_bf16_are_the_same_bytes(model_id):
    m = get_model(model_id)
    assert m["weights"]["fp16"]["bytes"] == m["weights"]["bf16"]["bytes"]


def test_quantised_weights_are_smaller_than_bf16(qwen25, qwen3):
    for m in (qwen25, qwen3):
        for q in ("q8_0", "q6_k", "q5_k_m", "q4_k_m"):
            assert m["weights"][q]["bytes"] < m["weights"]["bf16"]["bytes"], (m["id"], q)


@pytest.mark.parametrize("model_id", ["qwen2.5-7b-instruct", "qwen3-8b"])
def test_config_geometry_is_dense_and_derivable(model_id):
    m = get_model(model_id)
    geom = geometry_from_config(m["config"])
    assert geom.num_hidden_layers > 0
    assert geom.num_key_value_heads < geom.num_attention_heads  # GQA
    assert kv_bytes_per_token(geom) == 2 * geom.num_hidden_layers * geom.num_key_value_heads * geom.head_dim * 2
    for moe_key in ("num_experts", "num_local_experts", "n_routed_experts"):
        assert moe_key not in m["config"]
    assert m["config"]["use_sliding_window"] is False


def test_published_quants_are_exact_only(qwen25, qwen3):
    for m in (qwen25, qwen3):
        for q in published_quants(m):
            assert q in m["weights"]
            assert m["weights"][q]["source"] != "estimated_bytes_per_param"
        # estimates must never shadow a published quant
        for q in estimated_quants(m):
            assert q not in m["weights"]


def test_approximate_parameter_count_is_in_a_sane_range(qwen25, qwen3):
    # used only for labelled estimates, sanity check vs the family names
    assert 6.5e9 < _approx_parameters(qwen25) < 8.5e9
    assert 7.0e9 < _approx_parameters(qwen3) < 9.5e9


def test_model_lookup_accepts_alias_and_is_case_insensitive():
    assert get_model("Qwen3-8B")["id"] == "qwen3-8b"
    assert get_model("  qwen2.5-7b-instruct ")["id"] == "qwen2.5-7b-instruct"


def test_catalog_is_offline(qwen25):
    # no URL fields are allowed to be fetched at runtime; only links for humans
    assert qwen25["source"]["revision_url"].startswith("https://huggingface.co/")
    assert len(list_models()) == 2