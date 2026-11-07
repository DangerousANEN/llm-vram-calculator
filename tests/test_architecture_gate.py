# -*- coding: utf-8 -*-
"""Architecture gate + KV geometry.

Every rejection here is a promise: the tool refuses instead of guessing.
"""
from __future__ import annotations

import copy

import pytest

from vramcalc.core import (
    MAX_BATCH,
    geometry_from_config,
    kv_bytes_per_token,
    kv_cache_bytes,
)
from vramcalc.errors import InvalidParameter, UnsupportedArchitecture, UnsupportedFeature

DENSE = {
    "model_type": "qwen2",
    "architectures": ["Qwen2ForCausalLM"],
    "num_hidden_layers": 28,
    "num_attention_heads": 28,
    "num_key_value_heads": 4,
    "hidden_size": 3584,
    "max_position_embeddings": 32768,
}


def test_geometry_from_real_qwen25_config(qwen25):
    geom = geometry_from_config(qwen25["config"])
    assert geom.num_hidden_layers == 28
    assert geom.num_attention_heads == 28
    assert geom.num_key_value_heads == 4
    assert geom.head_dim == 128
    assert geom.head_dim_source == "config.hidden_size / config.num_attention_heads"
    assert geom.kv_width == 512


def test_geometry_uses_explicit_head_dim_for_qwen3(qwen3):
    geom = geometry_from_config(qwen3["config"])
    assert geom.head_dim == 128
    assert geom.head_dim_source == "config.head_dim"
    assert geom.num_hidden_layers == 36
    assert geom.num_key_value_heads == 8


def test_kv_bytes_per_token_is_key_plus_value(qwen25, qwen3):
    # 2 * 28 layers * 4 kv heads * 128 = 28672 elements * 2 bytes
    assert kv_bytes_per_token(geometry_from_config(qwen25["config"])) == 57344
    # 2 * 36 * 8 * 128 * 2
    assert kv_bytes_per_token(geometry_from_config(qwen3["config"])) == 147456


def test_kv_cache_scales_with_context_and_batch(qwen25):
    geom = geometry_from_config(qwen25["config"])
    one = kv_cache_bytes(geom, 8192, 1)
    assert one == 57344 * 8192 == 469_762_048
    assert kv_cache_bytes(geom, 8192, 4) == one * 4
    assert kv_cache_bytes(geom, 4096, 1) == one // 2


def test_bf16_and_fp16_kv_have_identical_size(qwen25):
    geom = geometry_from_config(qwen25["config"])
    assert kv_bytes_per_token(geom, "fp16") == kv_bytes_per_token(geom, "bf16")


@pytest.mark.parametrize("dtype", ["fp8", "q8_0", "int8", "", "fp32"])
def test_quantized_or_unknown_kv_dtype_is_refused(qwen25, dtype):
    geom = geometry_from_config(qwen25["config"])
    with pytest.raises(UnsupportedFeature) as exc:
        kv_bytes_per_token(geom, dtype)
    assert exc.value.code == "unsupported_kv_dtype"
    assert "fp16" in exc.value.message


def _cfg(**over):
    cfg = copy.deepcopy(DENSE)
    cfg.update(over)
    return cfg


def test_sliding_window_enabled_is_refused():
    with pytest.raises(UnsupportedArchitecture) as exc:
        geometry_from_config(_cfg(use_sliding_window=True, sliding_window=4096))
    assert exc.value.code == "sliding_window_not_supported"


def test_sliding_window_key_present_but_disabled_is_accepted(qwen25):
    # Qwen2.5 config carries sliding_window=131072 with use_sliding_window=false
    assert qwen25["config"]["sliding_window"] == 131072
    assert qwen25["config"]["use_sliding_window"] is False
    assert geometry_from_config(qwen25["config"]).num_hidden_layers == 28


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_experts", 8),
        ("num_local_experts", 64),
        ("n_routed_experts", 256),
        ("moe_intermediate_size", 1408),
        ("num_experts_per_tok", 2),
    ],
)
def test_moe_configs_are_refused(field, value):
    with pytest.raises(UnsupportedArchitecture) as exc:
        geometry_from_config(_cfg(**{field: value}))
    assert exc.value.code == "moe_not_supported"
    assert field in exc.value.details.get("field", "")


@pytest.mark.parametrize("field", ["kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim"])
def test_mla_configs_are_refused(field):
    with pytest.raises(UnsupportedArchitecture) as exc:
        geometry_from_config(_cfg(**{field: 512}))
    assert exc.value.code == "mla_not_supported"


def test_hybrid_layer_types_are_refused():
    with pytest.raises(UnsupportedArchitecture) as exc:
        geometry_from_config(_cfg(layer_types=["full_attention", "linear_attention"]))
    assert exc.value.code == "hybrid_attention_not_supported"


@pytest.mark.parametrize("model_type", ["llama", "deepseek_v3", "gpt_oss", "mamba", ""])
def test_unknown_architecture_is_refused(model_type):
    with pytest.raises(UnsupportedArchitecture) as exc:
        geometry_from_config(_cfg(model_type=model_type))
    assert exc.value.code == "unknown_architecture"
    assert "supported" in exc.value.message.lower()


def test_supported_model_type_with_unknown_class_is_refused():
    with pytest.raises(UnsupportedArchitecture) as exc:
        geometry_from_config(_cfg(architectures=["Qwen2MoeForCausalLM"]))
    assert exc.value.code == "unknown_architecture"


def test_missing_config_is_refused():
    with pytest.raises(UnsupportedArchitecture):
        geometry_from_config({})


def test_head_dim_must_be_derivable():
    with pytest.raises(UnsupportedArchitecture) as exc:
        geometry_from_config(_cfg(hidden_size=3585))
    assert exc.value.code == "head_dim_not_derivable"


def test_kv_heads_cannot_exceed_attention_heads():
    with pytest.raises(UnsupportedArchitecture) as exc:
        geometry_from_config(_cfg(num_key_value_heads=64))
    assert exc.value.code == "invalid_config"


def test_missing_kv_heads_falls_back_to_mha():
    cfg = _cfg()
    del cfg["num_key_value_heads"]
    geom = geometry_from_config(cfg)
    assert geom.num_key_value_heads == geom.num_attention_heads


def test_bound_constants_are_sane():
    from vramcalc.core import MAX_CONTEXT_TOKENS, MAX_POSITIONS

    assert MAX_CONTEXT_TOKENS >= 131072
    assert MAX_BATCH >= 64
    assert MAX_POSITIONS >= MAX_CONTEXT_TOKENS