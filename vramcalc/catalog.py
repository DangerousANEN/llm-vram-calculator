# -*- coding: utf-8 -*-
"""Access to the revision-pinned model catalogue.

The catalogue ships with the package (``vramcalc/data/models.json``) and is
built from official Hugging Face metadata by
``scripts/fetch_model_metadata.py``. Nothing here touches the network.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from .errors import UnknownModel, UnsupportedQuant

DATA_FILE = Path(__file__).resolve().parent / "data" / "models.json"


@lru_cache(maxsize=1)
def load_catalog() -> dict:
    with DATA_FILE.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    if data.get("schema_version") != 2:
        raise RuntimeError(f"unsupported catalogue schema: {data.get('schema_version')!r}")
    return data


def list_models() -> list:
    return list(load_catalog()["models"])


def get_model(model_id: str) -> dict:
    wanted = (model_id or "").strip().lower()
    for m in list_models():
        if m["id"].lower() == wanted:
            return m
        for alias in (m.get("short_name", ""), m.get("display_name", "")):
            if alias and alias.strip().lower() == wanted:
                return m
    raise UnknownModel(
        f"model {model_id!r} is not in the supported catalogue. Supported: "
        + ", ".join(m["id"] for m in list_models())
        + ". Unsupported families are refused instead of approximated.",
        code="unknown_model",
        details={"requested": model_id, "supported": [m["id"] for m in list_models()]},
    )


def published_quants(model: dict) -> list:
    """Quantisations with an exact official file listing, best first."""
    order = ["bf16", "fp16", "q8_0", "q6_k", "q5_k_m", "q5_0", "q4_k_m", "q4_0", "q3_k_m", "q2_k"]
    have = [q for q in order if q in model.get("weights", {})]
    extra = sorted(q for q in model.get("weights", {}) if q not in order)
    return have + extra


def estimated_quants(model: dict) -> list:
    return sorted((model.get("estimates") or {}).get("bytes_per_param", {}))


def model_quant_entry(model: dict, quant: str, allow_estimate: bool = False) -> tuple:
    """Return ``(entry, estimated)`` for a quantisation, or refuse."""
    q = (quant or "").strip().lower()
    weights = model.get("weights", {})
    if q in weights:
        return weights[q], False
    if allow_estimate:
        est = (model.get("estimates") or {}).get("bytes_per_param", {})
        if q in est:
            params = model.get("approx_parameters") or _approx_parameters(model)
            entry = {
                "bytes": int(round(est[q] * params)),
                "source": "estimated_bytes_per_param",
                "detail": (
                    f"estimate: {est[q]} bytes/parameter x {params} parameters; this quantisation has no "
                    "official file listing for this repository, so the number is approximate"
                ),
            }
            return entry, True
    raise UnsupportedQuant(
        f"quant {quant!r} has no exact official file listing for {model['id']}. "
        f"Published (exact): {', '.join(published_quants(model))}. "
        f"Add --allow-estimate to accept a labelled approximation"
        + (f" for: {', '.join(estimated_quants(model))}." if estimated_quants(model) else "."),
        code="unsupported_quant",
        details={
            "requested": quant,
            "published": published_quants(model),
            "estimable": estimated_quants(model),
        },
    )


def _approx_parameters(model: dict) -> int:
    """Parameter count for estimate arithmetic only (never used for KV geometry)."""
    cfg = model.get("config", {})
    hidden = int(cfg.get("hidden_size") or 0)
    layers = int(cfg.get("num_hidden_layers") or 0)
    inter = int(cfg.get("intermediate_size") or 0)
    vocab = int(cfg.get("vocab_size") or 0)
    heads = int(cfg.get("num_attention_heads") or 0)
    kv_heads = int(cfg.get("num_key_value_heads") or heads or 0)
    head_dim = int(cfg.get("head_dim") or (hidden // heads if heads else 0))
    attn = hidden * head_dim * (heads + 2 * kv_heads)
    mlp = 3 * hidden * inter
    per_layer = attn + mlp
    embeds = 2 * vocab * hidden
    return int(round(layers * per_layer + embeds))


def corpus_facts() -> dict:
    cat = load_catalog()
    return {
        "models": [m["id"] for m in cat["models"]],
        "generated_at": cat.get("generated_at"),
        "data_source": cat.get("data_source"),
        "notes": cat.get("notes", []),
    }
