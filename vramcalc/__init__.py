# -*- coding: utf-8 -*-
"""Config-grounded memory ledger for dense LLM inference.

Public API::

    from vramcalc import get_model, compute_ledger
    ledger = compute_ledger(get_model("qwen2.5-7b-instruct"), "q4_k_m", 8192, 1)
    ledger.to_dict()["ledger_total"]["gib"]
"""
from __future__ import annotations

__version__ = "2.1.1"

from .catalog import get_model, list_models, published_quants  # noqa: E402
from .core import (  # noqa: E402
    Geometry,
    Ledger,
    compute_ledger,
    geometry_from_config,
    kv_bytes_per_token,
    kv_cache_bytes,
)
from .errors import (  # noqa: E402
    CalcError,
    InvalidParameter,
    UnknownModel,
    UnsupportedArchitecture,
    UnsupportedFeature,
    UnsupportedQuant,
)

__all__ = [
    "__version__",
    "CalcError",
    "Geometry",
    "InvalidParameter",
    "Ledger",
    "UnknownModel",
    "UnsupportedArchitecture",
    "UnsupportedFeature",
    "UnsupportedQuant",
    "compute_ledger",
    "get_model",
    "geometry_from_config",
    "kv_bytes_per_token",
    "kv_cache_bytes",
    "list_models",
    "published_quants",
]