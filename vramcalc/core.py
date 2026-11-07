# -*- coding: utf-8 -*-
"""Config-grounded memory ledger for **dense** transformer inference.

Design rules (deliberate, not stylistic):

* Geometry comes from the model's own ``config.json`` (layers, KV heads,
  head_dim). It is never inferred from a parameter count.
* Weight footprint is the **exact byte size** of an official file listing at a
  pinned revision, or an explicitly labelled estimate. MoE parameter counts are
  never treated as resident dense weights.
* The KV cache is derived per token from key+value geometry and multiplied by
  the allocated context and the sequence (batch) count.
* Runtime overhead is **unknown**. It is never invented. The caller may add a
  manual, labelled bound; the resulting "ledger total" is not measured VRAM and
  is not a fit guarantee.
* Anything the formula cannot model (MoE, MLA, sliding window, hybrid attention,
  quantized KV cache, partial CPU offload) is refused loudly.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import cast

from .errors import InvalidParameter, UnsupportedArchitecture, UnsupportedFeature, UnsupportedQuant

GIB = 1024 ** 3

#: Only these ``model_type`` values are accepted. A new family needs a reviewed
#: fixture, not a silent guess from a similar config.
SUPPORTED_MODEL_TYPES = frozenset({"qwen2", "qwen3"})
SUPPORTED_ARCHITECTURES = frozenset({"Qwen2ForCausalLM", "Qwen3ForCausalLM"})

#: KV cache dtypes this release actually models. Both are 2 bytes per element.
KV_DTYPE_BYTES = {"fp16": 2, "bf16": 2}

#: Hard sanity bounds so absurd input cannot be reported as a precise answer.
MAX_CONTEXT_TOKENS = 1_048_576
MAX_BATCH = 4096
MAX_POSITIONS = 33_554_432  # context * batch ceiling
MAX_KV_BYTES = 1 << 40  # 1 TiB

#: Keys that mark an architecture the dense KV formula does not describe.
MOE_KEYS = (
    "num_experts",
    "num_local_experts",
    "n_routed_experts",
    "n_shared_experts",
    "moe_intermediate_size",
    "num_experts_per_tok",
    "expert_count",
)
MLA_KEYS = ("kv_lora_rank", "qk_nope_head_dim", "qk_rope_head_dim", "mla_latent_dim")
HYBRID_KEYS = ("layer_types", "attn_layer_types", "linear_attention", "n_linear_attn_layers")


def _is_positive_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _positive_int(cfg: dict, key: str, default: object = None) -> int:
    """Fetch a positive int from a config or refuse with a stable code."""
    value = cfg.get(key, default)
    if not _is_positive_int(value):
        raise UnsupportedArchitecture(
            f"config field {key!r} is missing or not a positive integer ({value!r})",
            code="invalid_config",
            details={"field": key},
        )
    return cast(int, value)


def _normalise_kv_dtype(kv_dtype: str | None) -> str:
    # None means "use the default"; an explicit empty/blank value is an error, not a default
    if kv_dtype is None:
        return "fp16"
    dtype = str(kv_dtype).strip().lower()
    if dtype not in KV_DTYPE_BYTES:
        raise UnsupportedFeature(
            f"KV cache dtype {kv_dtype!r} is not modelled. This release only computes "
            f"fp16/bf16 KV cache; quantized KV cache (fp8/q8/q4) has no exact-cache "
            f"claim here and is refused rather than guessed.",
            code="unsupported_kv_dtype",
            details={"requested": kv_dtype, "supported": sorted(KV_DTYPE_BYTES)},
        )
    return dtype


@dataclass(frozen=True)
class Geometry:
    """Attention geometry taken from the model config."""

    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    hidden_size: int
    head_dim_source: str

    @property
    def kv_width(self) -> int:
        return self.num_key_value_heads * self.head_dim

    def to_dict(self) -> dict:
        return {
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "num_key_value_heads": self.num_key_value_heads,
            "head_dim": self.head_dim,
            "hidden_size": self.hidden_size,
            "head_dim_source": self.head_dim_source,
        }


def geometry_from_config(cfg: dict) -> Geometry:
    """Validate a config and extract attention geometry, or refuse."""
    if not isinstance(cfg, dict) or not cfg:
        raise UnsupportedArchitecture(
            "model config is empty or malformed; refusing to guess attention geometry",
            code="invalid_config",
        )

    model_type = cfg.get("model_type")
    archs = cfg.get("architectures") or []
    if not isinstance(archs, list):
        archs = [archs]
    archs = [a for a in archs if isinstance(a, str)]

    if model_type not in SUPPORTED_MODEL_TYPES:
        raise UnsupportedArchitecture(
            f"model_type {model_type!r} is not a supported dense architecture in this release "
            f"(supported: {', '.join(sorted(SUPPORTED_MODEL_TYPES))}). Refusing a precise answer.",
            code="unknown_architecture",
            details={"model_type": model_type, "supported_model_types": sorted(SUPPORTED_MODEL_TYPES)},
        )
    if not any(a in SUPPORTED_ARCHITECTURES for a in archs):
        raise UnsupportedArchitecture(
            f"architectures {archs!r} are not supported (expected one of "
            f"{', '.join(sorted(SUPPORTED_ARCHITECTURES))}). Refusing a precise answer.",
            code="unknown_architecture",
            details={"architectures": archs, "supported_architectures": sorted(SUPPORTED_ARCHITECTURES)},
        )

    for key in MOE_KEYS:
        if key in cfg:
            raise UnsupportedArchitecture(
                f"config declares mixture-of-experts field {key!r}: a dense weight ledger would be "
                "wrong (active experts are not resident full weights). Refusing a precise answer.",
                code="moe_not_supported",
                details={"field": key},
            )
    for key in MLA_KEYS:
        if key in cfg:
            raise UnsupportedArchitecture(
                f"config declares Multi-head Latent Attention field {key!r}; the dense key+value "
                "KV formula does not apply. Refusing a precise answer.",
                code="mla_not_supported",
                details={"field": key},
            )
    for key in HYBRID_KEYS:
        if key in cfg and cfg[key]:
            raise UnsupportedArchitecture(
                f"config declares per-layer / hybrid attention field {key!r}; KV growth is not uniform "
                "across layers. Refusing a precise answer.",
                code="hybrid_attention_not_supported",
                details={"field": key},
            )

    sliding = cfg.get("sliding_window")
    if cfg.get("use_sliding_window") and sliding:
        raise UnsupportedArchitecture(
            "config enables a sliding attention window; the cache stops growing past the window and "
            "the uniform KV formula would overstate memory. Refusing a precise answer.",
            code="sliding_window_not_supported",
            details={"sliding_window": sliding},
        )

    layers = _positive_int(cfg, "num_hidden_layers")
    heads = _positive_int(cfg, "num_attention_heads")
    hidden = _positive_int(cfg, "hidden_size")

    kv_heads = cfg.get("num_key_value_heads", heads)
    if not _is_positive_int(kv_heads):
        raise UnsupportedArchitecture("config has an unusable num_key_value_heads", code="invalid_config")
    kv_heads = cast(int, kv_heads)
    if kv_heads > heads:
        raise UnsupportedArchitecture(
            f"num_key_value_heads ({kv_heads}) exceeds num_attention_heads ({heads})",
            code="invalid_config",
        )

    raw_head_dim = cfg.get("head_dim")
    if _is_positive_int(raw_head_dim):
        head_dim = cast(int, raw_head_dim)
        source = "config.head_dim"
    else:
        if hidden % heads != 0:
            raise UnsupportedArchitecture(
                f"hidden_size {hidden} is not divisible by num_attention_heads {heads}; head_dim cannot be "
                "derived and a partial dimension would not be a real tensor shape. Refusing a precise answer.",
                code="head_dim_not_derivable",
            )
        head_dim = hidden // heads
        source = "config.hidden_size / config.num_attention_heads"

    return Geometry(
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv_heads,
        head_dim=head_dim,
        hidden_size=hidden,
        head_dim_source=source,
    )


def kv_bytes_per_token(geom: Geometry, kv_dtype: str = "fp16") -> int:
    """Key+value bytes for ONE token of ONE sequence.

    ``2`` = one key tensor and one value tensor, each
    ``num_key_value_heads * head_dim`` wide at ``bytes_per_element``.
    """
    dtype = _normalise_kv_dtype(kv_dtype)
    return 2 * geom.num_hidden_layers * geom.num_key_value_heads * geom.head_dim * KV_DTYPE_BYTES[dtype]


def kv_cache_bytes(geom: Geometry, context: int, batch: int, kv_dtype: str = "fp16") -> int:
    return kv_bytes_per_token(geom, kv_dtype) * context * batch


def _check_int(name: str, value: object, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise InvalidParameter(
            f"{name} must be an integer number of tokens/sequences, got {value!r}",
            code=f"invalid_{name}",
        )
    if value < 1:
        raise InvalidParameter(f"{name} must be >= 1, got {value}", code=f"invalid_{name}")
    if value > maximum:
        raise InvalidParameter(
            f"{name}={value} exceeds the supported bound ({maximum}); refusing an absurd ledger",
            code=f"{name}_too_large",
            details={name: value, "maximum": maximum},
        )
    return value


@dataclass
class Ledger:
    """A transparent memory ledger. ``total`` is a ledger sum, not measured VRAM."""

    model_id: str
    display_name: str
    quant: str
    kv_dtype: str
    context: int
    batch: int
    geometry: Geometry
    weights_bytes: int
    weights_source: str
    weights_detail: str
    weights_estimated: bool
    kv_bytes_per_token: int
    kv_bytes_total: int
    manual_overhead_bytes: int
    manual_overhead_label: str | None
    source: dict
    max_context: int
    limits: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def subtotal_bytes(self) -> int:
        return self.weights_bytes + self.kv_bytes_total

    @property
    def ledger_total_bytes(self) -> int:
        return self.subtotal_bytes + self.manual_overhead_bytes

    def to_dict(self) -> dict:
        def gib(n: int) -> float:
            return round(n / GIB, 4)

        return {
            "model_id": self.model_id,
            "display_name": self.display_name,
            "quant": self.quant,
            "kv_dtype": self.kv_dtype,
            "context": self.context,
            "batch": self.batch,
            "unit": "GiB (1 GiB = 1024^3 bytes); bytes are exact too",
            "geometry": self.geometry.to_dict(),
            "weights": {
                "bytes": self.weights_bytes,
                "gib": gib(self.weights_bytes),
                "source": self.weights_source,
                "detail": self.weights_detail,
                "estimated": self.weights_estimated,
            },
            "kv_cache": {
                "bytes_per_token": self.kv_bytes_per_token,
                "bytes": self.kv_bytes_total,
                "gib": gib(self.kv_bytes_total),
                "formula": "2 (key+value) * layers * num_key_value_heads * head_dim * bytes_per_element * context * batch",
            },
            "runtime_overhead": {
                "bytes": self.manual_overhead_bytes,
                "gib": gib(self.manual_overhead_bytes),
                "label": self.manual_overhead_label,
                "kind": "manual_bound" if self.manual_overhead_bytes else "unknown_not_estimated",
                "note": (
                    "CUDA context, allocator fragmentation, activations, graph/workspace buffers and "
                    "tokenizer/server overhead are runtime dependent and are NOT estimated by this tool. "
                    "Add a manual bound if you have measured one."
                ),
            },
            "subtotal": {"bytes": self.subtotal_bytes, "gib": gib(self.subtotal_bytes)},
            "ledger_total": {
                "bytes": self.ledger_total_bytes,
                "gib": gib(self.ledger_total_bytes),
                "is_measured_vram": False,
                "is_fit_guarantee": False,
            },
            "source": self.source,
            "max_context": self.max_context,
            "limits": self.limits,
            "notes": self.notes,
            "verification": {
                "gpu_runtime_checked": False,
                "statement": (
                    "No GPU/runtime measurement was performed for this ledger; it is a byte-level "
                    "accounting model, not a measured VRAM report or a guaranteed fit."
                ),
            },
        }


def compute_ledger(
    model: dict,
    quant: str,
    context: int,
    batch: int = 1,
    kv_dtype: str = "fp16",
    manual_overhead_gib: float | None = None,
    allow_estimate: bool = False,
) -> Ledger:
    """Build a ledger for one catalog model entry."""
    from .catalog import model_quant_entry  # local import: catalog owns the data

    context = _check_int("context", context, MAX_CONTEXT_TOKENS)
    batch = _check_int("batch", batch, MAX_BATCH)
    if context * batch > MAX_POSITIONS:
        raise InvalidParameter(
            f"context*batch = {context * batch} exceeds the supported bound ({MAX_POSITIONS}); "
            "refusing to report an absurd KV allocation",
            code="positions_too_large",
            details={"context": context, "batch": batch, "maximum": MAX_POSITIONS},
        )

    cfg = model.get("config") or {}
    geom = geometry_from_config(cfg)
    kv_dtype = _normalise_kv_dtype(kv_dtype)

    max_ctx_cfg = cfg.get("max_position_embeddings")
    max_ctx = cast(int, max_ctx_cfg) if _is_positive_int(max_ctx_cfg) else MAX_CONTEXT_TOKENS
    if context > max_ctx:
        raise InvalidParameter(
            f"context={context} exceeds this config's max_position_embeddings={max_ctx}. "
            "Long-context extrapolation (YaRN/rope scaling) changes the cache shape and is not modelled.",
            code="context_exceeds_model_max",
            details={"context": context, "max_position_embeddings": max_ctx},
        )

    entry, estimated = model_quant_entry(model, quant, allow_estimate=allow_estimate)

    kv_bpt = kv_bytes_per_token(geom, kv_dtype)
    kv_total = kv_bpt * context * batch
    if kv_total > MAX_KV_BYTES:
        raise InvalidParameter(
            f"derived KV cache {kv_total} bytes exceeds the supported bound ({MAX_KV_BYTES} bytes)",
            code="kv_too_large",
        )

    if manual_overhead_gib is None:
        overhead_bytes = 0
        overhead_label = None
    else:
        if isinstance(manual_overhead_gib, bool) or not isinstance(manual_overhead_gib, (int, float)):
            raise InvalidParameter("manual overhead must be a number of GiB", code="invalid_overhead")
        if not math.isfinite(float(manual_overhead_gib)):
            raise InvalidParameter("manual overhead must be finite", code="nonfinite_overhead")
        if manual_overhead_gib < 0 or manual_overhead_gib > 4096:
            raise InvalidParameter(
                f"manual overhead {manual_overhead_gib} GiB is outside the accepted range 0..4096",
                code="invalid_overhead",
            )
        overhead_bytes = int(round(float(manual_overhead_gib) * GIB))
        overhead_label = f"manual bound provided by the user: {manual_overhead_gib} GiB"

    limits = [
        "Runtime overhead (CUDA context, allocator fragmentation, activations, workspaces) is not estimated.",
        "The ledger total is an arithmetic sum of bytes, not a measured VRAM reading and not a fit guarantee.",
        "Only full, uniform attention across layers is modelled; sliding windows, hybrid layers and MLA are refused.",
        "Only fp16/bf16 KV cache is modelled; quantized KV cache has no exact-size claim here.",
        "Tensor-parallel replication, CUDA graphs, paged-attention block padding and prefix caching are not modelled.",
        "Partial CPU/GPU offload is refused: with offload the GPU resident set is runtime dependent.",
        f"Context is bounded by this config's max_position_embeddings = {max_ctx}; rope/YaRN extrapolation is not modelled.",
    ]
    notes = [
        f"head_dim taken from {geom.head_dim_source} = {geom.head_dim}.",
        f"KV bytes/token = 2 * {geom.num_hidden_layers} layers * {geom.num_key_value_heads} KV heads * "
        f"{geom.head_dim} head_dim * {KV_DTYPE_BYTES[kv_dtype]} bytes = {kv_bpt} bytes.",
    ]
    if estimated:
        notes.append(
            "Weights for this quantisation are an ESTIMATE (bytes/parameter), because the official "
            "repository publishes no file of that type. Treat it as approximate and verify against the "
            "actual file you run."
        )

    return Ledger(
        model_id=model.get("id", ""),
        display_name=model.get("display_name", model.get("id", "")),
        quant=quant,
        kv_dtype=kv_dtype,
        context=context,
        batch=batch,
        geometry=geom,
        weights_bytes=int(entry["bytes"]),
        weights_source=str(entry["source"]),
        weights_detail=str(entry.get("detail", "")),
        weights_estimated=estimated,
        kv_bytes_per_token=kv_bpt,
        kv_bytes_total=kv_total,
        manual_overhead_bytes=overhead_bytes,
        manual_overhead_label=overhead_label,
        source={
            "hub": model.get("source", {}).get("hub"),
            "repo": model.get("source", {}).get("repo"),
            "revision": model.get("source", {}).get("revision"),
            "revision_url": model.get("source", {}).get("revision_url"),
            "config_sha256": model.get("source", {}).get("config_sha256"),
            "license": model.get("license"),
        },
        max_context=max_ctx,
        limits=limits,
        notes=notes,
    )


def refuse_offload(requested_gib: object) -> None:
    """Partial offload is a runtime property we do not model."""
    raise UnsupportedFeature(
        f"--cpu-offload-gib={requested_gib!r}: partial CPU/GPU offload is not modelled. With offload the GPU "
        "resident set depends on the runtime, layer placement and allocator behaviour, so any number here "
        "would be a guess. Remove the flag for a full-GPU ledger.",
        code="offload_unsupported",
        details={"requested": requested_gib},
    )


def refuse_legacy_params(params: object) -> None:
    """The old ``--params`` mode derived layers from a parameter count."""
    raise UnsupportedQuant(
        f"--params {params} is no longer accepted as an exact calculation: the old mode guessed layer "
        "counts and KV heads from a parameter count, which is not config-grounded. Use "
        "--model qwen2.5-7b-instruct (see `python -m vram_calc models`). If you only need a labelled "
        "rough estimate, pass --legacy-heuristic explicitly.",
        code="legacy_params_removed",
        details={"params": params, "migration": "python -m vram_calc calc --model <id> --quant <q> --ctx <n>"},
    )


#: Legacy bytes/parameter table, kept ONLY for the labelled heuristic mode that
#: the old CLI used. It is not used anywhere in the config-grounded ledger.
LEGACY_BYTES_PER_PARAM = {
    "fp16": 2.0,
    "bf16": 2.0,
    "fp8": 1.0,
    "q8_0": 1.06,
    "q6_k": 0.82,
    "q5_k_m": 0.70,
    "q4_k_m": 0.588,
    "q3_k_m": 0.45,
    "nvfp4": 0.52,
    "q2_k": 0.35,
}


def legacy_heuristic_estimate(
    params_b: float,
    quant: str = "q4_k_m",
    context: int = 8192,
    batch: int = 1,
    kv_dtype: str = "fp16",
) -> dict:
    """Reproduce the OLD heuristic, explicitly labelled as non-config-grounded.

    Previous releases guessed ``layers`` from a parameter count and hard-coded 8
    KV heads and head_dim 128. That is kept here behind ``--legacy-heuristic`` so
    existing users get a migration path, but the result is flagged
    ``config_grounded: False`` and is never presented as a ledger.
    """
    p = float(params_b)
    if not math.isfinite(p) or p <= 0 or p > 100_000:
        raise InvalidParameter(
            f"--params must be a finite parameter count in (0, 100000], got {params_b!r}",
            code="invalid_params",
        )
    q = (quant or "").strip().lower()
    if q not in LEGACY_BYTES_PER_PARAM:
        raise UnsupportedQuant(
            f"quant {quant!r} is not in the legacy table ({', '.join(sorted(LEGACY_BYTES_PER_PARAM))})",
            code="unsupported_quant",
        )
    context = _check_int("context", context, MAX_CONTEXT_TOKENS)
    batch = _check_int("batch", batch, MAX_BATCH)
    if context * batch > MAX_POSITIONS:
        raise InvalidParameter("context*batch exceeds the supported bound", code="positions_too_large")
    kv_dtype = _normalise_kv_dtype(kv_dtype)

    layers = int(math.ceil(14 + 7 * math.log2(max(1.0, p))))
    kv_heads = 8
    head_dim = 128
    weights_bytes = int(round(p * 1e9 * LEGACY_BYTES_PER_PARAM[q]))
    kv_bytes = 2 * layers * kv_heads * head_dim * KV_DTYPE_BYTES[kv_dtype] * context * batch
    return {
        "mode": "legacy_heuristic",
        "config_grounded": False,
        "params_b": p,
        "quant": q,
        "context": context,
        "batch": batch,
        "assumptions": [
            f"layers guessed as ceil(14 + 7*log2({p})) = {layers}",
            f"KV heads hard-coded to {kv_heads}, head_dim hard-coded to {head_dim}",
            "weights = params * bytes/param (no official file size involved)",
            "no runtime overhead included; not a ledger total",
        ],
        "weights_bytes": weights_bytes,
        "weights_gib": round(weights_bytes / GIB, 4),
        "kv_bytes": kv_bytes,
        "kv_gib": round(kv_bytes / GIB, 4),
        "total_gib": round((weights_bytes + kv_bytes) / GIB, 4),
        "warning": "HEURISTIC, not config-grounded: use --model for a real ledger.",
    }
