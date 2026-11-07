# -*- coding: utf-8 -*-
"""Command line interface.

Human output is a transparent ledger with explicit units, sources and limits.
``--json`` emits the same facts for scripts and tests. Anything unsupported is
refused with a stable error code and a migration hint instead of a guessed
number.
"""
from __future__ import annotations

import argparse
import json
import sys

from . import __version__
from .errors import CalcError
from .catalog import corpus_facts, get_model, list_models, published_quants
from .core import (
    GIB,
    KV_DTYPE_BYTES,
    compute_ledger,
    legacy_heuristic_estimate,
    refuse_legacy_params,
    refuse_offload,
)
from .bench import DEFAULT_PROMPT, run_bench

EPILOG = """\
units: every memory figure is GiB (1 GiB = 1024^3 bytes) and the exact byte value
       is printed too. Nothing here is a measured VRAM reading.

examples:
  python -m vram_calc models
  python -m vram_calc calc --model qwen2.5-7b-instruct --quant q4_k_m --ctx 8192
  python -m vram_calc calc --model qwen3-8b --quant bf16 --ctx 32768 --batch 4 --json
  python -m vram_calc bench --url http://127.0.0.1:8080/v1 --model local-model
"""


def _gib(n: int) -> str:
    return f"{n / GIB:.3f} GiB"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="vram-calc",
        description="Config-grounded memory ledger for dense LLM inference (weights + KV bytes).",
        epilog=EPILOG,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"llm-vram-calculator {__version__}")
    p.add_argument("--json", action="store_true", help="machine readable output")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("models", help="list revision-pinned supported models").add_argument(
        "--json", action="store_true", default=argparse.SUPPRESS, help="machine readable output"
    )

    c = sub.add_parser("calc", help="weights + KV cache ledger for one model")
    c.add_argument("--model", "-m", help="catalogue model id (see `models`)")
    c.add_argument("--quant", "-q", default="q4_k_m", help="weight quantisation (default: q4_k_m)")
    c.add_argument("--ctx", type=int, default=8192, help="allocated context in tokens (default: 8192)")
    c.add_argument("--batch", type=int, default=1, help="parallel sequences (default: 1)")
    c.add_argument(
        "--kv-dtype",
        default="fp16",
        choices=sorted(KV_DTYPE_BYTES),
        help="KV cache element type; only fp16/bf16 are modelled",
    )
    c.add_argument(
        "--overhead-gib",
        type=float,
        default=None,
        help="manual runtime-overhead bound in GiB (not estimated by the tool)",
    )
    c.add_argument(
        "--allow-estimate",
        action="store_true",
        help="allow a labelled bytes/parameter estimate when no official file exists for the quant",
    )
    # --- removed / refused legacy options -------------------------------------
    c.add_argument("--params", type=float, default=None, help="REMOVED: parameter-count heuristic mode")
    c.add_argument(
        "--legacy-heuristic",
        action="store_true",
        help="with --params: print a clearly labelled non-config-grounded rough estimate",
    )
    c.add_argument("--kv-quant", default=None, help="REMOVED: quantized KV cache is not modelled")
    c.add_argument("--mla", action="store_true", help="REMOVED: MLA geometry is not modelled")
    c.add_argument("--cpu-offload-gib", type=float, default=None, help="REMOVED: offload is not modelled")
    c.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine readable output")

    b = sub.add_parser("bench", help="streaming latency/throughput probe (token rate only when reported)")
    b.add_argument("--url", default="http://127.0.0.1:8000/v1", help="OpenAI-compatible base URL")
    b.add_argument("--model", default="local-model", help="model name for the request")
    b.add_argument("--prompt", default=DEFAULT_PROMPT)
    b.add_argument("--tokens", type=int, default=120, help="max_tokens (1..8192)")
    b.add_argument("--api-key", default=None, help="bearer token (prefer the env var instead)")
    b.add_argument("--json", action="store_true", default=argparse.SUPPRESS, help="machine readable output")
    return p


def _print_models(as_json: bool) -> int:
    if as_json:
        print(json.dumps({"models": list_models(), "catalog": corpus_facts()}, indent=2, ensure_ascii=False))
        return 0
    cat = corpus_facts()
    print("Supported revision-pinned dense models")
    print(f"catalogue: {cat['data_source']} (generated {cat['generated_at']})")
    for m in list_models():
        src = m["source"]
        cfg = m["config"]
        print()
        print(f"  {m['id']}  -  {m['display_name']}  [{m['license']}]")
        print(f"    repo      : {src['repo']} @ {src['revision']}")
        print(f"    config    : sha256 {src['config_sha256'][:16]}...")
        print(
            f"    geometry  : {cfg['num_hidden_layers']} layers, {cfg['num_attention_heads']} heads, "
            f"{cfg['num_key_value_heads']} KV heads"
        )
        qs = []
        for q in published_quants(m):
            qs.append(f"{q} ({m['weights'][q]['bytes']} B)")
        print(f"    exact quantisations: {', '.join(qs)}")
    print()
    print("Unsupported families (MoE, MLA, sliding window, hybrid attention) are refused, never approximated.")
    return 0


def _print_ledger(d: dict) -> None:
    geom = d["geometry"]
    print()
    print("=" * 68)
    print(f" Memory ledger: {d['display_name']}  [{d['quant']}]")
    print("=" * 68)
    print(f" source            : {d['source']['repo']} @ {d['source']['revision']}")
    print(f" config sha256     : {d['source']['config_sha256']}")
    print(f" license           : {d['source']['license']}")
    print("-" * 68)
    print(
        f" geometry          : {geom['num_hidden_layers']} layers x {geom['num_key_value_heads']} KV heads "
        f"x {geom['head_dim']} head_dim  (head_dim from {geom['head_dim_source']})"
    )
    print(
        f" context           : {d['context']} tokens x {d['batch']} sequence(s), KV dtype {d['kv_dtype']} "
        f"({KV_DTYPE_BYTES[d['kv_dtype']]} bytes/element)"
    )
    print("-" * 68)
    print(f" weights           : {_gib(d['weights']['bytes']):>12}  ({d['weights']['bytes']} bytes)")
    print(f"   source          : {d['weights']['source']}")
    print(f"   {d['weights']['detail']}")
    if d["weights"]["estimated"]:
        print("   !! ESTIMATE, not an official file size")
    print(f" KV cache          : {_gib(d['kv_cache']['bytes']):>12}"
          f"  ({d['kv_cache']['bytes']} bytes, {d['kv_cache']['bytes_per_token']} B/token/sequence)")
    print(f" subtotal          : {_gib(d['subtotal']['bytes']):>12}  (weights + KV)")
    print(
        f" runtime overhead  : {_gib(d['runtime_overhead']['bytes']):>12}  ({d['runtime_overhead']['kind']})"
    )
    if d["runtime_overhead"]["label"]:
        print(f"   {d['runtime_overhead']['label']}")
    print(f" LEDGER TOTAL      : {_gib(d['ledger_total']['bytes']):>12}  (arithmetic sum, NOT measured VRAM)")
    print("-" * 68)
    print(" limits:")
    for item in d["limits"]:
        print(f"   - {item}")
    print()
    print(" " + d["verification"]["statement"])
    print("=" * 68)
    print()


def _refuse_removed_flags(args) -> None:
    """Removed flags are refused before anything else, so they can never slip
    through the legacy migration path and look supported."""
    if args.cpu_offload_gib is not None:
        refuse_offload(args.cpu_offload_gib)
    if args.mla:
        from .errors import UnsupportedArchitecture

        raise UnsupportedArchitecture(
            "--mla: Multi-head Latent Attention has its own compressed cache geometry; this release does "
            "not model it and refuses instead of guessing.",
            code="mla_not_supported",
        )
    if args.kv_quant:
        from .errors import UnsupportedFeature

        raise UnsupportedFeature(
            f"--kv-quant {args.kv_quant}: only fp16/bf16 KV cache is modelled. Quantized KV cache sizes are "
            "runtime/block dependent and are refused instead of approximated. Use --kv-dtype fp16|bf16.",
            code="unsupported_kv_dtype",
        )


def _cmd_calc(args) -> int:
    _refuse_removed_flags(args)
    if args.params is not None and not args.legacy_heuristic:
        refuse_legacy_params(args.params)
    if args.params is not None and args.legacy_heuristic:
        est = legacy_heuristic_estimate(args.params, args.quant, args.ctx, args.batch)
        if args.json:
            print(json.dumps(est, indent=2, ensure_ascii=False))
            return 0
        print()
        print("=" * 68)
        print(" LEGACY HEURISTIC ESTIMATE - NOT CONFIG-GROUNDED")
        print("=" * 68)
        print(" This deliberately reproduces the old parameter-count heuristic so existing")
        print(" scripts can be migrated. Layer count and KV heads are GUESSED, not read from")
        print(" any model config, so this is not a ledger and must not be quoted as exact.")
        print("-" * 68)
        for k in est["assumptions"]:
            print(f"   - {k}")
        print("-" * 68)
        print(f" estimated weights : {est['weights_gib']:.3f} GiB ({est['weights_bytes']} bytes)")
        print(f" estimated KV      : {est['kv_gib']:.3f} GiB ({est['kv_bytes']} bytes)")
        print(f" heuristic total   : {est['total_gib']:.3f} GiB (no overhead estimate included)")
        print("=" * 68)
        print(f" migrate: python -m vram_calc calc --model <id> --quant {args.quant} --ctx {args.ctx}")
        print()
        return 0
    if not args.model:
        raise CalcError(
            "calc needs --model. Run `python -m vram_calc models` for the supported list.",
            code="missing_model",
        )

    model = get_model(args.model)
    ledger = compute_ledger(
        model,
        quant=args.quant,
        context=args.ctx,
        batch=args.batch,
        kv_dtype=args.kv_dtype,
        manual_overhead_gib=args.overhead_gib,
        allow_estimate=args.allow_estimate,
    )
    data = ledger.to_dict()
    if args.json:
        print(json.dumps(data, indent=2, ensure_ascii=False))
    else:
        _print_ledger(data)
    return 0


def _cmd_bench(args) -> int:
    import os

    api_key = args.api_key or os.environ.get("VRAM_CALC_API_KEY")
    res = run_bench(args.url, args.model, args.prompt, args.tokens, api_key=api_key)
    if args.json:
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    print()
    print("=" * 68)
    print(f" Streaming probe: {res['model']}")
    print("=" * 68)
    print(f" endpoint          : {res['url']}")
    print(f" TTFT              : {res['ttft_ms']} ms" if res["ttft_ms"] is not None else " TTFT              : n/a")
    if res["tps"] is not None:
        print(f" tokens/s          : {res['tps']}   basis: {res['tps_basis']}")
    else:
        print(f" tokens/s          : not claimed - {res['tps_unavailable_reason']}")
    print(f" completion tokens : {res['completion_tokens']} (provider reported)")
    print(f" stream chunks     : {res['stream_chunks']}   chunk rate: {res['chunk_rate_per_s']} "
          f"{res['chunk_rate_label']}")
    print(f" total wall time   : {res['total_time_s']} s")
    print("-" * 68)
    for w in res["warnings"]:
        print(f" ! {w}")
    print("=" * 68)
    print()
    return 0


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)
    # convenience: allow `vram-calc --model X --ctx 8192` without the subcommand
    if raw and raw[0].startswith("-") and raw[0] not in ("--json", "--version", "-h", "--help"):
        raw = ["calc"] + raw
    parser = build_parser()
    args = parser.parse_args(raw)
    as_json = bool(getattr(args, "json", False))
    try:
        if args.cmd == "models":
            return _print_models(as_json)
        if args.cmd == "bench":
            return _cmd_bench(args)
        if args.cmd == "calc":
            return _cmd_calc(args)
        parser.print_help()
        return 0
    except CalcError as exc:
        if as_json:
            print(json.dumps(exc.to_dict(), indent=2, ensure_ascii=False))
        else:
            print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return exc.exit_code


if __name__ == "__main__":
    raise SystemExit(main())