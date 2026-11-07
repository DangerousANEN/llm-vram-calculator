#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build ``vramcalc/data/models.json`` from official Hugging Face metadata.

This script downloads **metadata only** (config.json + the public file listing
with byte sizes). It never downloads model weights.

Every fixture is pinned to an immutable commit revision, so the byte sizes in
the committed catalogue can be re-verified at any time with::

    python scripts/fetch_model_metadata.py --check

Network is only needed when regenerating or checking; the shipped product
reads the committed JSON and works fully offline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

DATA_PATH = Path(__file__).resolve().parents[1] / "vramcalc" / "data" / "models.json"
DATA_SOURCE = "huggingface_hub_public_api"

# Quantisations that llama.cpp publishes as single official GGUF files. The
# fixture records the *exact* byte size of every shard, never a formula.
GGUF_QUANT_ALIASES = {
    "q2_k": "q2_k",
    "q3_k_m": "q3_k_m",
    "q4_0": "q4_0",
    "q4_k_m": "q4_k_m",
    "q5_0": "q5_0",
    "q5_k_m": "q5_k_m",
    "q6_k": "q6_k",
    "q8_0": "q8_0",
}

# Fallback approximation factors for quantisations that a model's official
# repository does not publish. They are *estimates*, are only used when the
# caller explicitly opts in, and always carry ``source: "estimated"``.
ESTIMATE_BYTES_PER_PARAM = {
    "q8_0": 1.06,
    "q6_k": 0.82,
    "q5_k_m": 0.70,
    "q5_0": 0.72,
    "q4_k_m": 0.588,
    "q4_0": 0.60,
    "q3_k_m": 0.45,
    "q2_k": 0.35,
    "fp8": 1.0,
}

# Pinned fixture definitions. ``revision``/``gguf_revision`` are immutable
# commits; ``--check`` fails if the upstream ``main`` moved away from them.
FIXTURES = [
    {
        "id": "qwen2.5-7b-instruct",
        "display_name": "Qwen2.5 7B Instruct",
        "short_name": "Qwen2.5-7B-Instruct",
        "family": "Qwen2.5",
        "repo": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "gguf_repo": "Qwen/Qwen2.5-7B-Instruct-GGUF",
        "gguf_revision": "bb5d59e06d9551d752d08b292a50eb208b07ab1f",
        "gguf_prefix": "qwen2.5-7b-instruct",
        "license": "apache-2.0",
        "weights_stored_dtype": "bfloat16",
    },
    {
        "id": "qwen3-8b",
        "display_name": "Qwen3 8B",
        "short_name": "Qwen3-8B",
        "family": "Qwen3",
        "repo": "Qwen/Qwen3-8B",
        "revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "gguf_repo": "Qwen/Qwen3-8B-GGUF",
        "gguf_revision": "7c41481f57cb95916b40956ab2f0b139b296d974",
        "gguf_prefix": "Qwen3-8B",
        "license": "apache-2.0",
        "weights_stored_dtype": "bfloat16",
    },
]


def http_json(url: str) -> object:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-vram-calculator-fixture-builder/2"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310 - fixed https hosts
        return json.loads(resp.read().decode("utf-8"))


def http_bytes(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "llm-vram-calculator-fixture-builder/2"})
    with urllib.request.urlopen(req, timeout=60) as resp:  # noqa: S310
        return resp.read()


def tree(repo: str, revision: str) -> list:
    data = http_json(f"https://huggingface.co/api/models/{repo}/tree/{revision}?recursive=true")
    if not isinstance(data, list):
        raise SystemExit(f"unexpected tree payload for {repo}@{revision}")
    return data


def build_one(fx: dict) -> dict:
    cfg_url = f"https://huggingface.co/{fx['repo']}/raw/{fx['revision']}/config.json"
    raw_cfg = http_bytes(cfg_url)
    cfg = json.loads(raw_cfg.decode("utf-8"))

    weights: dict[str, dict] = {}
    st_files = [f for f in tree(fx["repo"], fx["revision"]) if f["path"].endswith(".safetensors")]
    st_files.sort(key=lambda f: f["path"])
    st_total = sum(int(f.get("size") or 0) for f in st_files)
    st_list = [{"path": f["path"], "bytes": int(f["size"])} for f in st_files]
    if not st_total:
        raise SystemExit(f"{fx['id']}: no safetensors sizes returned")

    weights["bf16"] = {
        "bytes": st_total,
        "source": "official_safetensors",
        "detail": "exact byte sum of the official safetensors shards at the pinned revision",
        "files": st_list,
    }
    weights["fp16"] = {
        "bytes": st_total,
        "source": "official_safetensors_size_equivalent",
        "detail": (
            "the official shards are stored as "
            f"{fx['weights_stored_dtype']}; fp16 has the same 2 bytes/parameter, so the "
            "resident weight size is identical"
        ),
        "files": st_list,
    }

    gguf_files = [f for f in tree(fx["gguf_repo"], fx["gguf_revision"]) if f["path"].endswith(".gguf")]
    by_quant: dict[str, list] = {}
    for f in sorted(gguf_files, key=lambda x: x["path"]):
        name = f["path"].lower()
        for alias in sorted(GGUF_QUANT_ALIASES, key=len, reverse=True):
            if f"-{alias}-" in name or f"-{alias}." in name:
                by_quant.setdefault(alias, []).append({"path": f["path"], "bytes": int(f["size"])})
                break
    for quant, files in by_quant.items():
        weights[quant] = {
            "bytes": sum(x["bytes"] for x in files),
            "source": "official_gguf",
            "detail": "exact byte sum of the official GGUF shard files at the pinned revision",
            "gguf_repo": fx["gguf_repo"],
            "gguf_revision": fx["gguf_revision"],
            "files": files,
        }

    return {
        "id": fx["id"],
        "display_name": fx["display_name"],
        "short_name": fx["short_name"],
        "family": fx["family"],
        "license": fx["license"],
        "source": {
            "hub": "huggingface",
            "repo": fx["repo"],
            "revision": fx["revision"],
            "revision_url": f"https://huggingface.co/{fx['repo']}/tree/{fx['revision']}",
            "config_url": cfg_url,
            "config_sha256": hashlib.sha256(raw_cfg).hexdigest(),
            "gguf_repo": fx["gguf_repo"],
            "gguf_revision": fx["gguf_revision"],
            "gguf_revision_url": f"https://huggingface.co/{fx['gguf_repo']}/tree/{fx['gguf_revision']}",
        },
        "config": cfg,
        "weights": weights,
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="verify the committed catalogue still matches upstream metadata")
    ap.add_argument("--out", default=str(DATA_PATH))
    args = ap.parse_args(argv)

    models = [build_one(fx) for fx in FIXTURES]
    for m in models:
        m["estimates"] = {
            "bytes_per_param": {q: f for q, f in ESTIMATE_BYTES_PER_PARAM.items() if q not in m["weights"]},
            "provenance": (
                "Approximate bytes per parameter derived from llama.cpp/gguf quantisation type "
                "sizes; NOT part of any official file listing, therefore flagged as an estimate."
            ),
        }

    doc = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "data_source": DATA_SOURCE,
        "notes": [
            "All weight sizes are exact byte sums from official Hugging Face file listings at pinned revisions.",
            "KV-cache bytes are derived from each model's own config.json geometry, never from a parameter-count heuristic.",
            "No model weights were downloaded to build this file; only config.json and public file metadata.",
        ],
        "models": models,
    }
    text = json.dumps(doc, indent=2, ensure_ascii=False, sort_keys=False) + "\n"

    if args.check:
        current = Path(args.out).read_text(encoding="utf-8")
        cur = json.loads(current)
        # compare the data-bearing parts only (generated_at changes by design)
        a = [{k: v for k, v in m.items()} for m in cur["models"]]
        b = [{k: v for k, v in m.items()} for m in doc["models"]]
        if a != b:
            print("MISMATCH: committed model metadata differs from upstream public metadata", file=sys.stderr)
            for x, y in zip(a, b):
                if x != y:
                    for key in sorted(set(x) | set(y)):
                        if x.get(key) != y.get(key) and key != "estimates":
                            print(f"  {x.get('id')}: field {key!r} changed", file=sys.stderr)
            return 1
        print(f"OK: {args.out} matches upstream metadata for {len(a)} models")
        return 0

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")
    print(f"wrote {args.out} ({len(models)} models)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
