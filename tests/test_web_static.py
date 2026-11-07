# -*- coding: utf-8 -*-
"""Static-site build, branding, links and documentation honesty."""
from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
README = ROOT / "README.md"


def test_web_build_is_reproducible():
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "build_web.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    assert res.returncode == 0, res.stdout + res.stderr


def test_index_is_self_contained():
    html = INDEX.read_text(encoding="utf-8")
    assert "__CATALOG__" not in html
    assert "__STATE_JS__" not in html and "__APP_JS__" not in html and "__STYLES__" not in html
    assert 'name="viewport"' in html
    assert "width=device-width" in html
    # no runtime fetches / external scripts: the page must work from file://
    assert not re.search(r'<script[^>]+src="https?://', html)
    assert "fetch(" not in html
    assert "XMLHttpRequest" not in html


def test_embedded_catalog_matches_pinned_data(catalog_raw):
    html = INDEX.read_text(encoding="utf-8")
    m = re.search(r"window\.VRAM_CATALOG = (\{.*?\});\s*\n", html, re.S)
    assert m, "catalogue payload not embedded"
    embedded = json.loads(m.group(1))
    assert [x["id"] for x in embedded["models"]] == [x["id"] for x in catalog_raw["models"]]
    for got, want in zip(embedded["models"], catalog_raw["models"]):
        assert got["source"]["revision"] == want["source"]["revision"]
        assert got["weights"]["q4_k_m"]["bytes"] == want["weights"]["q4_k_m"]["bytes"]
        assert got["config"]["num_hidden_layers"] == want["config"]["num_hidden_layers"]


def test_branding_is_zexkora_hub_only():
    html = INDEX.read_text(encoding="utf-8")
    assert "Zexkora Hub" in html
    lowered = html.lower()
    # the old, broken brand and the retired gateway domain must be gone
    assert ".llm hub" not in lowered
    assert "llm_hubs" not in lowered
    assert "ai.zxc1x1.ru" not in lowered


def test_live_tool_urls_are_github_pages_only():
    for path in (INDEX, README):
        text = path.read_text(encoding="utf-8")
        assert "dangerousanen.github.io/llm-vram-calculator" in text
        for bad in ("ai.zxc1x1.ru", "zexkora.cc", "2api.", "token.zexkora"):
            assert bad not in text, f"{path.name} still points at {bad}"


def test_page_has_share_and_copy_controls():
    html = INDEX.read_text(encoding="utf-8")
    assert "t.me/share/url" in html
    assert 'id="copy-report"' in html
    assert 'id="copy-link"' in html
    assert "navigator.clipboard" in html


def test_page_states_honest_limits():
    html = INDEX.read_text(encoding="utf-8")
    for phrase in ("неизвестен", "не измеренная VRAM", "KV-кэш", "не гарантия"):
        assert phrase.lower() in html.lower()
    assert "max_position_embeddings" in html


def test_readme_documents_limits_formula_and_tests():
    text = README.read_text(encoding="utf-8")
    for needle in ("GiB", "max_position_embeddings", "head_dim", "pytest", "MoE", "MLA", "sliding window", "offload"):
        assert needle in text, needle
    assert "--params" in text and "legacy" in text.lower()
    assert "python -m vram_calc" in text


def test_readme_has_no_fake_precision_claims():
    text = README.read_text(encoding="utf-8").lower()
    # a guarantee claim is only acceptable when explicitly negated
    assert "гарантируем" not in text
    assert "guaranteed to fit" not in text
    assert "не является измеренной" in text or "не измеренная vram" in text
    assert "не гарантия" in text or "не гарантирует" in text


def test_quickstart_documents_offline_usage_and_tests():
    doc = ROOT / "docs" / "quickstart.en.md"
    assert doc.exists()
    text = doc.read_text(encoding="utf-8")
    for needle in ("pip install", "pytest", "vram-calc", "GiB", "revision", "Telegram"):
        assert needle in text, needle


def test_packaging_entry_points_are_consistent_with_package():
    text = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'llm-vram-calc = "vramcalc.cli:main"' in text
    assert 'vram-calc = "vramcalc.cli:main"' in text
    assert "models.json" in text or "package-data" in text


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_state_tests_pass():
    res = subprocess.run(
        ["node", str(ROOT / "tests" / "node" / "state.test.cjs")],
        capture_output=True,
        text=True,
        cwd=str(ROOT),
        timeout=120,
    )
    assert res.returncode == 0, res.stdout + res.stderr
    assert "OK" in res.stdout


@pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")
def test_node_ledger_matches_python_ledger(qwen25, qwen3):
    from vramcalc.core import compute_ledger

    probe = ROOT / "tests" / "node" / "ledger_probe.cjs"
    probe.write_text(
        "const S = require('../../web/state.js');\n"
        "const cat = require('../../vramcalc/data/models.json');\n"
        "const m = cat.models.find(x => x.id === process.argv[2]);\n"
        "const led = S.computeLedger(m, {quant: process.argv[3], context: +process.argv[4], batch: +process.argv[5], kvDtype: 'fp16', overhead: 0});\n"
        "console.log(JSON.stringify(led));\n",
        encoding="utf-8",
    )
    try:
        for model, quant, ctx, batch in (
            (qwen25, "q4_k_m", 8192, 1),
            (qwen25, "bf16", 32768, 2),
            (qwen3, "q6_k", 16384, 3),
        ):
            res = subprocess.run(
                ["node", str(probe), model["id"], quant, str(ctx), str(batch)],
                capture_output=True,
                text=True,
                cwd=str(probe.parent),
                timeout=60,
            )
            assert res.returncode == 0, res.stderr
            js = json.loads(res.stdout)
            py = compute_ledger(model, quant, ctx, batch).to_dict()
            assert js["weightsBytes"] == py["weights"]["bytes"]
            assert js["kvBytes"] == py["kv_cache"]["bytes"]
            assert js["totalBytes"] == py["ledger_total"]["bytes"]
    finally:
        probe.unlink(missing_ok=True)