# -*- coding: utf-8 -*-
"""CLI behaviour: usable happy paths, loud refusals, working migration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]


def run(*args, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    env.pop("VRAM_CALC_API_KEY", None)
    return subprocess.run(
        [sys.executable, "-m", "vramcalc", *args],
        cwd=str(cwd or ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_models_lists_both_fixtures_with_revisions():
    res = run("models")
    assert res.returncode == 0, res.stderr
    assert "qwen2.5-7b-instruct" in res.stdout
    assert "qwen3-8b" in res.stdout
    assert "a09a35458c702b33eeacc393d103063234e8bc28" in res.stdout
    assert "b968826d9c46dd6066d109eabc6255188de91218" in res.stdout
    assert "refused" in res.stdout


def test_models_json_is_machine_readable():
    res = run("models", "--json")
    data = json.loads(res.stdout)
    assert [m["id"] for m in data["models"]] == [
        "qwen2.5-7b-instruct",
        "qwen3-8b",
        "qwen3-14b",
        "qwen3-4b",
    ]
    assert data["catalog"]["data_source"] == "huggingface_hub_public_api"


def test_human_calc_output_states_units_and_limits():
    res = run("calc", "--model", "qwen2.5-7b-instruct", "--quant", "q4_k_m", "--ctx", "8192")
    assert res.returncode == 0, res.stderr
    out = res.stdout
    assert "GiB" in out
    assert "4683073632 bytes" in out
    assert "469762048" in out
    assert "NOT measured VRAM" in out
    assert "3922749440" not in out  # old wrong heuristic total, must be gone
    assert "limits:" in out


def test_calc_json_matches_core_numbers(qwen3):
    from vramcalc.core import compute_ledger

    res = run("calc", "--model", "qwen3-8b", "--quant", "bf16", "--ctx", "32768", "--batch", "2", "--json")
    assert res.returncode == 0, res.stderr
    data = json.loads(res.stdout)
    expected = compute_ledger(qwen3, "bf16", 32768, 2).to_dict()
    assert data["kv_cache"]["bytes"] == expected["kv_cache"]["bytes"] == 9_663_676_416
    assert data["ledger_total"]["bytes"] == expected["ledger_total"]["bytes"]
    assert data["ledger_total"]["is_fit_guarantee"] is False


def test_convenience_mode_without_subcommand():
    res = run("--model", "qwen2.5-7b-instruct", "--ctx", "4096")
    assert res.returncode == 0, res.stderr
    assert "Memory ledger" in res.stdout


def test_missing_model_is_a_clear_error():
    res = run("calc", "--quant", "q4_k_m", "--ctx", "8192")
    assert res.returncode == 2
    assert "missing_model" in res.stderr
    assert "models" in res.stderr


def test_unknown_model_is_refused():
    res = run("calc", "--model", "llama-3-70b", "--ctx", "8192")
    assert res.returncode == 2
    assert "unknown_model" in res.stderr
    assert "qwen3-8b" in res.stderr


def test_legacy_params_mode_is_refused_with_migration():
    res = run("calc", "--params", "7", "--quant", "q4_k_m", "--ctx", "8192")
    assert res.returncode == 2
    assert "legacy_params_removed" in res.stderr
    assert "--model" in res.stderr
    assert "heuristic" in res.stderr.lower()


def test_legacy_heuristic_requires_explicit_opt_in_and_is_labelled():
    res = run("calc", "--params", "7", "--quant", "q4_k_m", "--ctx", "8192", "--legacy-heuristic")
    assert res.returncode == 0, res.stderr
    assert "NOT CONFIG-GROUNDED" in res.stdout
    assert "HEURISTIC" in res.stdout.upper()
    assert "migrate:" in res.stdout


def test_legacy_heuristic_json_is_flagged():
    res = run("calc", "--params", "7", "--legacy-heuristic", "--json")
    data = json.loads(res.stdout)
    assert data["config_grounded"] is False
    assert data["mode"] == "legacy_heuristic"


def test_mla_flag_is_refused():
    res = run("calc", "--params", "671", "--mla", "--legacy-heuristic", "--ctx", "32768")
    assert res.returncode == 2
    assert "mla_not_supported" in res.stderr


def test_kv_quant_flag_is_refused():
    res = run("calc", "--model", "qwen3-8b", "--kv-quant", "fp8", "--ctx", "8192")
    assert res.returncode == 2
    assert "unsupported_kv_dtype" in res.stderr
    assert "fp16" in res.stderr


def test_offload_flag_is_refused():
    res = run("calc", "--model", "qwen3-8b", "--ctx", "8192", "--cpu-offload-gib", "4")
    assert res.returncode == 2
    assert "offload_unsupported" in res.stderr


def test_context_beyond_model_max_is_refused():
    res = run("calc", "--model", "qwen2.5-7b-instruct", "--ctx", "131072")
    assert res.returncode == 2
    assert "context_exceeds_model_max" in res.stderr


@pytest.mark.parametrize("args", [["--ctx", "0"], ["--ctx", "-5"], ["--batch", "0"], ["--batch", "9999"]])
def test_invalid_numeric_input_is_refused(args):
    res = run("calc", "--model", "qwen2.5-7b-instruct", *args)
    assert res.returncode == 2
    assert "invalid" in res.stderr or "too_large" in res.stderr


def test_json_error_output_is_parseable():
    res = run("calc", "--model", "qwen2.5-7b-instruct", "--quant", "fp8", "--json")
    assert res.returncode == 2
    payload = json.loads(res.stdout)
    assert payload["error"] == "unsupported_quant"
    assert payload["details"]["published"]


def test_version_flag():
    from vramcalc import __version__

    res = run("--version")
    assert res.returncode == 0
    assert __version__ in res.stdout


def test_legacy_script_shim_still_runs_with_migration_notice():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    res = subprocess.run(
        [sys.executable, str(ROOT / "vram_calc.py"), "calc", "--model", "qwen3-8b", "--ctx", "8192"],
        cwd=str(ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert res.returncode == 0, res.stderr
    assert "deprecated" in res.stderr
    assert "Memory ledger" in res.stdout


def test_help_mentions_units_and_examples():
    res = run("--help")
    assert res.returncode == 0
    assert "GiB" in res.stdout
    assert "python -m vram_calc calc" in res.stdout