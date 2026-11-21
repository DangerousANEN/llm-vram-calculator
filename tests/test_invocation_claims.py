# -*- coding: utf-8 -*-
"""P1-18: the invocation the project tells users to run must not be the shim.

Every documented command used `python -m vram_calc`, which in this repository
resolves to the legacy top-level module ``vram_calc.py`` (the real package is
``vramcalc``, no underscore). That shim prints a deprecation notice on stderr
for every command - including a notice that tells the reader to prefer
``python -m vram_calc``, i.e. the very command they just ran. Copying the
README therefore greets a new user with a self-referential deprecation warning.

These tests pin the claims a user reads (README, quickstart, ``--help``, the
shim notice and the CLI hints) to the module that actually exists.
"""
from __future__ import annotations

import os
import pathlib
import re
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
QUICKSTART = ROOT / "docs" / "quickstart.en.md"

DEPRECATED_MODULE = re.compile(r"python -m vram_calc(?![a-z])")
GOOD_MODULE = "python -m vramcalc"


def run(*args, cwd=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT)
    return subprocess.run(
        [sys.executable, *args],
        cwd=str(cwd or ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )


@pytest.mark.parametrize("doc", [README, QUICKSTART], ids=["README", "quickstart"])
def test_docs_never_instruct_the_deprecated_shim(doc):
    text = doc.read_text(encoding="utf-8")
    offenders = [m.group(0) for m in DEPRECATED_MODULE.finditer(text)]
    assert not offenders, f"{doc.name} still tells users to run {offenders[:3]}"
    assert GOOD_MODULE in text, f"{doc.name} no longer shows how to run the package"


def test_help_text_points_at_the_package_not_the_shim():
    res = run("-m", "vramcalc", "--help")
    assert res.returncode == 0, res.stderr
    assert not DEPRECATED_MODULE.search(res.stdout), "help text tells users to run the legacy shim"
    assert GOOD_MODULE in res.stdout


def test_documented_command_runs_clean_without_a_deprecation_notice():
    res = run("-m", "vramcalc", "models")
    assert res.returncode == 0, res.stderr
    assert "deprecated" not in res.stderr.lower(), res.stderr
    assert res.stderr.strip() == "", res.stderr


def test_legacy_shim_notice_names_the_working_command():
    res = run(str(ROOT / "vram_calc.py"), "calc", "--model", "qwen3-8b", "--ctx", "8192")
    assert res.returncode == 0, res.stderr
    assert "deprecated" in res.stderr.lower()
    assert GOOD_MODULE in res.stderr
    assert not DEPRECATED_MODULE.search(res.stderr), "the shim tells users to run the shim"


def test_refusal_hints_name_the_working_command():
    """Hints printed by the CLI must be copy-pasteable as written."""
    res = run("-m", "vramcalc", "calc", "--params", "7", "--quant", "q4_k_m", "--ctx", "8192")
    assert res.returncode != 0
    blob = res.stdout + res.stderr
    assert not DEPRECATED_MODULE.search(blob), blob
    assert GOOD_MODULE in blob, blob


def test_package_main_docstring_names_the_package():
    res = run("-c", "import vramcalc.__main__ as m; print(m.__doc__ or '')", cwd=str(ROOT))
    assert res.returncode == 0, res.stderr
    assert "vram_calc" not in res.stdout, res.stdout
