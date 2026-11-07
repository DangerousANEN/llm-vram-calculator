# -*- coding: utf-8 -*-
"""Shared fixtures. Imports are deliberately lazy so a missing package shows up
as a per-test failure with a readable message instead of a collection crash."""
from __future__ import annotations

import json
import pathlib
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="session")
def root() -> pathlib.Path:
    return ROOT


@pytest.fixture(scope="session")
def catalog_raw() -> dict:
    return json.loads((ROOT / "vramcalc" / "data" / "models.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def qwen25() -> dict:
    from vramcalc.catalog import get_model

    return get_model("qwen2.5-7b-instruct")


@pytest.fixture(scope="session")
def qwen3() -> dict:
    from vramcalc.catalog import get_model

    return get_model("qwen3-8b")


@pytest.fixture(scope="session")
def screenshots_dir() -> pathlib.Path:
    path = ROOT / "validation" / "screenshots"
    path.mkdir(parents=True, exist_ok=True)
    return path