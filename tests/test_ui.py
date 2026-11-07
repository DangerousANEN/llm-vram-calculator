# -*- coding: utf-8 -*-
"""Headless UI tests (Playwright: chromium, and firefox when available).

Run inside the isolated sandbox only:

    python -m pytest -m ui tests/test_ui.py

Real browser screenshots are written to ``validation/screenshots``; they are
inspection evidence, never a substitute for the DOM assertions here.
"""
from __future__ import annotations

import functools
import http.server
import json
import os
import socketserver
import threading

import pytest

pytestmark = pytest.mark.ui

pytest.importorskip("playwright.sync_api", reason="playwright is only installed in the UI sandbox")
from playwright.sync_api import sync_playwright  # noqa: E402

from vramcalc.core import compute_ledger  # noqa: E402
from vramcalc.catalog import get_model  # noqa: E402

GIB = 1024 ** 3


def fmt_gib(nbytes: int) -> str:
    v = nbytes / GIB
    if v >= 100:
        return f"{v:.1f} GiB"
    if v >= 10:
        return f"{v:.2f} GiB"
    return f"{v:.3f} GiB"


@pytest.fixture(scope="module")
def server():
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(root))
    handler.log_message = lambda *a, **k: None  # type: ignore[assignment]

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = Quiet(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/index.html"
    finally:
        httpd.shutdown()
        httpd.server_close()


SYSTEM_BROWSERS = {
    "chromium": ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"],
    "firefox": ["/usr/bin/firefox-esr", "/usr/bin/firefox"],
}


def _launch(browser_type, name):
    """Try the Playwright-bundled build first, then a system binary.

    The sandbox image ships /usr/bin/chromium (and Firefox when available); the
    bundled downloads are not required.
    """
    args = ["--no-sandbox", "--disable-dev-shm-usage"] if name == "chromium" else []
    try:
        return browser_type.launch(args=args)
    except Exception as first:
        for candidate in SYSTEM_BROWSERS.get(name, []):
            if not os.path.exists(candidate):
                continue
            try:
                return browser_type.launch(executable_path=candidate, args=args)
            except Exception as exc:
                last = exc
        print(f"[ui] {name} unavailable: {first}")
        if SYSTEM_BROWSERS.get(name):
            print(f"[ui] {name} system fallback failed: {locals().get('last')}")
        return None


@pytest.fixture(scope="module")
def browsers():
    with sync_playwright() as pw:
        started = {}
        for name in ("chromium", "firefox"):
            browser_type = getattr(pw, name, None)
            if browser_type is None:
                continue
            browser = _launch(browser_type, name)
            if browser is not None:
                started[name] = browser
        if "chromium" not in started:
            pytest.skip("chromium is required for UI evidence and could not start")
        try:
            yield started
        finally:
            for b in started.values():
                b.close()


@pytest.fixture(params=["chromium", "firefox"])
def page(request, browsers, server):
    if request.param not in browsers:
        pytest.skip(f"{request.param} not installed")
    context = browsers[request.param].new_context(
        viewport={"width": 1280, "height": 900},
        permissions=["clipboard-read", "clipboard-write"],
    )
    page = context.new_page()
    errors = []
    page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(server)
    page.wait_for_selector("#out-total")
    try:
        yield page
    finally:
        assert errors == [], f"{request.param} console errors: {errors}"
        context.close()


def total_text(page) -> str:
    return page.inner_text("#out-total").strip()


def test_default_state_renders_exact_ledger(page, server):
    assert "Zexkora Hub" in page.inner_text("header")
    assert page.title().startswith("Zexkora Hub")
    assert page.locator("#model option").count() == 2

    model = get_model("qwen2.5-7b-instruct")
    expected = compute_ledger(model, "bf16", 8192).to_dict()
    assert total_text(page) == fmt_gib(expected["ledger_total"]["bytes"])
    assert fmt_gib(expected["weights"]["bytes"]) in page.inner_text("#out-weights")
    assert page.inner_text("#out-kv-pertoken").strip() == "57344 B/token/seq"
    assert "НЕ измеренная VRAM" in page.inner_text(".ledger .total")

    url = page.url
    assert "m=qwen2.5-7b-instruct" in url
    assert "q=bf16" in url
    assert "c=8192" in url


def test_model_and_quant_change_recomputes_and_updates_url(page):
    page.select_option("#model", "qwen3-8b")
    page.select_option("#quant", "q8_0")
    page.fill("#ctx", "16384")
    page.dispatch_event("#ctx", "change")

    expected = compute_ledger(get_model("qwen3-8b"), "q8_0", 16384).to_dict()
    assert total_text(page) == fmt_gib(expected["ledger_total"]["bytes"])
    assert page.inner_text("#out-kv-pertoken").strip() == "147456 B/token/seq"
    assert "m=qwen3-8b" in page.url and "q=q8_0" in page.url and "c=16384" in page.url
    # provenance switches to the pinned revision
    assert get_model("qwen3-8b")["source"]["revision"][:12] in page.inner_text("#prov")


def test_share_url_reload_restores_parameters(page, server):
    page.goto(f"{server}?m=qwen3-8b&q=q4_k_m&c=40960&b=4&o=2&k=bf16")
    page.wait_for_selector("#out-total")
    assert page.input_value("#model") == "qwen3-8b"
    assert page.input_value("#quant") == "q4_k_m"
    assert page.input_value("#ctx") == "40960"
    assert page.input_value("#batch") == "4"
    assert page.input_value("#overhead") == "2"
    expected = compute_ledger(get_model("qwen3-8b"), "q4_k_m", 40960, 4, kv_dtype="bf16", manual_overhead_gib=2).to_dict()
    assert total_text(page) == fmt_gib(expected["ledger_total"]["bytes"])
    # 2 GiB manual overhead is visible as a manual bound, not an estimate
    assert "manual bound" in page.inner_text(".ledger").lower() or "2.000" in page.inner_text("#out-overhead")


def test_invalid_url_is_sanitised_with_visible_notices(page, server):
    page.goto(f"{server}?m=llama-3-70b&q=fp8&c=999999&b=0&k=fp8&o=abc")
    page.wait_for_selector("#out-total")
    assert not page.is_hidden("#notices")
    notices = page.inner_text("#notices").lower()
    assert "unknown model" in notices
    assert "clamped" in notices
    assert page.input_value("#model") == "qwen2.5-7b-instruct"
    assert page.input_value("#ctx") == "32768"
    assert page.input_value("#batch") == "1"
    assert page.input_value("#kvdtype") == "fp16"
    # page still produced a ledger instead of breaking
    assert "GiB" in total_text(page)


def test_copy_report_puts_honest_text_on_the_clipboard(page):
    page.click("#copy-report")
    page.wait_for_function("document.getElementById('status').textContent.length > 0", timeout=5000)
    assert "copied" in page.inner_text("#status").lower()
    clip = page.evaluate("navigator.clipboard.readText()")
    assert "NOT measured VRAM" in clip
    assert "not a fit guarantee" in clip
    assert "qwen2.5-7b-instruct" in clip


def test_telegram_share_link_is_official_and_carries_state(page):
    href = page.get_attribute("#share-tg", "href")
    assert href.startswith("https://t.me/share/url?url=")
    assert "c%3D8192" in href or "c=8192" in href
    assert "qwen2.5" in href.replace("%2E", ".").replace("%2D", "-") or "qwen" in href.lower()
    assert page.get_attribute("#share-tg", "target") == "_blank"
    assert "noopener" in (page.get_attribute("#share-tg", "rel") or "")


def test_keyboard_reachable_controls(page):
    page.keyboard.press("Tab")
    first = page.evaluate("document.activeElement.tagName + ':' + (document.activeElement.id || document.activeElement.textContent || '').slice(0,20)")
    assert first != "BODY", "first Tab should move focus into the page"
    # model select is reachable by keyboard
    page.focus("#model")
    page.keyboard.press("ArrowDown")
    page.dispatch_event("#model", "change")
    assert page.input_value("#model") == "qwen3-8b"


def test_mobile_360_has_no_horizontal_overflow(browsers, server, screenshots_dir):
    browser = browsers.get("chromium") or next(iter(browsers.values()))
    context = browser.new_context(viewport={"width": 360, "height": 740}, device_scale_factor=2)
    page = context.new_page()
    errors = []
    page.on("pageerror", lambda exc: errors.append(str(exc)))
    page.goto(server)
    page.wait_for_selector("#out-total")
    metrics = page.evaluate(
        "({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth,"
        " bw: document.body.scrollWidth})"
    )
    assert metrics["sw"] <= metrics["cw"] + 1, metrics
    assert metrics["bw"] <= metrics["cw"] + 1, metrics
    assert "GiB" in page.inner_text("#out-total").strip()
    page.screenshot(path=str(screenshots_dir / "mobile-360.png"), full_page=True)
    assert errors == []
    context.close()


def test_desktop_screenshots(browsers, server, screenshots_dir):
    browser = browsers.get("chromium") or next(iter(browsers.values()))
    context = browser.new_context(viewport={"width": 1280, "height": 980}, device_scale_factor=1)
    page = context.new_page()
    page.goto(server)
    page.wait_for_selector("#out-total")
    page.screenshot(path=str(screenshots_dir / "desktop-1280-default.png"), full_page=True)

    page.goto(f"{server}?m=qwen3-8b&q=q4_k_m&c=32768&b=2")
    page.wait_for_selector("#out-total")
    page.screenshot(path=str(screenshots_dir / "desktop-1280-qwen3-32k-batch2.png"), full_page=True)

    page.goto(f"{server}?m=llama-3-70b&q=fp8&c=999999")
    page.wait_for_selector("#notices:not([hidden])")
    page.screenshot(path=str(screenshots_dir / "desktop-1280-invalid-url-notice.png"), full_page=True)
    context.close()

    shots = sorted(p.name for p in screenshots_dir.glob("desktop-1280-*.png"))
    assert len(shots) >= 3, shots