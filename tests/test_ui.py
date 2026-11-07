# -*- coding: utf-8 -*-
"""Headless UI tests (Playwright: chromium + firefox).

Run inside the isolated sandbox only:

    python -m pytest -m ui tests/test_ui.py

Both browsers are REQUIRED: the previous revision silently skipped the whole
Firefox suite, which hid a real ``Unknown permission: clipboard-read`` crash.
If a required browser cannot be launched the module now FAILS instead of
skipping, unless ``UI_ALLOW_MISSING_BROWSERS=1`` is set explicitly (local
dev only - CI must never set it).

Clipboard: chromium grants ``clipboard-read``/``clipboard-write``. Firefox has
no such permission - it would raise ``Unknown permission: clipboard-read`` - so
it is launched with Firefox's own async-clipboard test preference
(``dom.events.testing.asyncClipboard``) which lets *real* ``navigator.clipboard``
reads work without stubbing any production code.

Real browser screenshots are written to ``validation/screenshots`` for both
Chromium and Firefox; they are inspection evidence, never a substitute for the
DOM assertions here.
"""
from __future__ import annotations

import functools
import http.server
import os
import pathlib
import socketserver
import threading

import pytest

pytestmark = pytest.mark.ui

try:
    from playwright.sync_api import sync_playwright  # noqa: E402
except ImportError:  # pragma: no cover - exercised only without playwright
    if os.environ.get("UI_REQUIRE_BROWSERS") == "1":
        raise
    pytest.importorskip("playwright.sync_api", reason="playwright is only installed in the UI sandbox")

from vramcalc.core import compute_ledger  # noqa: E402
from vramcalc.catalog import get_model  # noqa: E402

GIB = 1024 ** 3

REQUIRED_BROWSERS = ("chromium", "firefox")

# Per-browser launch options. Chromium needs the sandbox flags; Firefox needs its
# test-only async clipboard preference (there is no clipboard-read permission).
LAUNCH_OPTIONS = {
    "chromium": {"args": ["--no-sandbox", "--disable-dev-shm-usage"]},
    "firefox": {"firefox_user_prefs": {"dom.events.testing.asyncClipboard": True}},
}
# Chromium-only context permissions; Firefox rejects them outright.
CONTEXT_PERMISSIONS = {"chromium": ["clipboard-read", "clipboard-write"]}

SYSTEM_BROWSERS = {
    "chromium": ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/usr/bin/google-chrome"],
    "firefox": ["/usr/bin/firefox-esr", "/usr/bin/firefox"],
}


def fmt_gib(nbytes: int) -> str:
    v = nbytes / GIB
    if v >= 100:
        return f"{v:.1f} GiB"
    if v >= 10:
        return f"{v:.2f} GiB"
    return f"{v:.3f} GiB"


@pytest.fixture(scope="module")
def server():
    root = pathlib.Path(__file__).resolve().parents[1]
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


def _launch(browser_type, name):
    """Try the Playwright-bundled build first, then a system binary."""
    options = dict(LAUNCH_OPTIONS.get(name, {}))
    try:
        return browser_type.launch(**options)
    except Exception as first:
        last = None
        for candidate in SYSTEM_BROWSERS.get(name, []):
            if not os.path.exists(candidate):
                continue
            try:
                return browser_type.launch(executable_path=candidate, **options)
            except Exception as exc:
                last = exc
        print(f"[ui] {name} unavailable: {first}")
        if last is not None:
            print(f"[ui] {name} system fallback failed: {last}")
        return None


@pytest.fixture(scope="module")
def browsers():
    """Launch every REQUIRED browser; fail (not skip) when one is missing."""
    allow_missing = os.environ.get("UI_ALLOW_MISSING_BROWSERS") == "1"
    with sync_playwright() as pw:
        started: dict = {}
        missing: list = []
        for name in REQUIRED_BROWSERS:
            browser_type = getattr(pw, name, None)
            if browser_type is None:
                missing.append(f"{name}: playwright has no {name} browser type")
                continue
            browser = _launch(browser_type, name)
            if browser is None:
                missing.append(f"{name}: no launchable binary (bundled download and system fallback both failed)")
            else:
                started[name] = browser
        if missing:
            for b in started.values():
                b.close()
            message = "required UI browser(s) unavailable: " + "; ".join(missing)
            if allow_missing:
                pytest.skip(message + " (UI_ALLOW_MISSING_BROWSERS=1)")
            pytest.fail(message + " - install the browser or unset UI_ALLOW_MISSING_BROWSERS")
        try:
            yield started
        finally:
            for b in started.values():
                b.close()


def _new_context(browser, name, **overrides):
    kwargs = {"viewport": {"width": 1280, "height": 900}}
    if name in CONTEXT_PERMISSIONS:
        kwargs["permissions"] = CONTEXT_PERMISSIONS[name]
    kwargs.update(overrides)
    return browser.new_context(**kwargs)


@pytest.fixture(params=list(REQUIRED_BROWSERS))
def page(request, browsers, server):
    context = _new_context(browsers[request.param], request.param)
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


def read_clipboard(page) -> str:
    """Read the *real* clipboard through the production navigator.clipboard path.

    Prefers a direct async read (chromium permission / firefox test pref). If a
    browser refuses the direct read, fall back to the DOM paste path, which still
    exercises the genuine OS clipboard without stubbing production code.
    """
    try:
        return page.evaluate("() => navigator.clipboard.readText()")
    except Exception:
        page.evaluate(
            """() => {
                var d = document.createElement('div');
                d.id = '__clip_probe';
                d.contentEditable = 'true';
                d.style.position = 'fixed';
                d.style.top = '-1000px';
                document.body.appendChild(d);
                d.focus();
            }"""
        )
        page.keyboard.press("Control+V")
        return page.evaluate("() => document.getElementById('__clip_probe').textContent")


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
    # overhead is unknown, so it must NOT be pinned into the share link
    assert "o=0" not in url


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


def test_copy_report_puts_exact_honest_text_on_the_clipboard(page):
    expected = page.evaluate("() => document.getElementById('share-tg').dataset.report")
    page.click("#copy-report")
    page.wait_for_function("document.getElementById('status').textContent.length > 0", timeout=5000)
    assert "copied" in page.inner_text("#status").lower()
    clip = read_clipboard(page)
    # exact byte-for-byte clipboard content, read back through navigator.clipboard
    assert clip == expected
    assert "NOT measured VRAM" in clip
    assert "not a fit guarantee" in clip
    assert "qwen2.5-7b-instruct" in clip


def test_copy_link_puts_the_share_url_on_the_clipboard(page):
    page.click("#copy-link")
    page.wait_for_function("document.getElementById('status').textContent.length > 0", timeout=5000)
    clip = read_clipboard(page)
    assert "m=qwen2.5-7b-instruct" in clip and "c=8192" in clip


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
    browser = browsers["chromium"]
    context = _new_context(browser, "chromium", viewport={"width": 360, "height": 740}, device_scale_factor=2)
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
    browser = browsers["chromium"]
    context = _new_context(browser, "chromium", viewport={"width": 1280, "height": 980}, device_scale_factor=1)
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


def test_firefox_mobile_and_desktop_screenshots(browsers, server, screenshots_dir):
    """Firefox must be exercised visually too - not just launched."""
    assert "firefox" in browsers, "firefox is required for UI evidence"
    browser = browsers["firefox"]

    mobile = _new_context(browser, "firefox", viewport={"width": 360, "height": 740}, device_scale_factor=2)
    mpage = mobile.new_page()
    errors = []
    mpage.on("pageerror", lambda exc: errors.append(str(exc)))
    mpage.goto(server)
    mpage.wait_for_selector("#out-total")
    metrics = mpage.evaluate(
        "({sw: document.documentElement.scrollWidth, cw: document.documentElement.clientWidth,"
        " bw: document.body.scrollWidth})"
    )
    assert metrics["sw"] <= metrics["cw"] + 1, metrics
    assert metrics["bw"] <= metrics["cw"] + 1, metrics
    assert "GiB" in mpage.inner_text("#out-total").strip()
    mpage.screenshot(path=str(screenshots_dir / "firefox-360-mobile.png"), full_page=True)
    assert errors == []
    mobile.close()

    desktop = _new_context(browser, "firefox", viewport={"width": 1280, "height": 980}, device_scale_factor=1)
    dpage = desktop.new_page()
    dpage.goto(server)
    dpage.wait_for_selector("#out-total")
    dpage.screenshot(path=str(screenshots_dir / "firefox-1280-desktop.png"), full_page=True)

    dpage.goto(f"{server}?m=llama-3-70b&q=fp8&c=999999")
    dpage.wait_for_selector("#notices:not([hidden])")
    dpage.screenshot(path=str(screenshots_dir / "firefox-1280-invalid-url-notice.png"), full_page=True)
    desktop.close()

    shots = sorted(p.name for p in screenshots_dir.glob("firefox-*.png"))
    assert len(shots) >= 3, shots
