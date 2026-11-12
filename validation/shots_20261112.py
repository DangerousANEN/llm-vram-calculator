# -*- coding: utf-8 -*-
"""Screenshot the Qwen3-14B / Qwen3-4B catalogue additions (run inside the sandbox only)."""
from __future__ import annotations

import functools
import http.server
import pathlib
import socketserver
import threading

from playwright.sync_api import sync_playwright

ROOT = pathlib.Path("/workspace")
OUT = ROOT / "validation" / "screenshots"


def serve() -> str:
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(ROOT))
    handler.log_message = lambda *a, **k: None

    class Quiet(socketserver.TCPServer):
        allow_reuse_address = True

    httpd = Quiet(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/index.html"


URL = serve()
CASES = [
    ("qwen3-14b", "q4_k_m", 16384, 1280, 900, "desktop-1280-qwen3-14b-16k.png"),
    ("qwen3-4b", "q4_k_m", 8192, 390, 844, "mobile-390-qwen3-4b-8k.png"),
]

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox", "--disable-dev-shm-usage"])
    for model, quant, ctx, w, h, name in CASES:
        page = browser.new_page(viewport={"width": w, "height": h})
        page.goto(f"{URL}?m={model}&q={quant}&c={ctx}&b=1")
        page.wait_for_selector("#out-total")
        page.wait_for_timeout(300)
        path = OUT / name
        page.screenshot(path=str(path), full_page=True)
        info = {
            "model_option": page.input_value("#model"),
            "quant_option": page.input_value("#quant"),
            "ctx": page.input_value("#ctx"),
            "options": page.locator("#model option").count(),
            "kv_per_token": page.inner_text("#out-kv-pertoken").strip(),
            "total": page.inner_text("#out-total").strip(),
            "doc_width": page.evaluate("() => document.documentElement.scrollWidth"),
            "viewport": w,
        }
        print(name, info)
        page.close()
    browser.close()
