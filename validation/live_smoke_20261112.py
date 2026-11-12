# -*- coding: utf-8 -*-
"""End-to-end smoke against the LIVE GitHub Pages calculator (sandbox only)."""
from __future__ import annotations

import json

from playwright.sync_api import sync_playwright

BASE = "https://dangerousanen.github.io/llm-vram-calculator/"

with sync_playwright() as pw:
    browser = pw.chromium.launch(executable_path="/usr/bin/chromium", args=["--no-sandbox", "--disable-dev-shm-usage"])
    page = browser.new_page(viewport={"width": 1280, "height": 900})
    errors: list = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.goto(BASE + "?m=qwen3-14b&q=q4_k_m&c=16384&b=1", wait_until="load")
    page.wait_for_selector("#out-total")
    page.wait_for_timeout(400)
    result = {
        "url": page.url,
        "title": page.title(),
        "model_options": page.locator("#model option").count(),
        "model": page.input_value("#model"),
        "quant": page.input_value("#quant"),
        "ctx": page.input_value("#ctx"),
        "kv_per_token": page.inner_text("#out-kv-pertoken").strip(),
        "weights": page.inner_text("#out-weights").strip().replace("\n", " | "),
        "total": page.inner_text("#out-total").strip(),
        "overhead_row": page.inner_text("#out-overhead").strip().replace("\n", " | "),
        "provenance_has_revision": "40c069824f42" in page.inner_text("#prov"),
        "notice_is_not_fit_verdict": "НЕ измеренная VRAM" in page.inner_text(".ledger .total"),
        "doc_scroll_width": page.evaluate("() => document.documentElement.scrollWidth"),
        "console_errors": errors,
    }
    # an unsupported model id must degrade to a notice, not break the page
    page.goto(BASE + "?m=deepseek-v3&q=q4_k_m&c=8192&b=1", wait_until="load")
    page.wait_for_selector("#out-total")
    result["bad_url_notice"] = page.inner_text("#notices").strip()[:160]
    result["bad_url_model"] = page.input_value("#model")
    page.screenshot(path="/workspace/validation/screenshots/live-1280-qwen3-14b-16k.png", full_page=True)
    browser.close()

print(json.dumps(result, ensure_ascii=False, indent=2))
