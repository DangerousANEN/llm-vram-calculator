import json, pathlib
from playwright.sync_api import sync_playwright

URL = "https://dangerousanen.github.io/llm-vram-calculator/"
checks = []
def ck(name, ok, extra=""):
    checks.append((name, bool(ok), extra))

with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/usr/bin/chromium")
    pg = b.new_page(viewport={"width": 1280, "height": 900})
    errs = []
    pg.on("console", lambda m: errs.append(m.text) if m.type == "error" else None)
    pg.on("pageerror", lambda e: errs.append(str(e)))
    pg.goto(URL, wait_until="load")
    pg.wait_for_timeout(1200)
    ck("title has Zexkora", "Zexkora" in pg.title(), pg.title())
    ck("4 models in select", pg.eval_on_selector("#model", "e=>e.options.length") == 4)
    total = pg.inner_text("#out-total")
    ck("total rendered", "GiB" in total, total)
    # change model to qwen3-14b and verify the ledger follows
    pg.select_option("#model", "qwen3-14b")
    pg.wait_for_timeout(400)
    ck("model switch renames ledger", "Qwen3 14B" in pg.inner_text("#results-live"), pg.inner_text("#results-live"))
    # context slider to max
    mx = pg.eval_on_selector("#ctxRange", "e=>e.max")
    ck("ctx max from config", mx and int(mx) > 0, mx)
    # invalid context -> refusal notice, not a silent number
    pg.fill("#ctx", "999999")
    pg.dispatch_event("#ctx", "change")
    pg.wait_for_timeout(400)
    notices = pg.inner_text("#notices")
    ck("out-of-range ctx notices", "max" in notices.lower() or "context" in notices.lower(), notices[:120])
    # share link carries state
    pg.fill("#ctx", "8192")
    pg.dispatch_event("#ctx", "change")
    pg.wait_for_timeout(400)
    href = pg.get_attribute("#share-tg", "href")
    ck("telegram share official", href and href.startswith("https://t.me/share/url?"), href[:60])
    ck("share carries params", href and "c%3D8192" in href, href[:120])
    # 360px no overflow
    pg.set_viewport_size({"width": 360, "height": 800})
    pg.wait_for_timeout(500)
    sw = pg.evaluate("document.documentElement.scrollWidth")
    ck("360px no horizontal overflow", sw <= 360, f"scrollWidth={sw}")
    pg.screenshot(path="/workspace/validation/shots_20261118/360.png", full_page=True)
    pg.set_viewport_size({"width": 1280, "height": 900})
    pg.wait_for_timeout(300)
    pg.screenshot(path="/workspace/validation/shots_20261118/desktop.png", full_page=True)
    ck("0 console errors", len(errs) == 0, "; ".join(errs[:3]))
    b.close()

for n, ok, extra in checks:
    print(("PASS " if ok else "FAIL ") + n + ("  [" + str(extra) + "]" if extra else ""))
fails = [c for c in checks if not c[1]]
print(f"\n{len(checks)-len(fails)}/{len(checks)} passed")
raise SystemExit(1 if fails else 0)
