#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate the static ``index.html`` from the template + the pinned catalogue.

The page must work from GitHub Pages and from ``file://`` with no fetch and no
build step at runtime, so the catalogue, the state library, the controller and
the styles are inlined into one self-contained HTML file.

Single source of truth: ``vramcalc/data/models.json``. If the catalogue changes,
run this script and commit the regenerated ``index.html``. CI runs ``--check``
and fails when the committed page is stale.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "web" / "template.html"
STYLES = ROOT / "web" / "styles.css"
STATE_JS = ROOT / "web" / "state.js"
APP_JS = ROOT / "web" / "app.js"
DATA = ROOT / "vramcalc" / "data" / "models.json"
OUT = ROOT / "index.html"

MARKERS = {
    "/*__STYLES__*/": STYLES,
    "/*__STATE_JS__*/": STATE_JS,
    "/*__APP_JS__*/": APP_JS,
}


def _safe_json(text: str) -> str:
    """Inline JSON that cannot terminate the surrounding <script> block."""
    return (
        text.replace("</", "<\\/")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def build() -> str:
    html = TEMPLATE.read_text(encoding="utf-8")
    catalog_raw = json.loads(DATA.read_text(encoding="utf-8"))

    # Only what the page needs: catalogue identity, geometry, weights, source.
    slim = {
        "schema_version": catalog_raw["schema_version"],
        "generated_at": catalog_raw["generated_at"],
        "data_source": catalog_raw["data_source"],
        "models": [
            {
                "id": m["id"],
                "display_name": m["display_name"],
                "short_name": m.get("short_name", m["id"]),
                "license": m["license"],
                "source": {
                    k: m["source"].get(k)
                    for k in (
                        "hub",
                        "repo",
                        "revision",
                        "revision_url",
                        "config_sha256",
                        "config_url",
                        "gguf_repo",
                        "gguf_revision",
                        "gguf_revision_url",
                    )
                },
                "config": m["config"],
                "weights": m["weights"],
            }
            for m in catalog_raw["models"]
        ],
    }

    html = html.replace("/*__CATALOG__*/ null", _safe_json(json.dumps(slim, ensure_ascii=False, separators=(",", ":"))), 1)
    for marker, path in MARKERS.items():
        replacement = path.read_text(encoding="utf-8")
        if marker not in html:
            raise SystemExit(f"template is missing marker {marker}")
        html = html.replace(marker, replacement, 1)

    leftover = [m for m in list(MARKERS) + ["__CATALOG__"] if m in html]
    if leftover:
        raise SystemExit(f"unreplaced template markers: {leftover}")
    return html


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if index.html is not the current build")
    args = ap.parse_args(argv)

    html = build()
    if args.check:
        current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
        if current != html:
            print("OUT OF DATE: index.html differs from the template + catalogue build", file=sys.stderr)
            print("run: python scripts/build_web.py", file=sys.stderr)
            return 1
        print(f"OK: {OUT.name} is up to date ({len(html)} bytes)")
        return 0

    OUT.write_text(html, encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(html)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())