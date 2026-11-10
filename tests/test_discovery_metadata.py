# -*- coding: utf-8 -*-
"""Discovery metadata for the single public surface we own (GitHub Pages).

The tool is only reachable through its Pages URL, so the head of the page and the
two crawler files are the whole discovery layer. These tests pin the facts a link
preview or a search engine actually reads, and they refuse invented claims:
the structured data may only describe what the page really is (a free, offline,
client-side calculator), and the image must be a file that ships with the repo.
"""
from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
INDEX = ROOT / "index.html"
TEMPLATE = ROOT / "web" / "template.html"
CANONICAL = "https://dangerousanen.github.io/llm-vram-calculator/"


def _head(html: str) -> str:
    end = html.find("</head>")
    assert end > 0, "page has no </head>"
    return html[:end]


def _meta(html: str, key: str, attr: str = "property") -> str | None:
    m = re.search(
        rf'<meta\s+{attr}="{re.escape(key)}"\s+content="([^"]*)"',
        html,
    )
    if m is None:  # tolerate attribute order: content first
        m = re.search(
            rf'<meta\s+content="([^"]*)"\s+{attr}="{re.escape(key)}"',
            html,
        )
    return m.group(1) if m else None


@pytest.mark.parametrize("source", ["index.html", "web/template.html"])
def test_canonical_and_og_url_agree(source):
    html = _head((ROOT / source).read_text(encoding="utf-8"))
    link = re.search(r'<link\s+rel="canonical"\s+href="([^"]+)"', html)
    assert link, f"{source}: missing <link rel=canonical>"
    assert link.group(1) == CANONICAL
    assert _meta(html, "og:url") == CANONICAL
    assert _meta(html, "og:type") == "website"
    assert _meta(html, "og:site_name") == "Zexkora Hub"


@pytest.mark.parametrize("source", ["index.html", "web/template.html"])
def test_og_image_is_absolute_and_ships_with_the_repo(source):
    html = _head((ROOT / source).read_text(encoding="utf-8"))
    image = _meta(html, "og:image")
    assert image, f"{source}: missing og:image"
    assert image.startswith("https://"), "og:image must be absolute for previews"
    assert image.startswith(CANONICAL), "og:image must live on the canonical surface"
    rel = image[len(CANONICAL):]
    asset = ROOT / rel
    assert asset.is_file(), f"og:image points at a file that is not in the repo: {rel}"
    assert asset.stat().st_size > 10_000, "preview image is too small to be real art"
    assert _meta(html, "og:image:alt"), "og:image needs alt text for accessibility"


@pytest.mark.parametrize("source", ["index.html", "web/template.html"])
def test_twitter_card_metadata_present(source):
    html = _head((ROOT / source).read_text(encoding="utf-8"))
    assert _meta(html, "twitter:card", attr="name") == "summary_large_image"
    assert _meta(html, "twitter:title", attr="name")
    assert _meta(html, "twitter:description", attr="name")
    assert _meta(html, "twitter:image", attr="name")


@pytest.mark.parametrize("source", ["index.html", "web/template.html"])
def test_jsonld_describes_the_real_free_client_side_tool(source):
    html = _head((ROOT / source).read_text(encoding="utf-8"))
    m = re.search(
        r'<script\s+type="application/ld\+json">(.*?)</script>', html, re.S
    )
    assert m, f"{source}: missing JSON-LD block"
    data = json.loads(m.group(1))  # must be valid JSON, not a comment
    assert data["@context"] == "https://schema.org"
    assert data["@type"] == "SoftwareApplication"
    assert "Zexkora Hub" in data["name"]
    assert data["url"] == CANONICAL
    assert data["applicationCategory"] == "DeveloperApplication"
    assert data["operatingSystem"] == "Any"
    assert data["inLanguage"] == "ru"
    # free and client-side are true; anything about pricing or telemetry is not.
    assert data["isAccessibleForFree"] is True
    assert "offers" not in data and "aggregateRating" not in data
    assert data["codeRepository"] == "https://github.com/DangerousANEN/llm-vram-calculator"


def test_robots_txt_allows_crawling_and_points_at_the_sitemap():
    robots = (ROOT / "robots.txt").read_text(encoding="utf-8")
    lowered = robots.lower()
    assert "user-agent: *" in lowered
    assert re.search(r"^allow:\s*/\s*$", lowered, re.M)
    assert "disallow: /\n" not in lowered and not lowered.rstrip().endswith("disallow: /")
    assert f"sitemap: {CANONICAL}sitemap.xml" in lowered


def test_sitemap_lists_exactly_the_canonical_page():
    raw = (ROOT / "sitemap.xml").read_text(encoding="utf-8")
    root = ET.fromstring(raw)
    ns = {"sm": "http://www.sitemaps.org/schemas/sitemap/0.9"}
    urls = [u.findtext("sm:loc", namespaces=ns) for u in root.findall("sm:url", ns)]
    assert urls == [CANONICAL]
    lastmod = root.findtext("sm:url/sm:lastmod", namespaces=ns)
    assert lastmod and re.fullmatch(r"\d{4}-\d{2}-\d{2}", lastmod)


def test_metadata_adds_no_runtime_fetch_or_third_party_script():
    html = INDEX.read_text(encoding="utf-8")
    assert not re.search(r'<script[^>]+src="https?://', html)
    assert "fetch(" not in html and "XMLHttpRequest" not in html
    assert "googletagmanager" not in html and "mc.yandex" not in html
