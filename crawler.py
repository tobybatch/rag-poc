#!/usr/bin/env python3
"""
Crawls a site (default: playbook.dxw.com) and saves each page as clean text to data/raw/.

Run this on a machine with normal internet access (your laptop, etc).
It discovers pages by following internal links from the homepage (the
Playbook's nav links out to every doc), so re-running it later will pick
up new/removed pages automatically.

Usage:
    uv run crawler.py                                   # crawl the whole site
    uv run crawler.py --max-pages 20                    # crawl only the first 20 pages (for a quick test)
    uv run crawler.py --base-url https://example.com    # crawl a different site
    uv run crawler.py --xpath "//main//article"         # only save text from this part of each page
"""
import argparse
import json
import os
import re
import time
from html import escape
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from lxml import etree
from lxml import html as lxml_html
from tqdm import tqdm

DEFAULT_BASE_URL = "https://playbook.dxw.com"
DEFAULT_XPATH = "/html/body/main"
RAW_DIR = os.path.join(os.path.dirname(__file__), "data", "raw")
USER_AGENT = "playbook-rag-crawler/1.0 (personal use; contact: internal dxw tool)"
REQUEST_DELAY_SECONDS = 0.5  # be polite - don't hammer the site


def is_internal_doc_link(href: str, base_url: str) -> bool:
    if not href:
        return False
    joined = urljoin(base_url, href)
    parsed = urlparse(joined)
    if parsed.netloc != urlparse(base_url).netloc:
        return False
    # The Playbook (an Outline wiki) publishes pages under /doc/<slug>-<id>
    return parsed.path.startswith("/doc/")


def normalise(url: str, base_url: str) -> str:
    joined = urljoin(base_url, url)
    parsed = urlparse(joined)
    return f"{parsed.scheme}://{parsed.netloc}{parsed.path}"


def fetch(session: requests.Session, url: str) -> requests.Response | None:
    try:
        resp = session.get(url, timeout=20)
        resp.raise_for_status()
        return resp
    except requests.RequestException as e:
        print(f"  ! failed to fetch {url}: {e}")
        return None


def select_xpath(html: str, xpath: etree.XPath) -> str | None:
    """Return the HTML of every element matching `xpath`, joined together,
    or None if nothing matched. Text/attribute results are wrapped in <p>."""
    tree = lxml_html.fromstring(html)
    parts = []
    for match in xpath(tree):
        if isinstance(match, etree._Element):
            parts.append(lxml_html.tostring(match, encoding="unicode"))
        elif str(match).strip():
            parts.append(f"<p>{escape(str(match))}</p>")
    return "\n".join(parts) or None


def extract_page(html: str, url: str, xpath: etree.XPath | None = None) -> dict | None:
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("h1")
    title = title_tag.get_text(strip=True) if title_tag else (soup.title.get_text(strip=True) if soup.title else url)

    if xpath is not None:
        # Restrict extraction to the part of the page the XPath selects.
        selected = select_xpath(html, xpath)
        if selected is None:
            return None
        soup = BeautifulSoup(f"<html><body>{selected}</body></html>", "lxml")

    # Drop elements that are never part of the actual document content.
    for selector in ["nav", "header", "footer", "script", "style", "[role='navigation']"]:
        for tag in soup.select(selector):
            tag.decompose()

    content_root = (
        soup.find("main")
        or soup.find(attrs={"class": re.compile(r"(document|editor|content)", re.I)})
        or soup.body
    )
    if content_root is None:
        return None

    text = content_root.get_text("\n", strip=True)
    text = re.sub(r"\n{3,}", "\n\n", text)

    if len(text) < 40:
        # Likely a nav-only or empty page - skip it.
        return None

    return {"url": url, "title": title, "text": text}


def discover_links(html: str, base_url: str) -> set[str]:
    soup = BeautifulSoup(html, "lxml")
    links = set()
    for a in soup.find_all("a", href=True):
        if is_internal_doc_link(a["href"], base_url):
            links.add(normalise(a["href"], base_url))
    return links


def crawl(base_url: str = DEFAULT_BASE_URL, max_pages: int | None = None, xpath: str | None = None) -> list[dict]:
    """Crawl `base_url`. If `xpath` is given, only the text inside the
    elements it matches is saved for each page (link discovery still uses
    the whole page, so navigation menus keep working)."""
    compiled_xpath = etree.XPath(xpath) if xpath else None
    base_url = base_url.rstrip("/")
    os.makedirs(RAW_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": USER_AGENT})

    to_visit = {base_url}
    visited = set()
    pages = []

    with tqdm(desc="crawling", unit="page") as pbar:
        while to_visit:
            if max_pages is not None and len(visited) >= max_pages:
                break
            url = to_visit.pop()
            if url in visited:
                continue
            visited.add(url)

            resp = fetch(session, url)
            time.sleep(REQUEST_DELAY_SECONDS)
            if resp is None:
                continue

            # Always harvest links, even from the homepage, so we discover
            # every doc even though the homepage itself isn't saved as a doc.
            to_visit |= discover_links(resp.text, base_url) - visited

            if url != base_url:
                page = extract_page(resp.text, url, compiled_xpath)
                if page:
                    pages.append(page)
                    slug = urlparse(url).path.rstrip("/").split("/")[-1] or "index"
                    out_path = os.path.join(RAW_DIR, f"{slug}.json")
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(page, f, ensure_ascii=False, indent=2)

            pbar.update(1)
            pbar.set_postfix(queued=len(to_visit), saved=len(pages))

    return pages


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help=f"Root URL of the site to crawl (default: {DEFAULT_BASE_URL})")
    parser.add_argument("--max-pages", type=int, default=None, help="Stop after crawling this many pages (useful for a quick test)")
    parser.add_argument("--xpath", default=None, help="Only save text from the elements this XPath matches on each page, e.g. \"//main\" (pages with no match are skipped)")
    args = parser.parse_args()

    if args.xpath:
        xpath = args.xpath
    else:
        xpath = DEFAULT_XPATH
    try:
        etree.XPath(xpath)
    except etree.XPathSyntaxError as e:
        parser.error(f"invalid --xpath {xpath!r}: {e}")

    pages = crawl(base_url=args.base_url, max_pages=args.max_pages, xpath=xpath)
    print(f"\nDone. Saved {len(pages)} pages to {os.path.abspath(RAW_DIR)}")
