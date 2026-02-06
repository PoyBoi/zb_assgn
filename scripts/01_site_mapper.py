import asyncio
import json
import re
import time
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

ROOT_URL = "https://help.zipboard.co"
OUT_JSON = "assignments/zb_assgn/data/site_map.json"

MAX_PAGES = 2000          # safety cap
PAGE_TIMEOUT = 15000     # ms
CRAWL_DELAY = 0.05       # seconds
PERSIST_EVERY = 10

def normalize(url: str) -> str:
    """Drop query + fragment, normalize trailing slash."""
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    path = re.sub(r"/{2,}", "/", parsed.path)
    norm = f"{scheme}://{netloc}{path}"
    if norm.endswith("/") and len(path) > 1:
        norm = norm[:-1]
    return norm

def is_internal(url: str) -> bool:
    try:
        return urlparse(url).netloc == urlparse(ROOT_URL).netloc
    except Exception:
        return False

def persist(site_map: dict):
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(site_map, f, indent=2, ensure_ascii=False)

async def extract_links(page, current_url):
    html = await page.content()
    soup = BeautifulSoup(html, "lxml")
    links = set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()

        if href.startswith(("mailto:", "tel:", "javascript:")):
            continue

        full = urljoin(current_url, href)
        if is_internal(full):
            links.add(normalize(full))

    return sorted(links)

async def crawl():
    visited = set()
    site_map = {}
    stack = [normalize(ROOT_URL)]
    pages_processed = 0

    print(f"\n🚀 Starting Playwright DFS crawl from: {ROOT_URL}\n", flush=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        while stack:
            current = stack.pop()

            if current in visited:
                print(f"↺ Already visited: {current}", flush=True)
                continue

            if pages_processed >= MAX_PAGES:
                print(f"\n⚠ Reached MAX_PAGES={MAX_PAGES}. Stopping.\n", flush=True)
                break

            print(f"→ Visiting: {current}", flush=True)

            try:
                await page.goto(current, timeout=PAGE_TIMEOUT, wait_until="networkidle")
            except Exception as e:
                print(f"  ✗ Navigation failed: {e}", flush=True)
                visited.add(current)
                site_map[current] = []
                continue

            try:
                children = await extract_links(page, current)
            except Exception as e:
                print(f"  ✗ Link extraction failed: {e}", flush=True)
                children = []

            site_map[current] = children
            visited.add(current)
            pages_processed += 1

            if children:
                for c in children:
                    print(f"  ├─ {c}", flush=True)
            else:
                print("  └─ (no internal links)", flush=True)

            # DFS: push children in reverse so first link is explored next
            for child in reversed(children):
                if child not in visited:
                    stack.append(child)

            if pages_processed % PERSIST_EVERY == 0:
                persist(site_map)
                print(f"[info] persisted {pages_processed} pages → {OUT_JSON}", flush=True)

            await asyncio.sleep(CRAWL_DELAY)

        persist(site_map)
        await browser.close()

    print("\n✅ DONE")
    print(f"Pages visited: {len(site_map)}")
    print(f"Output written to: {OUT_JSON}\n")

if __name__ == "__main__":
    asyncio.run(crawl())
