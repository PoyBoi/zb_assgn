#!/usr/bin/env python3
import asyncio
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

# ---------------- CONFIG ----------------
ROOT_URL = "https://help.zipboard.co"
OUT_JSON = "assignments/zb_assgn/data/site_map.json"

MAX_PAGES = 2000          # safety cap
PAGE_TIMEOUT = 15000      # ms
CRAWL_DELAY = 0.05        # seconds
PERSIST_EVERY = 10

SNAPSHOT_DIR = Path("assignments/zb_assgn/data/snapshots")
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

DIFF_JSON = str(Path(OUT_JSON).with_name(Path(OUT_JSON).stem + "_diff.json"))

# ---------------- Utilities ----------------
def normalize(url: str) -> str:
    """Drop query + fragment, normalize trailing slash."""
    parsed = urlparse(url)
    scheme = parsed.scheme or "https"
    netloc = parsed.netloc
    path = re.sub(r"/{2,}", "/", parsed.path or "/")
    norm = f"{scheme}://{netloc}{path}"
    if norm.endswith("/") and len(path) > 1:
        norm = norm[:-1]
    return norm

def is_internal(url: str) -> bool:
    try:
        return urlparse(url).netloc == urlparse(ROOT_URL).netloc
    except Exception:
        return False

def load_json(path: str) -> Dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

def persist(site_map: dict):
    Path(OUT_JSON).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(site_map, f, indent=2, ensure_ascii=False)

def persist_diff(diff: dict):
    try:
        with open(DIFF_JSON, "w", encoding="utf-8") as f:
            json.dump(diff, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[warn] could not write diff file: {e}", flush=True)

def archive_prev(prev_map: dict):
    if not prev_map:
        return
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    name = Path(OUT_JSON).with_suffix("").name
    out_path = SNAPSHOT_DIR / f"{name}_snapshot_{ts}.json"
    try:
        out_path.write_text(json.dumps(prev_map, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[info] archived previous site_map → {out_path}", flush=True)
    except Exception as e:
        print(f"[warn] failed to archive previous site_map: {e}", flush=True)

# ---------------- Link extraction ----------------
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

# ---------------- Diff logic ----------------
def diff_site_maps(old_map: Dict[str, List[str]], new_map: Dict[str, List[str]]) -> Dict:
    old_keys: Set[str] = set(old_map.keys())
    new_keys: Set[str] = set(new_map.keys())

    added_pages = sorted(new_keys - old_keys)
    removed_pages = sorted(old_keys - new_keys)

    changed_pages = []
    for url in sorted(old_keys & new_keys):
        old_links = set(old_map.get(url) or [])
        new_links = set(new_map.get(url) or [])
        if old_links != new_links:
            links_added = sorted(new_links - old_links)
            links_removed = sorted(old_links - new_links)
            changed_pages.append({
                "url": url,
                "links_added": links_added,
                "links_removed": links_removed,
                "old_links_count": len(old_links),
                "new_links_count": len(new_links),
            })

    return {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "summary": {
            "added_pages_count": len(added_pages),
            "removed_pages_count": len(removed_pages),
            "changed_pages_count": len(changed_pages),
        },
        "added_pages": added_pages,
        "removed_pages": removed_pages,
        "changed_pages": changed_pages,
    }

# ---------------- Crawler ----------------
async def crawl():
    # load previous state (for diff)
    previous_map = load_json(OUT_JSON)

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
                pages_processed += 1
                # persist periodically as before
                if pages_processed % PERSIST_EVERY == 0:
                    persist(site_map)
                    print(f"[info] persisted {pages_processed} pages → {OUT_JSON}", flush=True)
                await asyncio.sleep(CRAWL_DELAY)
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

        # finished crawl — compute diff vs previous
        print("\n[info] crawl finished; computing diff against previous snapshot...", flush=True)
        diff = diff_site_maps(previous_map, site_map)

        # archive previous map (if present) and persist diff and new map
        if previous_map:
            archive_prev(previous_map)

        persist(site_map)
        persist_diff(diff)

        await browser.close()

    # print summary and short samples
    print("\n✅ DONE")
    print(f"Pages visited: {len(site_map)}")
    print(f"Output written to: {OUT_JSON}")
    print(f"Diff written to: {DIFF_JSON}\n")

    # compact human summary
    s = diff.get("summary", {})
    print(f"Summary — added: {s.get('added_pages_count',0)}, removed: {s.get('removed_pages_count',0)}, changed: {s.get('changed_pages_count',0)}\n", flush=True)

    # print up to 8 sample urls for each category (keeps logs compact)
    def _print_samples(key: str, items: List[str]):
        if not items:
            return
        print(f"{key} (sample {min(8,len(items))}):")
        for u in items[:8]:
            print(f"  - {u}")
        print("", flush=True)

    _print_samples("Added pages", diff.get("added_pages", []))
    _print_samples("Removed pages", diff.get("removed_pages", []))
    if diff.get("changed_pages"):
        print("Changed pages (sample up to 8):")
        for ch in diff["changed_pages"][:8]:
            print(f"  - {ch['url']} (+{len(ch['links_added'])} / -{len(ch['links_removed'])})")
        print("", flush=True)

if __name__ == "__main__":
    asyncio.run(crawl())