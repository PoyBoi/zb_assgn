import asyncio
import json
import csv
import re
import os
from urllib.parse import urlparse, urljoin
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

SITE_MAP_FILE = "assignments/zb_assgn/data/site_map.json"
SITE_MAP_DIFF_FILE = "assignments/zb_assgn/data/site_map_diff.json"  # ← NEW
OUT_CSV = "assignments/zb_assgn/data/file_content.csv"

PAGE_TIMEOUT = 15000  # ms
CRAWL_DELAY = 0.05

# ---------------- Utilities ----------------

def load_all_unique_pages(site_map: dict) -> list[str]:
    pages = set(site_map.keys())
    for links in site_map.values():
        pages.update(links)
    return sorted(pages)

def load_incremental_pages(site_map: dict) -> list[str]:
    """
    If site_map_diff.json exists:
      - scrape added pages
      - scrape changed pages
    Else:
      - fallback to full scrape
    """
    if not os.path.exists(SITE_MAP_DIFF_FILE):
        print("[info] No diff file found — running full scrape", flush=True)
        return load_all_unique_pages(site_map)

    try:
        with open(SITE_MAP_DIFF_FILE, "r", encoding="utf-8") as f:
            diff = json.load(f)
    except Exception:
        print("[warn] Failed to read diff file — running full scrape", flush=True)
        return load_all_unique_pages(site_map)

    added = diff.get("added_pages", [])
    changed = [c["url"] for c in diff.get("changed_pages", [])]

    pages = sorted(set(added + changed))

    if not pages:
        print("[info] Diff present but no added/changed pages — nothing to scrape", flush=True)

    return pages

def extract_category(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "General"
    return path.split("/")[0].replace("-", " ").title()

def normalize_date(date_str: str) -> str:
    if not date_str or not isinstance(date_str, str):
        return ""

    date_str = date_str.strip()

    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%d",
        "%d %b %Y",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
    ):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue

    match = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})", date_str)
    if match:
        try:
            return datetime.strptime(match.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
        except Exception:
            pass

    return ""

def extract_last_updated(soup: BeautifulSoup, headers: dict) -> str:
    for name in ("last-modified", "updated", "modified", "lastupdate"):
        meta = soup.find("meta", {"name": re.compile(name, re.I)})
        if meta and meta.get("content"):
            dt = normalize_date(meta["content"])
            if dt:
                return dt

    for prop in ("article:modified_time", "og:updated_time"):
        meta = soup.find("meta", {"property": prop})
        if meta and meta.get("content"):
            dt = normalize_date(meta["content"])
            if dt:
                return dt

    text = soup.get_text(" ", strip=True)
    match = re.search(
        r"(last updated|updated on|last modified)\s*[:\-]?\s*([A-Za-z0-9,\s\-T:Z+]{6,40})",
        text,
        re.I,
    )
    if match:
        dt = normalize_date(match.group(2))
        if dt:
            return dt

    lm = headers.get("last-modified") or headers.get("Last-Modified")
    return normalize_date(lm)

def extract_article_text(soup: BeautifulSoup) -> str:
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s{2,}", " ", text)
    return text

def extract_images(soup: BeautifulSoup, base_url: str) -> list[str]:
    images = set()
    for img in soup.find_all("img"):
        src = img.get("src")
        if src:
            images.add(urljoin(base_url, src))
    return sorted(images)

def extract_youtube_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    yt_links = set()

    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "")
        if "youtube.com" in src or "youtu.be" in src:
            yt_links.add(src)

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "youtube.com" in href or "youtu.be" in href:
            yt_links.add(urljoin(base_url, href))

    return sorted(yt_links)

def detect_screenshots(image_links: list[str]) -> str:
    for img in image_links:
        if "screen" in img.lower():
            return "Yes"
    return "No"

# ---------------- Main extraction ----------------

async def extract_all():
    with open(SITE_MAP_FILE, "r", encoding="utf-8") as f:
        site_map = json.load(f)

    # 🔹 NEW: choose incremental vs full scrape
    pages_to_scrape = load_incremental_pages(site_map)

    if not pages_to_scrape:
        print("[info] No pages to scrape — exiting cleanly")
        return []

    print(f"Scraping {len(pages_to_scrape)} pages\n", flush=True)

    rows = []
    article_counter = 1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        for url in pages_to_scrape:
            print(f"→ Extracting: {url}", flush=True)

            try:
                response = await page.goto(
                    url,
                    timeout=PAGE_TIMEOUT,
                    wait_until="networkidle"
                )
            except Exception as e:
                print(f"  ✗ Failed: {e}", flush=True)
                continue

            html = await page.content()
            soup = BeautifulSoup(html, "lxml")

            title = (soup.title.string or "").strip() if soup.title else ""
            category = extract_category(url)

            article_text = extract_article_text(soup)
            image_links = extract_images(soup, url)
            youtube_links = extract_youtube_links(soup, url)
            has_screenshots = detect_screenshots(image_links)
            last_updated = extract_last_updated(soup, response.headers if response else {})

            rows.append({
                "Article ID": f"KB-{article_counter:03d}",
                "Name": title,
                "Category": category,
                "URL": url,
                "Last Updated": last_updated,
                "Has Screenshots": has_screenshots,
                "YouTube Links": ", ".join(youtube_links),
                "Image Links": ", ".join(image_links),
                "Article Content": article_text,
                "Linked Pages": ", ".join(site_map.get(url, []))
            })

            article_counter += 1
            await asyncio.sleep(CRAWL_DELAY)

        await browser.close()

    return rows

# ---------------- CSV writer ----------------

def write_csv(rows):
    fields = [
        "Article ID",
        "Name",
        "Category",
        "URL",
        "Last Updated",
        "Has Screenshots",
        "YouTube Links",
        "Image Links",
        "Article Content",
        "Linked Pages"
    ]
    
    # NEW CODE
    import sys, csv
    csv.field_size_limit(sys.maxsize) # to cure field larger than field limit error

    existing_urls = set()

    # 1️⃣ Read existing URLs if CSV exists
    if os.path.exists(OUT_CSV):
        with open(OUT_CSV, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("URL"):
                    existing_urls.add(row["URL"])

    # 2️⃣ Append only new rows
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)

        # write header ONLY if file did not exist
        if not existing_urls:
            writer.writeheader()

        appended = 0
        for row in rows:
            url = row.get("URL")
            if url and url not in existing_urls:
                writer.writerow(row)
                existing_urls.add(url)
                appended += 1

    print(f"[info] appended {appended} new rows to {OUT_CSV}")

# ---------------- Entry ----------------

if __name__ == "__main__":
    rows = asyncio.run(extract_all())
    write_csv(rows)
    print(f"\n✅ Done. Wrote {len(rows)} rows to {OUT_CSV}")