import asyncio
import json
import csv
import re
from urllib.parse import urlparse
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

SITE_MAP_FILE = "assignments/zb_assgn/backup/site_map.json"
OUT_CSV = "assignments/zb_assgn/backup/file_content.csv"

PAGE_TIMEOUT = 15000  # ms
CRAWL_DELAY = 0.05

# ---------------- Utilities ----------------

def load_all_unique_pages(site_map: dict) -> list[str]:
    pages = set(site_map.keys())
    for links in site_map.values():
        pages.update(links)
    return sorted(pages)

def extract_category(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "General"
    return path.split("/")[0].replace("-", " ").title()

def normalize_date(date_str: str) -> str:
    """Try to normalize common HTTP / meta date formats."""
    if not date_str:
        return ""
    for fmt in (
        "%a, %d %b %Y %H:%M:%S %Z",
        "%Y-%m-%d",
        "%d %b %Y",
    ):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except Exception:
            continue
    return ""

def detect_screenshots(soup: BeautifulSoup) -> str:
    for img in soup.find_all("img"):
        src = (img.get("src") or "").lower()
        alt = (img.get("alt") or "").lower()
        if "screen" in src or "screen" in alt:
            return "Yes"
    return "No"

def extract_last_updated(soup: BeautifulSoup, headers: dict) -> str:
    # 1) meta tags (best effort)
    meta = soup.find("meta", {"name": re.compile("modified|updated", re.I)})
    if meta and meta.get("content"):
        return normalize_date(meta["content"])

    # 2) visible text patterns
    text = soup.get_text(" ", strip=True)
    match = re.search(r"last updated[:\s]+([A-Za-z0-9, ]{8,})", text, re.I)
    if match:
        return normalize_date(match.group(1))

    # 3) HTTP headers fallback
    return normalize_date(headers.get("Last-Modified", ""))

# ---------------- Main extraction ----------------

async def extract_all():
    with open(SITE_MAP_FILE, "r", encoding="utf-8") as f:
        site_map = json.load(f)

    all_pages = load_all_unique_pages(site_map)

    print(f"Found {len(all_pages)} unique pages to extract\n", flush=True)

    rows = []
    article_counter = 1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        for url in all_pages:
            print(f"→ Extracting: {url}", flush=True)

            try:
                response = await page.goto(
                    url,
                    timeout=PAGE_TIMEOUT,
                    wait_until="networkidle"
                )
            except Exception as e:
                print(f"  ✗ Failed to load: {e}", flush=True)
                continue

            html = await page.content()
            soup = BeautifulSoup(html, "lxml")

            title = (soup.title.string or "").strip() if soup.title else ""
            category = extract_category(url)
            has_screenshots = detect_screenshots(soup)
            last_updated = extract_last_updated(soup, response.headers if response else {})

            rows.append({
                "Article ID": f"KB-{article_counter:03d}",
                "Name": title,
                "Category": category,
                "URL": url,
                "Last Updated": last_updated,
                "Has Screenshots": has_screenshots,
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
        "Linked Pages"
    ]

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

# ---------------- Entry ----------------

if __name__ == "__main__":
    rows = asyncio.run(extract_all())
    write_csv(rows)
    print(f"\n✅ Done. Wrote {len(rows)} rows to {OUT_CSV}")
