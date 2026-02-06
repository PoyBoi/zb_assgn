import asyncio
import json
import csv
import re
import argparse
from urllib.parse import urlparse, urljoin
from datetime import datetime

from bs4 import BeautifulSoup
from playwright.async_api import async_playwright

PAGE_TIMEOUT = 30_000  # ms
CRAWL_DELAY = 0.05


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
    # Try to parse heuristically with regex (e.g. "January 2, 2023")
    m = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4})", date_str)
    if m:
        try:
            return datetime.strptime(m.group(1), "%B %d, %Y").strftime("%Y-%m-%d")
        except Exception:
            pass
    return ""


def extract_last_updated(soup: BeautifulSoup, headers: dict) -> str:
    # meta tags
    for name in ("last-modified", "updated", "modified", "lastupdate", "article:modified_time"):
        meta = soup.find("meta", {"name": re.compile(name, re.I)})
        if meta and meta.get("content"):
            dt = normalize_date(meta["content"])
            if dt:
                return dt
    # property attribute variants
    for prop in ("article:modified_time", "og:updated_time"):
        meta = soup.find("meta", {"property": prop})
        if meta and meta.get("content"):
            dt = normalize_date(meta["content"])
            if dt:
                return dt

    # Search visible text for typical phrases
    text = soup.get_text(" ", strip=True)
    match = re.search(r"(?:last updated|updated on|last modified|updated:)\s*[:\-]?\s*([A-Za-z0-9,\s-]{6,40})", text, re.I)
    if match:
        dt = normalize_date(match.group(1))
        if dt:
            return dt

    # headers fallback
    lm = ""
    try:
        lm = headers.get("last-modified", "") or headers.get("Last-Modified", "")
    except Exception:
        lm = ""
    return normalize_date(lm)


def extract_article_text(soup: BeautifulSoup) -> str:
    # remove scripts/styles
    for tag in soup(["script", "style", "noscript", "svg"]):
        tag.decompose()
    # Try to isolate main content: look for article, main, .content, .article
    selectors = ["article", "main", "[role=main]", ".article", ".post", ".content", ".kb-article"]
    for sel in selectors:
        block = soup.select_one(sel)
        if block:
            text = block.get_text(" ", strip=True)
            text = re.sub(r"\s{2,}", " ", text)
            if len(text) > 120:
                return text
    # fallback: entire page text
    text = soup.get_text(" ", strip=True)
    return re.sub(r"\s{2,}", " ", text)


def extract_images(soup: BeautifulSoup, base_url: str) -> list[str]:
    images = set()
    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src")
        if not src:
            continue
        images.add(urljoin(base_url, src))
    return sorted(images)


def extract_youtube_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    yt_links = set()
    for iframe in soup.find_all("iframe"):
        src = iframe.get("src", "")
        if "youtube.com" in src or "youtu.be" in src:
            yt_links.add(urljoin(base_url, src))
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "youtube.com" in href or "youtu.be" in href:
            yt_links.add(urljoin(base_url, href))
    return sorted(yt_links)


def detect_screenshots(image_links: list[str]) -> str:
    for img in image_links:
        if "screen" in img.lower() or "screenshot" in img.lower():
            return "Yes"
    return "No"


def extract_internal_links(soup: BeautifulSoup, base_url: str) -> list[str]:
    parsed = urlparse(base_url)
    base_netloc = parsed.netloc
    links = set()
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith("#") or href.lower().startswith("mailto:") or href.lower().startswith("tel:"):
            continue
        full = urljoin(base_url, href)
        p = urlparse(full)
        if p.netloc == base_netloc:
            links.add(full.split("#")[0])  # remove fragments
    return sorted(links)


async def scrape_url(url: str) -> dict:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            response = await page.goto(url, timeout=PAGE_TIMEOUT, wait_until="networkidle")
        except Exception as e:
            # try a more tolerant navigation
            try:
                response = await page.goto(url, timeout=PAGE_TIMEOUT)
                await page.wait_for_load_state("networkidle", timeout=PAGE_TIMEOUT)
            except Exception as e2:
                await browser.close()
                raise RuntimeError(f"Navigation failed: {e} / {e2}") from e2

        # small delay to let JS paint if needed
        await asyncio.sleep(CRAWL_DELAY)
        html = await page.content()
        headers = {}
        try:
            headers = dict(response.headers) if response else {}
        except Exception:
            headers = {}

        soup = BeautifulSoup(html, "lxml")
        title = (soup.title.string or "").strip() if soup.title else ""
        category = extract_category(url)
        article_text = extract_article_text(soup)
        image_links = extract_images(soup, url)
        youtube_links = extract_youtube_links(soup, url)
        has_screenshots = detect_screenshots(image_links)
        last_updated = extract_last_updated(soup, headers)
        internal_links = extract_internal_links(soup, url)

        # gather meta description and H1s for extra context
        meta_desc = ""
        md = soup.find("meta", {"name": "description"})
        if md and md.get("content"):
            meta_desc = md["content"].strip()
        h1s = [h.get_text(" ", strip=True) for h in soup.find_all("h1")]

        await browser.close()

        result = {
            "URL": url,
            "Name": title,
            "Meta Description": meta_desc,
            "H1s": h1s,
            "Category": category,
            "Last Updated": last_updated,
            "Has Screenshots": has_screenshots,
            "YouTube Links": youtube_links,
            "Image Links": image_links,
            "Article Content": article_text,
            "Internal Links": internal_links,
            "Response Headers": headers,
        }
        return result


def write_csv_row(out_csv: str, row: dict):
    fields = [
        "URL",
        "Name",
        "Category",
        "Last Updated",
        "Has Screenshots",
        "YouTube Links",
        "Image Links",
        "Article Content",
        "Internal Links",
    ]
    # normalize lists into comma-joined strings
    csv_row = {k: (", ".join(v) if isinstance(v, list) else v) for k, v in row.items() if k in fields}
    # write header if file doesn't exist / is empty
    write_header = True
    try:
        with open(out_csv, "r", encoding="utf-8") as f:
            if f.read().strip():
                write_header = False
    except FileNotFoundError:
        write_header = True

    with open(out_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(csv_row)


def parse_args():
    parser = argparse.ArgumentParser(description="One-shot web scraper (Playwright + BS4)")
    parser.add_argument("--url", "-u", default="https://help.zipboard.co/article/63-how-do-i-integrate-zipboard-with-jira",
                        help="URL to scrape (default: zipboard jira integration article)")
    parser.add_argument("--csv", "-c", default=None, help="Optional CSV output file to append one row")
    parser.add_argument("--no-print-headers", dest="print_headers", action="store_false",
                        help="Don't print response headers (they can be large).")
    return parser.parse_args()


def main():
    args = parse_args()
    url = args.url

    try:
        result = asyncio.run(scrape_url(url))
    except Exception as e:
        print(f"ERROR: {e}")
        return

    # Optionally remove headers before printing if the user disabled printing them
    if not args.print_headers:
        result_copy = dict(result)
        result_copy.pop("Response Headers", None)
    else:
        result_copy = result

    # Pretty-print full JSON to stdout
    print("\n--- SCRAPED RESULT (JSON) ---\n")
    print(json.dumps(result_copy, indent=2, ensure_ascii=False))

    # Also print a short human-readable summary
    print("\n--- SUMMARY ---\n")
    print(f"URL: {result['URL']}")
    print(f"Title: {result['Name']}")
    print(f"Category: {result['Category']}")
    print(f"Last Updated: {result['Last Updated']}")
    print(f"Has Screenshots: {result['Has Screenshots']}")
    print(f"Number of Images: {len(result['Image Links'])}")
    print(f"Number of Internal Links: {len(result['Internal Links'])}")
    print(f"Number of YouTube Links: {len(result['YouTube Links'])}")

    # Optionally write CSV
    if args.csv:
        write_csv_row(args.csv, result)
        print(f"\n✅ Wrote one row to CSV: {args.csv}")


if __name__ == "__main__":
    main()
