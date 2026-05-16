"""
SHL Catalog Scraper
Fetches all Individual Test Solutions from the SHL product catalog.
Run once before starting the server: python scrape_catalog.py
"""

import requests
import json
import time
import re
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.shl.com"
CATALOG_URL = "https://www.shl.com/solutions/products/product-catalog/"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

TEST_TYPE_MAP = {
    "A": "Ability & Aptitude",
    "B": "Biodata & Situational Judgement",
    "C": "Competencies",
    "D": "Development & 360",
    "E": "Assessment Exercises",
    "K": "Knowledge & Skills",
    "P": "Personality & Behavior",
    "S": "Simulations",
}


def fetch_page(start: int, retries: int = 3) -> str:
    """Fetch one page of catalog results (type=1 = Individual Test Solutions)."""
    params = {
        "action_doFilteringForm": "Search",
        "f": "1",
        "type": "1",
        "start": str(start),
    }
    for attempt in range(retries):
        try:
            resp = requests.get(
                CATALOG_URL, params=params, headers=HEADERS, timeout=30
            )
            resp.raise_for_status()
            return resp.text
        except Exception as e:
            print(f"  Attempt {attempt + 1} failed for start={start}: {e}")
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to fetch page start={start} after {retries} retries")


def parse_catalog_page(html: str) -> list[dict]:
    """Parse product rows from one catalog HTML page."""
    soup = BeautifulSoup(html, "html.parser")
    results = []

    # The table has class 'custom-table' or similar; find all rows
    table = soup.find("table")
    if not table:
        return results

    rows = table.find_all("tr")
    for row in rows:
        cells = row.find_all("td")
        if len(cells) < 4:
            continue

        # Cell 0: name + link
        name_cell = cells[0]
        link_tag = name_cell.find("a")
        if not link_tag:
            continue
        name = link_tag.get_text(strip=True)
        relative_url = link_tag.get("href", "")
        url = urljoin(BASE_URL, relative_url)

        # Cell 1: Remote Testing (checkmark or empty)
        remote = bool(cells[1].find(["span", "img", "i"]))

        # Cell 2: Adaptive/IRT
        adaptive = bool(cells[2].find(["span", "img", "i"]))

        # Cell 3: Test Type letter(s) — look for span text or title
        type_cell = cells[3]
        type_spans = type_cell.find_all("span")
        test_types = []
        for span in type_spans:
            letter = span.get_text(strip=True)
            if letter in TEST_TYPE_MAP:
                test_types.append(letter)
        # Fallback: look for any single capital letter
        if not test_types:
            raw = type_cell.get_text(strip=True)
            for ch in raw:
                if ch in TEST_TYPE_MAP:
                    test_types.append(ch)

        results.append(
            {
                "name": name,
                "url": url,
                "remote_testing": remote,
                "adaptive_irt": adaptive,
                "test_types": test_types,
                "test_type": test_types[0] if test_types else "",
                "test_type_labels": [TEST_TYPE_MAP.get(t, t) for t in test_types],
            }
        )

    return results


def get_total_count(html: str) -> int:
    """Extract total result count or estimate from pagination."""
    soup = BeautifulSoup(html, "html.parser")
    # Try to find "Showing X of Y" text
    text = soup.get_text()
    match = re.search(r"(\d+)\s+result", text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    # Fallback: count pagination links to get last page number
    last_page = 1
    for a in soup.find_all("a", href=True):
        href = a["href"]
        m = re.search(r"start=(\d+)", href)
        if m:
            page_start = int(m.group(1))
            last_page = max(last_page, page_start)
    return last_page + 12  # 12 items per page


def fetch_product_detail(url: str) -> dict:
    """Fetch additional detail from a product's own page."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=20)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        detail = {}

        # Description: first meaningful paragraph after h1
        h1 = soup.find("h1")
        if h1:
            for sib in h1.find_next_siblings():
                text = sib.get_text(strip=True)
                if text and len(text) > 40:
                    detail["description"] = text
                    break

        # Job levels
        text = soup.get_text(" ", strip=True)
        levels = []
        level_keywords = [
            "Entry-Level", "Graduate", "Manager", "Mid-Professional",
            "Professional Individual Contributor", "Director", "Executive",
            "Front Line Manager", "General Population", "Supervisor",
        ]
        for level in level_keywords:
            if level in text:
                levels.append(level)
        detail["job_levels"] = levels

        # Languages
        lang_pattern = re.compile(
            r"(English[^,\n]*|Spanish|French[^,\n]*|German|Portuguese[^,\n]*|"
            r"Dutch|Italian|Japanese|Chinese[^,\n]*|Arabic|Russian|Turkish|"
            r"Korean|Swedish|Norwegian|Danish|Polish|Czech|Hungarian|Romanian|"
            r"Finnish|Greek|Bulgarian|Croatian|Serbian|Slovak|Estonian|Latvian|"
            r"Lithuanian|Thai|Vietnamese|Indonesian|Malay|Icelandic|Flemish)"
        )
        langs = list(set(lang_pattern.findall(text)))
        detail["languages"] = langs[:20]  # cap at 20

        # Duration if present
        dur_match = re.search(r"(\d+)\s*(?:minute|min)", text, re.IGNORECASE)
        if dur_match:
            detail["duration_minutes"] = int(dur_match.group(1))

        return detail
    except Exception as e:
        print(f"  Warning: could not fetch detail for {url}: {e}")
        return {}


def scrape_all():
    """Main scraping routine. Returns list of assessment dicts."""
    print("Starting SHL catalog scrape (Individual Test Solutions only)...")

    # Fetch first page to gauge total
    html0 = fetch_page(0)
    first_batch = parse_catalog_page(html0)
    total_estimate = get_total_count(html0)
    print(f"  First page: {len(first_batch)} items. Estimated total: {total_estimate}")

    all_items = list(first_batch)
    seen_urls = {item["url"] for item in first_batch}

    # Pages are 12 items each; iterate until no new items
    start = 12
    consecutive_empty = 0
    while start <= total_estimate + 12:
        print(f"  Fetching start={start}...")
        try:
            html = fetch_page(start)
            batch = parse_catalog_page(html)
        except RuntimeError as e:
            print(f"  Stopping due to error: {e}")
            break

        new_items = [it for it in batch if it["url"] not in seen_urls]
        if not new_items:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print("  No new items for 3 pages; stopping.")
                break
        else:
            consecutive_empty = 0
            for it in new_items:
                seen_urls.add(it["url"])
            all_items.extend(new_items)

        start += 12
        time.sleep(0.5)  # polite delay

    print(f"\nTotal items scraped from listing: {len(all_items)}")

    # Enrich with detail pages (rate-limited)
    print("\nFetching detail pages for descriptions and job levels...")
    for i, item in enumerate(all_items):
        if i % 20 == 0:
            print(f"  Detail {i+1}/{len(all_items)}...")
        detail = fetch_product_detail(item["url"])
        item.update(detail)
        time.sleep(0.4)

    return all_items


def main():
    items = scrape_all()

    output_path = "data/shl_catalog.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(items)} assessments to {output_path}")

    # Quick summary
    from collections import Counter
    type_counts = Counter()
    for item in items:
        for t in item.get("test_types", []):
            type_counts[t] += 1
    print("\nTest type breakdown:")
    for letter, count in sorted(type_counts.items()):
        print(f"  {letter} ({TEST_TYPE_MAP.get(letter, '?')}): {count}")


if __name__ == "__main__":
    main()
