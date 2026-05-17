#!/usr/bin/env python3
"""
Scrape all posts from drcraigchildressblog.com and save as .md files,
organized by category — ready to drop into the same GitHub repo as the
YouTube transcripts and indexed by the same Streamlit chatbot.

Requirements:
    pip install requests
"""

import re
import json
import time
import requests
from pathlib import Path
from html.parser import HTMLParser

SITE = "drcraigchildressblog.com"
API = f"https://public-api.wordpress.com/wp/v2/sites/{SITE}"
OUTPUT_DIR = Path("childress_blog")
MASTER_FILE = OUTPUT_DIR / "_ALL_BLOG_POSTS.md"
PER_PAGE = 100
DELAY = 0.5  # seconds between requests to be polite


# ── HTML → plain text ──────────────────────────────────────────────────────

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self.skip_tags = {"script", "style"}
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in self.skip_tags:
            self._skip = True
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "blockquote", "hr"):
            self.result.append("\n")

    def handle_endtag(self, tag):
        if tag in self.skip_tags:
            self._skip = False

    def handle_data(self, data):
        if not self._skip:
            self.result.append(data)

    def get_text(self):
        return re.sub(r"\n{3,}", "\n\n", "".join(self.result)).strip()


def html_to_text(html: str) -> str:
    s = HTMLStripper()
    s.feed(html)
    return s.get_text()


# ── WordPress API helpers ──────────────────────────────────────────────────

def get_categories() -> dict[int, str]:
    """Return {id: name} for all categories."""
    cats = {}
    page = 1
    while True:
        r = requests.get(f"{API}/categories", params={"per_page": 100, "page": page}, timeout=15)
        if r.status_code != 200:
            break
        data = r.json()
        if not data:
            break
        for c in data:
            cats[c["id"]] = c["name"]
        if len(data) < 100:
            break
        page += 1
    return cats


def get_all_posts() -> list[dict]:
    """Fetch every post from the blog."""
    posts = []
    page = 1
    while True:
        print(f"  Fetching page {page}...", end=" ", flush=True)
        r = requests.get(
            f"{API}/posts",
            params={"per_page": PER_PAGE, "page": page, "status": "publish"},
            timeout=30,
        )
        if r.status_code == 400:
            print("done.")
            break
        if r.status_code != 200:
            print(f"Error {r.status_code}")
            break
        data = r.json()
        if not data:
            print("done.")
            break
        posts.extend(data)
        print(f"{len(data)} posts")
        if len(data) < PER_PAGE:
            break
        page += 1
        time.sleep(DELAY)
    return posts


# ── File helpers ───────────────────────────────────────────────────────────

def sanitize(name: str, max_len: int = 50) -> str:
    name = re.sub(r'[\\/*?:"<>|#]', "-", name)
    name = name.strip().strip(".")
    return name[:max_len]


def format_date(iso: str) -> str:
    """2024-03-15T12:00:00 → 2024-03-15"""
    return iso[:10] if iso else "0000-00-00"


def date_prefix(iso: str) -> str:
    """2024-03-15T12:00:00 → 2024.03.15"""
    d = iso[:10] if iso else "0000-00-00"
    return d.replace("-", ".")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching categories...")
    categories = get_categories()
    print(f"  Found {len(categories)} categories: {list(categories.values())}")

    print("\nFetching all posts...")
    posts = get_all_posts()
    print(f"\n  Total posts: {len(posts)}")

    # Group posts by primary category
    by_category: dict[str, list[dict]] = {}
    for post in posts:
        cat_ids = post.get("categories", [])
        cat_name = categories.get(cat_ids[0], "Uncategorized") if cat_ids else "Uncategorized"
        by_category.setdefault(cat_name, []).append(post)

    master_entries = []
    total_saved = 0

    for cat_name, cat_posts in sorted(by_category.items()):
        folder = OUTPUT_DIR / sanitize(cat_name)
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 {cat_name} ({len(cat_posts)} posts)")

        master_entries.append(
            f"# 📂 {cat_name}\n\n---\n\n"
        )

        # Sort posts by date ascending
        cat_posts.sort(key=lambda p: p.get("date", ""))

        for post in cat_posts:
            title = post.get("title", {}).get("rendered", "Untitled")
            title = html_to_text(title)
            url = post.get("link", "")
            date_iso = post.get("date", "")
            body_html = post.get("content", {}).get("rendered", "")
            body_text = html_to_text(body_html)

            prefix = date_prefix(date_iso)
            safe_title = sanitize(title, max_len=60)
            filename = f"{prefix}_{safe_title}.md"
            filepath = folder / filename

            content = "\n".join([
                f"# {title}",
                "",
                f"**Link:** {url}",
                f"**Category:** {cat_name}",
                f"**Date:** {format_date(date_iso)}",
                "",
                "---",
                "",
                body_text,
            ])
            filepath.write_text(content, encoding="utf-8")
            print(f"  ✓ {filename[:70]}")
            total_saved += 1

            status = ""
            master_entries.append(
                f"## {title}{status}\n\n"
                f"**Link:** {url}  \n"
                f"**Category:** {cat_name}  \n"
                f"**Date:** {format_date(date_iso)}\n\n"
                f"{body_text}\n\n---\n\n"
            )

    # Write master file
    master_header = (
        f"# Dr. Childress Blog – All Posts\n\n"
        f"**Site:** https://{SITE}  \n"
        f"**Total posts:** {total_saved}  \n\n"
        f"---\n\n"
    )
    MASTER_FILE.write_text(master_header + "".join(master_entries), encoding="utf-8")

    print(f"\n✅ Done! {total_saved} posts saved to {OUTPUT_DIR.resolve()}")
    print(f"   Master file: {MASTER_FILE.resolve()}")


if __name__ == "__main__":
    main()
