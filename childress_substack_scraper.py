#!/usr/bin/env python3
"""
Scrape all posts from drcachildress.substack.com and save as .md files,
organized by section — ready to drop into the same GitHub repo as the
YouTube transcripts and indexed by the same Streamlit chatbot.

Requirements:
    pip install requests
"""

import re
import time
import requests
from pathlib import Path
from html.parser import HTMLParser

SUBSTACK_URL = "https://drcachildress.substack.com"
API          = f"{SUBSTACK_URL}/api/v1"
OUTPUT_DIR   = Path("childress_substack")
MASTER_FILE  = OUTPUT_DIR / "_ALL_SUBSTACK_POSTS.md"
PER_PAGE     = 12   # Substack API max per request
DELAY        = 0.5  # seconds between requests


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
    s.feed(html or "")
    return s.get_text()


# ── Substack API helpers ───────────────────────────────────────────────────

def get_sections() -> dict[str, str]:
    """Return {section_id: section_name}."""
    try:
        r = requests.get(f"{API}/sections", timeout=10)
        if r.status_code == 200:
            return {str(s["id"]): s["name"] for s in r.json()}
    except Exception:
        pass
    return {}


def get_all_posts() -> list[dict]:
    """Fetch all posts via the public Substack API."""
    posts = []
    offset = 0
    while True:
        print(f"  Fetching posts {offset}–{offset + PER_PAGE}...", end=" ", flush=True)
        try:
            r = requests.get(
                f"{API}/posts",
                params={"limit": PER_PAGE, "offset": offset},
                timeout=20,
            )
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
            offset += PER_PAGE
            time.sleep(DELAY)
        except Exception as e:
            print(f"Error: {e}")
            break
    return posts


# ── File helpers ───────────────────────────────────────────────────────────

def sanitize(name: str, max_len: int = 50) -> str:
    name = re.sub(r'[\\/*?:"<>|#]', "-", name)
    name = name.strip().strip(".")
    return name[:max_len]


def format_date(iso: str) -> str:
    return iso[:10] if iso else "0000-00-00"


def date_prefix(iso: str) -> str:
    d = iso[:10] if iso else "0000-00-00"
    return d.replace("-", ".")


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Fetching sections...")
    sections = get_sections()
    print(f"  Found {len(sections)} sections: {list(sections.values()) or ['(none)']}")

    print("\nFetching all posts...")
    posts = get_all_posts()
    print(f"\n  Total posts: {len(posts)}")

    if not posts:
        print("No posts found. Check the Substack URL.")
        return

    # Group by section, falling back to post type
    by_section: dict[str, list[dict]] = {}
    for post in posts:
        section_id   = str(post.get("section_id") or "")
        section_name = sections.get(section_id, "") or post.get("type", "newsletter").title()
        by_section.setdefault(section_name, []).append(post)

    master_entries = []
    total_saved = 0

    for section_name, section_posts in sorted(by_section.items()):
        folder = OUTPUT_DIR / sanitize(section_name)
        folder.mkdir(parents=True, exist_ok=True)
        print(f"\n📂 {section_name} ({len(section_posts)} posts)")

        master_entries.append(f"# 📂 {section_name}\n\n---\n\n")

        # Sort by date ascending
        section_posts.sort(key=lambda p: p.get("post_date", ""))

        for post in section_posts:
            title     = html_to_text(post.get("title", "Untitled"))
            subtitle  = html_to_text(post.get("subtitle", ""))
            body_html = post.get("body_html", "") or post.get("truncated_body_text", "")
            body_text = html_to_text(body_html)
            if subtitle and subtitle not in body_text:
                body_text = subtitle + "\n\n" + body_text

            date_iso  = post.get("post_date", "")
            slug      = post.get("slug", "")
            url       = f"{SUBSTACK_URL}/p/{slug}" if slug else SUBSTACK_URL

            if not body_text or len(body_text) < 50:
                print(f"  – Skipped (no content): {title[:55]}")
                continue

            prefix     = date_prefix(date_iso)
            safe_title = sanitize(title, max_len=60)
            filename   = f"{prefix}_{safe_title}.md"
            filepath   = folder / filename

            content = "\n".join([
                f"# {title}",
                "",
                f"**Link:** {url}",
                f"**Section:** {section_name}",
                f"**Date:** {format_date(date_iso)}",
                f"**Source:** drcachildress.substack.com",
                "",
                "---",
                "",
                body_text,
            ])
            filepath.write_text(content, encoding="utf-8")
            print(f"  ✓ {filename[:70]}")
            total_saved += 1

            master_entries.append(
                f"## {title}\n\n"
                f"**Link:** {url}  \n"
                f"**Section:** {section_name}  \n"
                f"**Date:** {format_date(date_iso)}\n\n"
                f"{body_text}\n\n---\n\n"
            )

    # Write master file
    master_header = (
        f"# Dr. Childress Substack – All Posts\n\n"
        f"**Site:** {SUBSTACK_URL}  \n"
        f"**Total posts:** {total_saved}  \n\n"
        f"---\n\n"
    )
    MASTER_FILE.write_text(master_header + "".join(master_entries), encoding="utf-8")

    print(f"\n✅ Done! {total_saved} posts saved to {OUTPUT_DIR.resolve()}")
    print(f"   Master file: {MASTER_FILE.resolve()}")


if __name__ == "__main__":
    main()
