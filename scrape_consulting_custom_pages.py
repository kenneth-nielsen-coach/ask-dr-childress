#!/usr/bin/env python3
"""
scrape_consulting_custom_pages.py
=================================

Scrapes every page under https://drcachildress-consulting.com/custom-page/,
extracts the page text AND every linked document (PDF / DOC / DOCX / PPT / etc.),
and writes clean .md files ready to be chunked and embedded into the
Ask-Dr-Childress RAG corpus.

Crawl scope
-----------
Seeds from the 4 paginated archive pages, then recursively follows every
button / link that points to another /custom-page/<slug>/ page, to any depth,
until no new pages appear -- so sub-pages reachable only via an on-page button
are still captured.

Filenames
---------
Each .md is named  yyyy.mm.dd_Title.md  where the date is the WordPress
modified_time for that page (from JSON-LD dateModified, else the
article:modified_time meta tag, else the WP REST API). Documents inherit their
parent page's modified date. If no date can be found, the prefix is omitted.

What it produces
----------------
OUTPUT_DIR/
  pages/         one .md per custom page  (page text + list of its documents)
  documents/     one .md per document     (extracted text of each PDF/DOCX)
  _raw/          the original downloaded files (NOT for embedding)
  manifest.json  url -> file map, so re-runs are incremental

Each .md starts with YAML front matter (title, source_url, ...) so the
retrieval layer can cite the original source with a real link.

Fetching
--------
The site sits behind Cloudflare. Two engines are supported:
  --engine requests    (default) plain HTTP with a browser User-Agent. Fast;
                       works whenever Cloudflare is not serving a JS challenge.
  --engine playwright  renders the page in a real headless browser. Slower but
                       clears Cloudflare's managed challenge. Falls back to this
                       automatically if a requests fetch looks like a CF block.

Install
-------
    pip install requests beautifulsoup4 markdownify pdfminer.six python-docx
    # only if you need the browser engine:
    pip install playwright && playwright install chromium

Run
---
    python scrape_consulting_custom_pages.py
    python scrape_consulting_custom_pages.py --engine playwright
    python scrape_consulting_custom_pages.py --force        # re-download everything
    python scrape_consulting_custom_pages.py --limit 3      # test on a few pages
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify as md

# ----------------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------------

ARCHIVE_URL = "https://drcachildress-consulting.com/custom-page/"
BASE_DOMAIN = "drcachildress-consulting.com"

OUTPUT_DIR = Path("childress_custom_pages")     # everything lands here
REQUEST_DELAY = 1.5                              # seconds between requests (be polite)
REQUEST_TIMEOUT = 45
MAX_ARCHIVE_PAGES = 25                           # safety cap on pagination crawl

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"
)

# extensions we treat as "documents" to download + convert
DOC_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".ppt", ".pptx",
    ".xls", ".xlsx", ".rtf", ".txt", ".odt",
}

# ----------------------------------------------------------------------------
# Filename sanitisation  (matches the existing pipeline's conventions)
#   '#' -> '-', invalid chars stripped, no double dashes,
#   folder names <= 50 chars, file title portions <= 60 chars,
#   date prefix 'yyyy.mm.dd_Title.md' when a date is known.
# ----------------------------------------------------------------------------

# Keep word characters (Unicode-aware, so Danish æ/ø/å survive), spaces,
# dots and dashes. Everything else -- &, commas, colons, parentheses,
# quotes, and all Windows-illegal chars -- gets stripped.
_PUNCT = re.compile(r"[^\w .\-]", re.UNICODE)
_MULTI_DASH = re.compile(r"-{2,}")
_MULTI_SPACE = re.compile(r"\s+")


def _clean(text: str) -> str:
    text = text.replace("#", "-")
    text = _PUNCT.sub("", text)
    text = _MULTI_SPACE.sub(" ", text).strip()
    text = text.replace(" ", "-")
    text = _MULTI_DASH.sub("-", text)
    return text.strip("-. ")


def sanitize_title(title: str, max_len: int = 60) -> str:
    cleaned = _clean(title)
    if len(cleaned) > max_len:
        cleaned = cleaned[:max_len].rstrip("-. ")
        cleaned = _MULTI_DASH.sub("-", cleaned)
    return cleaned or "untitled"


def sanitize_folder(name: str, max_len: int = 50) -> str:
    return sanitize_title(name, max_len=max_len)


def build_filename(title: str, date: str | None = None, ext: str = ".md") -> str:
    """date must be 'yyyy.mm.dd' if provided."""
    stem = sanitize_title(title)
    if date:
        return f"{date}_{stem}{ext}"
    return f"{stem}{ext}"


# ----------------------------------------------------------------------------
# Fetching  (requests engine + optional playwright engine)
# ----------------------------------------------------------------------------

class Fetcher:
    def __init__(self, engine: str = "requests"):
        self.engine = engine
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        })
        self._pw = None            # lazy playwright browser
        self._pw_page = None

    # -- public ------------------------------------------------------------
    def get_html(self, url: str) -> str:
        if self.engine == "playwright":
            return self._get_html_playwright(url)
        html = self._get_html_requests(url)
        if _looks_like_cloudflare_block(html):
            print("  ! Cloudflare challenge detected -> switching to playwright")
            self.engine = "playwright"
            return self._get_html_playwright(url)
        return html

    def download(self, url: str, dest: Path) -> bool:
        """Stream a binary document to disk. Returns True on success."""
        try:
            with self._session.get(url, stream=True, timeout=REQUEST_TIMEOUT) as r:
                r.raise_for_status()
                dest.parent.mkdir(parents=True, exist_ok=True)
                with open(dest, "wb") as fh:
                    for chunk in r.iter_content(chunk_size=1 << 15):
                        fh.write(chunk)
            return True
        except Exception as exc:  # noqa: BLE001
            print(f"  ! download failed: {url}  ({exc})")
            return False

    def close(self):
        if self._pw is not None:
            try:
                self._pw_browser.close()
                self._pw.stop()
            except Exception:  # noqa: BLE001
                pass

    # -- engines -----------------------------------------------------------
    def _get_html_requests(self, url: str) -> str:
        r = self._session.get(url, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        return r.text

    def _ensure_playwright(self):
        if self._pw_page is not None:
            return
        try:
            from playwright.sync_api import sync_playwright
        except ImportError as exc:
            raise SystemExit(
                "Playwright engine requested but not installed.\n"
                "  pip install playwright && playwright install chromium"
            ) from exc
        self._pw = sync_playwright().start()
        self._pw_browser = self._pw.chromium.launch(headless=True)
        ctx = self._pw_browser.new_context(user_agent=USER_AGENT, locale="en-US")
        self._pw_page = ctx.new_page()

    def _get_html_playwright(self, url: str) -> str:
        self._ensure_playwright()
        self._pw_page.goto(url, wait_until="networkidle", timeout=REQUEST_TIMEOUT * 1000)
        # give any Cloudflare interstitial a moment to resolve
        self._pw_page.wait_for_timeout(2500)
        return self._pw_page.content()


def _looks_like_cloudflare_block(html: str) -> bool:
    lowered = html.lower()
    signals = (
        "just a moment",
        "cf-browser-verification",
        "checking your browser",
        "cf-challenge",
        "enable javascript and cookies to continue",
    )
    return any(s in lowered for s in signals) and len(html) < 15000


# ----------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------

@dataclass
class DocLink:
    url: str
    label: str


@dataclass
class ScrapedPage:
    url: str
    title: str
    markdown: str
    date: str | None = None                       # 'yyyy.mm.dd' from WP modified_time
    documents: list[DocLink] = field(default_factory=list)
    child_pages: list[str] = field(default_factory=list)  # sub-page/button links found


def _same_site(url: str) -> bool:
    return urlparse(url).netloc.endswith(BASE_DOMAIN)


def normalize_url(base: str, href: str) -> str:
    """Absolute URL, no query/fragment, single trailing slash -- so the crawl
    dedupes /foo, /foo/, /foo?x, /foo# to one canonical key."""
    absu = urljoin(base, href.split("#")[0].split("?")[0])
    pr = urlparse(absu)
    path = pr.path if pr.path.endswith("/") else pr.path + "/"
    return f"{pr.scheme}://{pr.netloc}{path}"


def _is_custom_subpage(url: str) -> bool:
    """True for /custom-page/<slug>/ but not the archive root or /page/N/."""
    m = re.match(r"^/custom-page/([^/]+)/$", urlparse(url).path)
    return bool(m) and m.group(1) != "page"


# --- WordPress modified_time extraction ------------------------------------

def _parse_iso_date(value: str | None) -> str | None:
    """Any ISO-8601 timestamp -> 'yyyy.mm.dd'."""
    if not value:
        return None
    m = re.match(r"(\d{4})-(\d{2})-(\d{2})", value.strip().replace("Z", "+00:00"))
    return f"{m.group(1)}.{m.group(2)}.{m.group(3)}" if m else None


def _dates_from_jsonld(soup: BeautifulSoup) -> list[str]:
    """Walk every JSON-LD block (Yoast uses an @graph) collecting dateModified."""
    out: list[str] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or tag.get_text()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:  # noqa: BLE001
            continue
        stack = [data]
        while stack:
            node = stack.pop()
            if isinstance(node, dict):
                d = _parse_iso_date(node.get("dateModified") if isinstance(node.get("dateModified"), str) else None)
                if d:
                    out.append(d)
                stack.extend(v for v in node.values() if isinstance(v, (dict, list)))
            elif isinstance(node, list):
                stack.extend(node)
    return out


def extract_modified_date_from_html(soup: BeautifulSoup) -> str | None:
    dates = _dates_from_jsonld(soup)
    if dates:
        return max(dates)                          # latest = most recent modification
    for prop in ("article:modified_time", "og:updated_time", "article:published_time"):
        tag = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if tag and tag.get("content"):
            d = _parse_iso_date(tag["content"])
            if d:
                return d
    return None


def fetch_date_via_rest(fetcher: "Fetcher", url: str) -> str | None:
    """Last-resort: ask the WordPress REST API for this slug's `modified` date."""
    pr = urlparse(url)
    slug = pr.path.rstrip("/").split("/")[-1]
    origin = f"{pr.scheme}://{pr.netloc}"
    for base in ("custom-page", "custom-pages", "pages", "posts"):
        api = f"{origin}/wp-json/wp/v2/{base}?slug={slug}"
        try:
            r = fetcher._session.get(api, timeout=REQUEST_TIMEOUT)
            if r.status_code != 200:
                continue
            arr = r.json()
            if isinstance(arr, list) and arr:
                d = _parse_iso_date(arr[0].get("modified") or arr[0].get("modified_gmt"))
                if d:
                    return d
        except Exception:  # noqa: BLE001
            continue
    return None


def discover_custom_pages(fetcher: Fetcher) -> list[str]:
    """Crawl the paginated archive and return every /custom-page/<slug>/ URL."""
    seen: set[str] = set()
    ordered: list[str] = []
    to_visit = [ARCHIVE_URL]
    visited_archive: set[str] = set()

    while to_visit and len(visited_archive) < MAX_ARCHIVE_PAGES:
        archive_url = to_visit.pop(0)
        if archive_url in visited_archive:
            continue
        visited_archive.add(archive_url)
        print(f"[archive] {archive_url}")
        soup = BeautifulSoup(fetcher.get_html(archive_url), "html.parser")
        time.sleep(REQUEST_DELAY)

        for a in soup.find_all("a", href=True):
            href = normalize_url(archive_url, a["href"])
            if not _same_site(href):
                continue
            # pagination link -> queue it
            if re.search(r"/custom-page/page/\d+/$", urlparse(href).path):
                if href not in visited_archive:
                    to_visit.append(href)
                continue
            # a real custom sub-page
            if _is_custom_subpage(href) and href not in seen:
                seen.add(href)
                ordered.append(href)

    return ordered


def extract_page(url: str, html: str) -> ScrapedPage:
    soup = BeautifulSoup(html, "html.parser")

    # read the WP modified date FIRST -- it lives in <script type=ld+json>
    # and <meta> tags that the chrome-stripping below is about to remove.
    date = extract_modified_date_from_html(soup)

    # strip chrome: scripts, styles, header, footer, nav
    for sel in ["script", "style", "noscript", "header", "footer", "nav"]:
        for el in soup.select(sel):
            el.decompose()
    for sel in [
        ".elementor-location-header", ".elementor-location-footer",
        "#masthead", "#colophon", "[role=navigation]", ".site-header", ".site-footer",
    ]:
        for el in soup.select(sel):
            el.decompose()

    # find the main content region
    root = (
        soup.select_one("#content")
        or soup.select_one("main")
        or soup.select_one("article")
        or soup.body
        or soup
    )

    # title: first h1, else <title> minus the site suffix, else slug
    title = ""
    h1 = root.find(["h1"])
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(strip=True)
    if not title and soup.title:
        title = re.sub(r"\s*[-|]\s*Dr Craig Childress.*$", "", soup.title.get_text(strip=True))
    if not title:
        title = urlparse(url).path.rstrip("/").split("/")[-1].replace("-", " ").title()

    # walk anchors once: split into documents vs. child sub-pages (buttons)
    docs: list[DocLink] = []
    seen_doc_urls: set[str] = set()
    child_pages: list[str] = []
    seen_child: set[str] = set()
    for a in root.find_all("a", href=True):
        raw = a["href"].split("#")[0]
        if not raw:
            continue
        target = urljoin(url, raw)
        ext = Path(urlparse(target).path).suffix.lower()
        if ext in DOC_EXTENSIONS:                       # a downloadable document
            if target not in seen_doc_urls:
                seen_doc_urls.add(target)
                label = a.get_text(strip=True) or Path(urlparse(target).path).name
                docs.append(DocLink(url=target, label=label))
            continue
        norm = normalize_url(url, raw)                  # a link to another page
        if _same_site(norm) and _is_custom_subpage(norm) and norm != url:
            if norm not in seen_child:
                seen_child.add(norm)
                child_pages.append(norm)

    markdown = md(str(root), heading_style="ATX", strip=["a"]).strip()
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)

    return ScrapedPage(
        url=url, title=title, markdown=markdown, date=date,
        documents=docs, child_pages=child_pages,
    )


# ----------------------------------------------------------------------------
# Document text extraction
# ----------------------------------------------------------------------------

def extract_document_text(path: Path) -> str:
    ext = path.suffix.lower()
    try:
        if ext == ".pdf":
            from pdfminer.high_level import extract_text
            return extract_text(str(path)) or ""
        if ext == ".docx":
            import docx
            doc = docx.Document(str(path))
            return "\n".join(p.text for p in doc.paragraphs)
        if ext in {".txt", ".rtf"}:
            return path.read_text(errors="ignore")
    except Exception as exc:  # noqa: BLE001
        print(f"  ! could not extract text from {path.name}: {exc}")
    return ""


# ----------------------------------------------------------------------------
# Writing markdown
# ----------------------------------------------------------------------------

def _front_matter(fields: dict) -> str:
    lines = ["---"]
    for k, v in fields.items():
        if isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {item}")
        else:
            val = str(v).replace('"', "'")
            lines.append(f'{k}: "{val}"')
    lines.append("---\n")
    return "\n".join(lines)


def write_page_md(page: ScrapedPage, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = build_filename(page.title, date=page.date)
    path = out_dir / fname
    fm = _front_matter({
        "title": page.title,
        "source_url": page.url,
        "type": "custom-page",
        "modified": page.date or "",
        "documents": [d.url for d in page.documents],
    })
    body = f"# {page.title}\n\n{page.markdown}\n"
    if page.documents:
        body += "\n## Linked documents\n\n"
        for d in page.documents:
            body += f"- [{d.label}]({d.url})\n"
    path.write_text(fm + body, encoding="utf-8")
    return path


def write_document_md(doc: DocLink, text: str, parent: ScrapedPage, out_dir: Path) -> Path | None:
    if not text.strip():
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    title = doc.label or Path(urlparse(doc.url).path).stem
    fname = build_filename(title, date=parent.date)   # inherit parent's modified date
    path = out_dir / fname
    # avoid clobbering when two docs sanitise to the same name
    if path.exists():
        stem = path.stem
        i = 2
        while (out_dir / f"{stem}-{i}.md").exists():
            i += 1
        path = out_dir / f"{stem}-{i}.md"
    fm = _front_matter({
        "title": title,
        "source_url": doc.url,
        "type": "document",
        "modified": parent.date or "",
        "parent_page": parent.url,
    })
    body = f"# {title}\n\n{text.strip()}\n"
    path.write_text(fm + body, encoding="utf-8")
    return path


# ----------------------------------------------------------------------------
# Orchestration
# ----------------------------------------------------------------------------

def load_manifest(path: Path) -> dict:
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {"pages": {}, "documents": {}}


def save_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run(engine: str, force: bool, limit: int | None) -> None:
    pages_dir = OUTPUT_DIR / "pages"
    docs_dir = OUTPUT_DIR / "documents"
    raw_dir = OUTPUT_DIR / "_raw"
    manifest_path = OUTPUT_DIR / "manifest.json"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = {"pages": {}, "documents": {}} if force else load_manifest(manifest_path)
    fetcher = Fetcher(engine=engine)

    try:
        # 1) seed the crawl from the 4 paginated archive pages
        seeds = discover_custom_pages(fetcher)
        print(f"\nArchive lists {len(seeds)} custom pages. "
              f"Following button-links from each to find any more...\n")

        # 2) recursive crawl: a FIFO queue that grows as pages reveal sub-pages
        queue: list[str] = list(seeds)
        queued: set[str] = set(seeds)
        scraped: set[str] = set()
        processed = 0

        while queue:
            url = queue.pop(0)
            if url in scraped:
                continue
            scraped.add(url)
            processed += 1

            if limit and processed > limit:
                print(f"  (stopping at --limit {limit})")
                break

            print(f"[{processed}] {url}")
            if (not force) and url in manifest["pages"]:
                print("  = already done, skipping (use --force to redo)")
                # still follow its recorded children so the crawl stays complete
                for child in manifest["pages"][url].get("child_pages", []):
                    if child not in queued:
                        queued.add(child)
                        queue.append(child)
                continue
            try:
                html = fetcher.get_html(url)
            except Exception as exc:  # noqa: BLE001
                print(f"  ! fetch failed: {exc}")
                continue
            time.sleep(REQUEST_DELAY)

            page = extract_page(url, html)

            # fill in the date from the REST API if the HTML had none
            if not page.date:
                page.date = fetch_date_via_rest(fetcher, url)

            # enqueue any newly discovered sub-pages (buttons)
            new_children = 0
            for child in page.child_pages:
                if child not in queued:
                    queued.add(child)
                    queue.append(child)
                    new_children += 1

            page_md = write_page_md(page, pages_dir)
            print(f"  + page -> {page_md.name}   "
                  f"(date={page.date or '?'}, {len(page.documents)} docs, "
                  f"+{new_children} new sub-pages)")

            doc_files = []
            for doc in page.documents:
                if (not force) and doc.url in manifest["documents"]:
                    continue
                raw_name = sanitize_title(Path(urlparse(doc.url).path).stem) + \
                    Path(urlparse(doc.url).path).suffix.lower()
                raw_path = raw_dir / raw_name
                if force or not raw_path.exists():
                    ok = fetcher.download(doc.url, raw_path)
                    time.sleep(REQUEST_DELAY)
                    if not ok:
                        continue
                text = extract_document_text(raw_path)
                doc_md = write_document_md(doc, text, page, docs_dir)
                if doc_md:
                    print(f"      + doc  -> {doc_md.name}")
                    doc_files.append(doc_md.name)
                manifest["documents"][doc.url] = {
                    "parent": url,
                    "raw": raw_path.name,
                    "md": doc_md.name if doc_md else None,
                }

            manifest["pages"][url] = {
                "md": page_md.name,
                "modified": page.date,
                "documents": doc_files,
                "child_pages": page.child_pages,
            }
            save_manifest(manifest_path, manifest)

    finally:
        fetcher.close()
        save_manifest(manifest_path, manifest)

    print(f"\nDone. Output in: {OUTPUT_DIR.resolve()}")
    print(f"  pages:     {len(manifest['pages'])}")
    print(f"  documents: {len(manifest['documents'])}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", choices=["requests", "playwright"], default="requests")
    ap.add_argument("--force", action="store_true", help="re-scrape / re-download everything")
    ap.add_argument("--limit", type=int, default=None, help="only process first N pages (testing)")
    args = ap.parse_args()
    run(engine=args.engine, force=args.force, limit=args.limit)


if __name__ == "__main__":
    main()
