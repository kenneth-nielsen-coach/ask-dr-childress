#!/usr/bin/env python3
"""
Converts local PDF files to .md files for the Dr. Childress chatbot.

For normal PDFs:      extracts text directly
For scanned PDFs:     extracts embedded hyperlinks, then downloads and
                      transcribes the linked documents (PDFs or web pages)

Folder structure:
    pdf_input/
        normal-document.pdf          → text extracted directly
        scanned-document.pdf         → links extracted → linked docs downloaded
        Court Documents/
            report.pdf

Requirements:
    pip install pdfminer.six requests beautifulsoup4
"""

import re
import sys
import time
import requests
from pathlib import Path
from urllib.parse import urljoin, urlparse
from html.parser import HTMLParser
from io import BytesIO

try:
    from pdfminer.high_level import extract_text
    from pdfminer.pdfparser import PDFSyntaxError, PDFParser
    from pdfminer.pdfdocument import PDFDocument
    from pdfminer.pdfpage import PDFPage
    from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
    from pdfminer.converter import PDFPageAggregator
    from pdfminer.layout import LAParams, LTPage, LTAnno, LTChar
    import pdfminer.pdftypes as pdftypes
except ImportError:
    print("pdfminer.six not installed. Run: pip install pdfminer.six")
    sys.exit(1)

# ── Configuration ──────────────────────────────────────────────────────────

INPUT_DIR      = Path("pdf_input")
OUTPUT_DIR     = Path("childress_documents")
MASTER_FILE    = OUTPUT_DIR / "_ALL_DOCUMENTS.md"
CHUNK_SIZE     = 2000
CHUNK_OVERLAP  = 200
DEFAULT_SOURCE = "drcachildress-consulting.com"
DELAY          = 0.75  # seconds between downloads
MIN_TEXT_LEN   = 50    # chars — below this = treat as scanned

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


# ── HTML → plain text ──────────────────────────────────────────────────────

class HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.result = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        if tag in ("p", "br", "li", "h1", "h2", "h3", "h4", "blockquote"):
            self.result.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
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


# ── Text helpers ───────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    lines = text.splitlines()
    counts: dict[str, int] = {}
    for line in lines:
        s = line.strip()
        if s:
            counts[s] = counts.get(s, 0) + 1
    cleaned = [l for l in lines if not l.strip() or counts.get(l.strip(), 0) < 4]
    return "\n".join(cleaned).strip()


def chunk_text(text: str) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        end = start + CHUNK_SIZE
        if end < len(text):
            b = text.rfind(". ", start + CHUNK_SIZE - 200, end)
            if b == -1:
                b = text.rfind("\n", start + CHUNK_SIZE - 200, end)
            if b != -1:
                end = b + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start = end - CHUNK_OVERLAP
    return chunks or [text.strip()]


# ── Filename helpers ───────────────────────────────────────────────────────

def sanitize(name: str, max_len: int = 60) -> str:
    name = re.sub(r'[\\/*?:"<>|#]', "-", name)
    name = re.sub(r"-{2,}", "-", name)
    return name.strip().strip(".-")[:max_len]


def section_from_path(pdf_path: Path) -> str:
    rel = pdf_path.relative_to(INPUT_DIR)
    if len(rel.parts) > 1:
        return rel.parts[0].replace("-", " ").replace("_", " ").title()
    return "Documents"


# ── PDF link extraction ────────────────────────────────────────────────────

def extract_links_from_pdf(pdf_path: Path) -> list[str]:
    """
    Extract all hyperlinks embedded in a PDF's annotation layer.
    Works even on scanned/image PDFs since links are stored separately
    from the text layer.
    """
    links = []
    try:
        with open(pdf_path, "rb") as f:
            parser = PDFParser(f)
            doc = PDFDocument(parser)
            for page in PDFPage.create_pages(doc):
                if page.annots:
                    annots = page.annots
                    # Resolve indirect references
                    if hasattr(annots, "resolve"):
                        annots = annots.resolve()
                    if not isinstance(annots, list):
                        continue
                    for annot in annots:
                        if hasattr(annot, "resolve"):
                            annot = annot.resolve()
                        if not isinstance(annot, dict):
                            continue
                        subtype = annot.get("Subtype")
                        if hasattr(subtype, "name"):
                            subtype = subtype.name
                        if subtype != "Link":
                            continue
                        action = annot.get("A")
                        if hasattr(action, "resolve"):
                            action = action.resolve()
                        if isinstance(action, dict):
                            uri = action.get("URI")
                            if uri:
                                if isinstance(uri, bytes):
                                    uri = uri.decode("utf-8", errors="replace")
                                uri = uri.strip()
                                if uri.startswith("http") and uri not in links:
                                    links.append(uri)
    except Exception as e:
        print(f"    Link extraction error: {e}")
    return links


# ── Download helpers ───────────────────────────────────────────────────────

def download_pdf_text(url: str) -> str | None:
    """Download a PDF from a URL and extract its text."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=30)
        if r.status_code != 200:
            return None
        text = extract_text(BytesIO(r.content)) or ""
        return clean_text(text) or None
    except Exception as e:
        print(f"    PDF download error: {e}")
        return None


def download_page_text(url: str) -> str | None:
    """Download a web page and extract its text."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=20)
        if r.status_code != 200:
            return None
        return html_to_text(r.text) or None
    except Exception as e:
        print(f"    Page download error: {e}")
        return None


def download_linked_content(url: str) -> tuple[str, str] | None:
    """
    Download content from a URL (PDF or web page).
    Returns (text, content_type) or None.
    """
    parsed = urlparse(url)
    if parsed.path.lower().endswith(".pdf"):
        text = download_pdf_text(url)
        return (text, "PDF") if text else None
    else:
        text = download_page_text(url)
        return (text, "Web page") if text else None


# ── Save helper ────────────────────────────────────────────────────────────

def save_md(folder: Path, filename_base: str, header: str, text: str) -> int:
    folder.mkdir(parents=True, exist_ok=True)
    safe = sanitize(filename_base)
    filepath = folder / f"{safe}.md"
    filepath.write_text(header + text, encoding="utf-8")
    return len(chunk_text(text))


# ── Process a single PDF ───────────────────────────────────────────────────

def process_pdf(pdf_path: Path, master_entries: list) -> tuple[int, int]:
    """
    Process one PDF file.
    Returns (files_saved, chunks_estimated).
    """
    section = section_from_path(pdf_path)
    title   = pdf_path.stem.replace("-", " ").replace("_", " ").strip()
    folder  = OUTPUT_DIR / sanitize(section, max_len=50)

    print(f"\n  📄 {pdf_path.name}")

    # ── Try direct text extraction ─────────────────────────────────
    try:
        raw  = extract_text(str(pdf_path)) or ""
        text = clean_text(raw)
    except (PDFSyntaxError, Exception) as e:
        print(f"    ✗ Extraction error: {e}")
        return 0, 0

    if text and len(text) >= MIN_TEXT_LEN:
        # Normal PDF with text layer
        print(f"    ✓ Text extracted ({len(text):,} chars)")
        header = _make_header(title, DEFAULT_SOURCE, section)
        n = save_md(folder, title, header, text)
        master_entries.append(_master_entry(title, DEFAULT_SOURCE, section, text))
        return 1, n

    # ── Scanned PDF — extract links ────────────────────────────────
    print(f"    ⚠ No text layer (scanned PDF) — extracting links...")
    links = extract_links_from_pdf(pdf_path)

    if not links:
        print(f"    ✗ No links found either — skipping.")
        return 0, 0

    print(f"    Found {len(links)} link(s):")
    files_saved = total_chunks = 0

    for url in links:
        print(f"      → {url[:80]}", end=" ", flush=True)
        result = download_linked_content(url)
        time.sleep(DELAY)

        if not result:
            print("✗ no content")
            continue

        linked_text, content_type = result
        # Title from URL slug
        slug       = urlparse(url).path.rstrip("/").split("/")[-1]
        link_title = slug.replace("-", " ").replace("_", " ").replace(".pdf", "").strip()
        link_title = link_title or f"Linked from {title}"

        link_folder = folder / sanitize(title, max_len=50)
        header = _make_header(link_title, url, section, source_type=content_type)
        n = save_md(link_folder, link_title, header, linked_text)
        master_entries.append(_master_entry(link_title, url, section, linked_text))
        files_saved += 1
        total_chunks += n
        print(f"✓ ({len(linked_text):,} chars)")

    return files_saved, total_chunks


# ── Header / entry helpers ─────────────────────────────────────────────────

def _make_header(title: str, url: str, section: str, source_type: str = "PDF") -> str:
    return "\n".join([
        f"# {title}", "",
        f"**Link:** {url}",
        f"**Category:** {section}",
        f"**Source:** {DEFAULT_SOURCE}",
        f"**Type:** {source_type}",
        "", "---", "",
    ])


def _master_entry(title: str, url: str, section: str, text: str) -> str:
    preview = text[:400] + ("..." if len(text) > 400 else "")
    return (
        f"## {title}\n\n"
        f"**Link:** {url}  \n"
        f"**Category:** {section}\n\n"
        f"{preview}\n\n---\n\n"
    )


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not INPUT_DIR.exists():
        print(f"Input folder '{INPUT_DIR.resolve()}' not found.")
        print(f"Create it and put your PDF files inside.")
        print()
        print("Subfolder tip:")
        print("  pdf_input/")
        print("  pdf_input/Assessment Tools/checklist.pdf")
        print("  pdf_input/Court Documents/report.pdf")
        return

    pdf_files = sorted(INPUT_DIR.rglob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files\n")

    if not pdf_files:
        print("No PDF files found.")
        return

    master_entries: list[str] = []
    total_files = total_chunks = 0

    for pdf_path in pdf_files:
        files, chunks = process_pdf(pdf_path, master_entries)
        total_files  += files
        total_chunks += chunks

    # Master file
    MASTER_FILE.write_text(
        f"# Dr. Childress Documents\n\n"
        f"**Total files:** {total_files}  \n"
        f"**Estimated chunks:** {total_chunks}  \n\n"
        f"---\n\n"
        + "".join(master_entries),
        encoding="utf-8",
    )

    print(f"\n✅ Done!")
    print(f"   {total_files} documents → {OUTPUT_DIR.resolve()}")
    print(f"   Master file: {MASTER_FILE.resolve()}")


if __name__ == "__main__":
    main()
