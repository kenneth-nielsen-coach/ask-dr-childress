#!/usr/bin/env python3
"""
Convert the "printed webpage" PDFs in childress_pages/ into .md files the
Q&A app can index, in the same folder, next to their source PDFs.

These PDFs were made by printing pages from drcachildress-consulting.com
to PDF, so each page carries browser print artifacts that repeat on every
page break: a "Home » Custom Pages » <title>" breadcrumb, and a footer with
the capture date/time, the page title again, the source URL, and a page
number (e.g. "8/1/26, 6:54 PM ... https://.../custom-page/p7-parent-resources/
3/4"). Left in, that noise is duplicated once per page inside every PDF and
sits in the middle of the actual content, fragmenting chunks and diluting
the embeddings with the same handful of boilerplate lines repeated across
all 29 documents. This strips it out before writing the .md file, and pulls
the real source URL out of that footer instead of losing it.

Requirements:
    pip install pypdf

Usage:
    python convert_page_pdfs.py
"""

import re
from collections import Counter
from pathlib import Path

from pypdf import PdfReader

PAGES_DIR = Path("childress_pages")

# The breadcrumb sometimes wraps onto a second line ("Home (...) » Custom
# Pages (...)" then, on its own line, "» <page title>"), so both are
# stripped: the line starting with "Home (", and any line starting with "»"
# left over on its own afterwards.
BREADCRUMB_RE = re.compile(r"^Home \(https://[^)]+\)\s*».*$", re.MULTILINE)
BREADCRUMB_CONT_RE = re.compile(r"^».*$\n?", re.MULTILINE)
URL_RE = re.compile(r"https://drcachildress-consulting\.com/custom-page/[^\s)]+")


def make_footer_re(total_pages: int) -> re.Pattern:
    """
    Build a footer regex anchored to this document's real page count.

    The date/title/url/page-number footer wraps across a line break (the
    title ends one line, the URL + page number start the next), and two of
    these sit back-to-back with no separator at all when one page ends and
    the next begins (e.g. ".../p7/ 3/48/1/26, 6:54 PM ..." — footer for page
    3 immediately followed by the footer for page 4). A generic \\d+/\\d+
    for the page-number fraction is ambiguous there: it can't tell "3/4"
    followed by a stray "8" apart from "3/48". Anchoring the denominator to
    the document's actual page count (known from len(reader.pages)) removes
    that ambiguity.
    """
    return re.compile(
        r"\d{1,2}/\d{1,2}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M\s+.{0,200}?"
        rf"https://\S+\s*\d{{1,2}}/{total_pages}(?!\d)",
        re.DOTALL,
    )


def clean_text(raw: str, total_pages: int) -> tuple[str, str]:
    """Strip print artifacts and return (body, source_url)."""
    urls = URL_RE.findall(raw)
    url = Counter(urls).most_common(1)[0][0] if urls else ""

    text = BREADCRUMB_RE.sub("", raw)
    text = BREADCRUMB_CONT_RE.sub("", text)
    text = make_footer_re(total_pages).sub("", text)
    # Two footers glued with no separator at all (page N's ".../N/total"
    # immediately followed by page N+1's date, e.g. "...3/48/1/26, 6:54 PM
    # ...4/4") can leave one trailing fragment behind: the shared boundary
    # digit gets claimed by whichever match the regex engine finds first,
    # so the other footer partially survives. That only ever happens at the
    # very end of a document (the last footer has nothing after it to glue
    # to), so it's always safe to trim anything footer-shaped off the tail.
    text = re.sub(
        r"\d{1,3}/\d{1,3}/\d{2,4},\s*\d{1,2}:\d{2}\s*[AP]M.*\Z", "", text, flags=re.DOTALL
    )
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text, url


def main():
    pdf_files = sorted(PAGES_DIR.glob("*.pdf"))
    if not pdf_files:
        print(f"No PDFs found in {PAGES_DIR}/")
        return

    converted = 0
    for pdf_path in pdf_files:
        md_path = pdf_path.with_suffix(".md")
        if md_path.exists():
            continue  # don't clobber a hand-edited or already-converted file

        reader = PdfReader(pdf_path)
        raw = "".join(page.extract_text() or "" for page in reader.pages)
        if len(raw.strip()) < 50:
            print(f"  skip (no extractable text, likely scanned): {pdf_path.name}")
            continue

        body, url = clean_text(raw, len(reader.pages))
        title = (reader.metadata.get("/Title") or pdf_path.stem) if reader.metadata else pdf_path.stem
        title = re.sub(r"\s*-\s*Dr Craig Childress Consulting\s*$", "", title).strip()

        content = f"# {title}\n\n**Link:** {url}\n\n---\n\n{body}\n"
        md_path.write_text(content, encoding="utf-8")
        converted += 1
        print(f"  ✓ {pdf_path.name} -> {md_path.name}")

    print(f"\nConverted {converted} PDF(s) to .md in {PAGES_DIR}/")


if __name__ == "__main__":
    main()
