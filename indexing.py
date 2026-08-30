"""
Shared indexing logic for the Dr. Childress Q&A bot.

Both app.py (at request time) and build_embeddings.py (offline, to
precompute the index) need to agree on exactly which files count as
"the corpus" and how they get chunked. Keeping that logic in one place
avoids the two ever drifting apart.
"""

import glob
import os

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

TRANSCRIPTS_DIRS = [
    "childress_transcripts",   # YouTube transcripts
    "childress_blog",          # Blog posts
    "childress_substack",      # Substack posts
    "childress_pages",         # Converted site pages (see convert_page_pdfs.py)
    "childress_papers",        # Book chapters / papers
    "kenneth_tests",           # Kenneth's test folder for new files before adding to main folders
]


def list_source_files() -> list[str]:
    """
    Collect every .md file to index.

    Skips files whose name starts with "_" — each scraper
    (childress_transcripts.py, childress_blog_scraper.py,
    childress_substack_scraper.py, and the childress_papers.py /
    childress_pdf_converter.py pair) writes one of these alongside the
    per-item files (_ALL_TRANSCRIPTS.md, _ALL_BLOG_POSTS.md,
    _ALL_SUBSTACK_POSTS.md, _ALL_DOCUMENTS.md), and each one is just every
    other file in that folder concatenated together. Indexing them too
    means every piece of content gets embedded twice: once on its own and
    once inside the rollup, which roughly doubles indexing time for no
    benefit and lets near-duplicate chunks crowd out genuinely different
    results at query time.
    """
    files = []
    for d in TRANSCRIPTS_DIRS:
        for path in glob.glob(f"{d}/**/*.md", recursive=True):
            if not os.path.basename(path).startswith("_"):
                files.append(path)
    return files


def parse_file(path: str) -> tuple[dict, str]:
    """Read one markdown source file and split it into (metadata, body)."""
    text = open(path, encoding="utf-8", errors="replace").read()

    title, url, date, playlist = "Unknown", "", "", ""
    for line in text.splitlines()[:10]:
        if line.startswith("# "):
            title = line[2:].strip().strip("*").strip()
        elif line.startswith("**Link:**"):
            url = line.replace("**Link:**", "").strip()
        elif line.startswith("**Date:**"):
            date = line.replace("**Date:**", "").strip()
        elif line.startswith("**Playlist:**"):
            playlist = line.replace("**Playlist:**", "").strip()
        elif line.startswith("**Category:**"):
            playlist = line.replace("**Category:**", "").strip()
        elif line.startswith("**Section:**"):
            playlist = line.replace("**Section:**", "").strip()

    body = text.split("---\n", 1)[-1].strip()
    metadata = {"title": title, "url": url, "date": date, "playlist": playlist}
    return metadata, body


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks or [text]


def build_corpus() -> tuple[list[str], list[dict], int]:
    """Read every source file and split it into (chunks, metadatas, file_count)."""
    files = list_source_files()
    chunks, metadatas = [], []
    for path in files:
        metadata, body = parse_file(path)
        for chunk in chunk_text(body):
            chunks.append(chunk)
            metadatas.append(metadata)
    return chunks, metadatas, len(files)
