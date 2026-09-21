"""
Shared indexing logic for the Dr. Childress Q&A bot.

Both app.py (at request time) and build_embeddings.py (offline, to
precompute the index) need to agree on exactly which files count as
"the corpus" and how they get chunked. Keeping that logic in one place
avoids the two ever drifting apart.
"""

import glob
import os
import re

EMBED_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

# Matches sentence boundaries: a run of ., ?, or ! followed by whitespace.
# Deliberately simple (no abbreviation handling) — good enough for splitting
# transcript/blog prose, and a slightly-off boundary here and there is much
# less harmful than cutting a chunk mid-sentence every single time.
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")

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


def _split_sentences(paragraph: str) -> list[str]:
    """Split one paragraph into sentences, falling back to the whole
    paragraph if the regex finds no boundaries (e.g. it's one long run-on
    line, as some transcript segments are)."""
    sentences = _SENTENCE_END.split(paragraph.strip())
    return [s for s in sentences if s]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Last-resort fixed-character split, only used for a single sentence
    or paragraph that's still too long to fit in one chunk on its own."""
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks or [text]


def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into chunks close to `size` characters, but breaking on
    paragraph and sentence boundaries wherever possible instead of cutting
    mid-sentence. This keeps each chunk a coherent, self-contained unit of
    meaning, which improves both embedding quality (MiniLM embeds a
    complete thought better than a fragment) and what ends up in Claude's
    context (no half-sentences at chunk edges).

    Sentences are accumulated into a chunk until adding the next one would
    exceed `size`; the chunk is then closed and a new one started, carried
    over with the last `overlap` characters of the previous chunk so
    context isn't lost across the boundary. A single sentence longer than
    `size` (rare, but some transcript lines run on) falls back to a hard
    character split just for that sentence.
    """
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return [text] if text.strip() else []

    units = []
    for paragraph in paragraphs:
        units.extend(_split_sentences(paragraph))
        units.append("")  # paragraph-break marker, joined as a blank line below

    chunks: list[str] = []
    current = ""
    for unit in units:
        piece = "\n\n" if unit == "" else unit
        candidate = f"{current} {piece}".strip() if current and piece != "\n\n" else current + piece

        if len(unit) > size:
            # One oversized sentence: flush what we have, hard-split the
            # sentence on its own, and carry on.
            if current.strip():
                chunks.append(current.strip())
                current = ""
            chunks.extend(_hard_split(unit, size, overlap))
            continue

        if len(candidate) > size and current.strip():
            chunks.append(current.strip())
            # Carry the tail of the previous chunk forward for overlap,
            # trimmed to the next whitespace boundary so it doesn't start
            # mid-word (the raw slice can land inside a word, or just
            # after a paragraph-break newline).
            tail = current[-overlap:] if overlap > 0 else ""
            ws = re.search(r"\s", tail)
            tail = tail[ws.end():] if ws else ""
            current = f"{tail} {unit}".strip() if unit else tail
        else:
            current = candidate

    if current.strip():
        chunks.append(current.strip())

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
