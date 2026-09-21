"""
Shared indexing logic for the Dr. Childress Q&A bot.

Both app.py (at request time) and build_embeddings.py (offline, to
precompute the index) need to agree on exactly which files count as
"the corpus" and how they get chunked. Keeping that logic in one place
avoids the two ever drifting apart.
"""

import glob
import hashlib
import os
import re
from collections import Counter
from pathlib import Path

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
    "childress_papers",        # Book chapters / papers. (childress_documents/ is only a reformatted copy of
                               # these, made by childress_papers.py - indexing it too would store every paper twice.)
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
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()

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


# childress_transcripts.py saves this one-line body for a video YouTube had no
# captions for yet. It holds no content, so it must never become a search chunk.
_PLACEHOLDER_PREFIX = "*No automatic transcript available"
_YOUTUBE_ID = re.compile(r"youtube\.com/watch\?v=([\w-]{6,})")


def _is_placeholder(body: str) -> bool:
    b = body.strip()
    return not b or (b.startswith(_PLACEHOLDER_PREFIX) and len(b) < 120)


def _fingerprint(body: str) -> str:
    """Hash of a body ignoring whitespace and case, to spot identical text."""
    return hashlib.md5(re.sub(r"\s+", " ", body).strip().lower().encode("utf-8")).hexdigest()


def load_documents(stats: dict | None = None) -> list[tuple[str, dict, str]]:
    """
    Read every source file and return the documents worth indexing as
    (path, metadata, body), after dropping:

      * placeholders - videos YouTube had no captions for yet - and empty files;
      * repeat copies of the same YouTube video. childress_transcripts.py saves
        a video once for every playlist it belongs to, so one video can exist as
        2-7 files. Left in, the same passage fills several of the top search
        results and crowds out other material. The longest copy is kept;
      * files whose text is identical to an earlier file's (for example a paper
        accidentally saved twice under two names). The first one is kept.

    If `stats` is given it is filled in, for build_embeddings.py to print.
    """
    paths = sorted(list_source_files())
    found = Counter(Path(p).parts[0] for p in paths)

    placeholders, loaded = [], []
    for path in paths:
        metadata, body = parse_file(path)
        if _is_placeholder(body):
            placeholders.append(metadata["title"])
            continue
        loaded.append((path, metadata, body))

    # Same video in several playlist folders -> keep the longest copy.
    best_for_video: dict[str, tuple[str, dict, str]] = {}
    for item in loaded:
        m = _YOUTUBE_ID.search(item[1]["url"])
        if m and (m.group(1) not in best_for_video or len(item[2]) > len(best_for_video[m.group(1)][2])):
            best_for_video[m.group(1)] = item
    unique_videos, repeats = [], 0
    for item in loaded:
        m = _YOUTUBE_ID.search(item[1]["url"])
        if m and best_for_video[m.group(1)] is not item:
            repeats += 1
            continue
        unique_videos.append(item)

    # Identical text under different file names -> keep the first.
    seen: dict[str, str] = {}
    docs, identical = [], []
    for path, metadata, body in unique_videos:
        fp = _fingerprint(body)
        if fp in seen:
            identical.append((os.path.basename(seen[fp]), os.path.basename(path)))
            continue
        seen[fp] = path
        docs.append((path, metadata, body))

    if stats is not None:
        stats.update(
            found=found,
            indexed=Counter(Path(p).parts[0] for p, _, _ in docs),
            placeholder_files=len(placeholders),
            placeholder_titles=sorted(set(placeholders)),
            repeated_videos=repeats,
            identical=identical,
        )
    return docs


def build_corpus(stats: dict | None = None) -> tuple[list[str], list[dict], int]:
    """Read every source file and split it into (chunks, metadatas, file_count).
    file_count is the number of unique documents actually indexed."""
    docs = load_documents(stats)
    chunks, metadatas = [], []
    for _path, metadata, body in docs:
        for chunk in chunk_text(body):
            chunks.append(chunk)
            metadatas.append(metadata)
    if stats is not None:
        stats["chunks"] = len(chunks)
    return chunks, metadatas, len(docs)
