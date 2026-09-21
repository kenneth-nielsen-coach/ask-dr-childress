#!/usr/bin/env python3
"""
Precompute the embedding index for the Dr. Childress corpus and save it
to data/index/.

Why this exists
----------------
app.py used to call SentenceTransformer.encode() on every chunk of the
corpus (tens of thousands of them) inside @st.cache_resource. That cache
only lives as long as the running process — every time the Streamlit
Cloud container rebuilds (waking from sleep, a new deploy, a reboot),
the cache is gone and the app had to re-embed the entire corpus from
scratch before it could answer a single question. On a free-tier
single-core machine that's several minutes of pure CPU work, on every
cold start.

Run this script once locally whenever the source content changes (new
transcripts, blog posts, etc.) and commit the resulting data/index/
files. app.py then just loads them from disk, so a cold start only has
to embed the user's one question, which is effectively instant.

Usage:
    pip install -r requirements.txt
    python build_embeddings.py
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from indexing import EMBED_MODEL, TRANSCRIPTS_DIRS, build_corpus

INDEX_DIR = Path("data/index")


def print_report(stats: dict, file_count: int, chunk_count: int) -> None:
    """Show exactly what is going into the index, so gaps are easy to spot."""
    print("\nWhat is going into the index")
    print(f"  {'folder':26} {'found':>6} {'indexed':>8}")
    for d in TRANSCRIPTS_DIRS:
        print(f"  {d:26} {stats['found'].get(d, 0):6d} {stats['indexed'].get(d, 0):8d}")
        if stats["found"].get(d, 0) == 0:
            print(f"    WARNING: no .md files found in {d}/")
    print(f"  {'TOTAL':26} {sum(stats['found'].values()):6d} {file_count:8d}   -> {chunk_count} chunks")
    if stats["repeated_videos"]:
        print(f"\n  Dropped {stats['repeated_videos']} repeat copies of videos that sit in several playlist folders.")
    if stats["identical"]:
        print(f"  Dropped {len(stats['identical'])} file(s) with text identical to another file:")
        for kept, dropped in stats["identical"][:10]:
            print(f"    '{dropped}' is the same as '{kept}'")
    if stats["placeholder_files"]:
        titles = stats["placeholder_titles"]
        print(f"  Skipped {stats['placeholder_files']} 'no transcript' placeholder file(s) for {len(titles)} video(s) - their captions could not be downloaded yet:")
        for title in titles[:10]:
            print(f"    {title}")
    print()


def main():
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Reading and chunking source files...")
    stats: dict = {}
    chunks, metadatas, file_count = build_corpus(stats)
    print(f"  {file_count} files -> {len(chunks)} chunks")
    print_report(stats, file_count, len(chunks))

    if not chunks:
        print("No chunks found — check that your transcript folders contain .md files.")
        return

    print("Encoding chunks (this is the slow part; it only needs to run once)...")
    start = time.time()
    embeddings = np.asarray(embedder.encode(chunks, show_progress_bar=True, batch_size=64), dtype=np.float32)
    print(f"  done in {time.time() - start:.1f}s")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    # Stored as float16: half the size (about 24 MB instead of 48 MB), far from GitHub's
    # 50 MB warning / 100 MB limit, with no visible effect on search results.
    # app.py converts it back to float32 when it loads the file.
    np.save(INDEX_DIR / "embeddings.npy", embeddings.astype(np.float16))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    (INDEX_DIR / "metadatas.json").write_text(json.dumps(metadatas), encoding="utf-8")
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps({
            "file_count": file_count,
            "chunk_count": len(chunks),
            "embed_model": EMBED_MODEL,
            "embeddings_dtype": "float16",
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved index to {INDEX_DIR}/:")
    print("  embeddings.npy, chunks.json, metadatas.json, manifest.json")
    print("Commit these files so app.py can load the index directly instead of rebuilding it.")


if __name__ == "__main__":
    main()
