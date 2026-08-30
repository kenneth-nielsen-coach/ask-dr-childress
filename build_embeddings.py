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

from indexing import EMBED_MODEL, build_corpus

INDEX_DIR = Path("data/index")


def main():
    print(f"Loading embedding model: {EMBED_MODEL}")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Reading and chunking source files...")
    chunks, metadatas, file_count = build_corpus()
    print(f"  {file_count} files -> {len(chunks)} chunks")

    if not chunks:
        print("No chunks found — check that your transcript folders contain .md files.")
        return

    print("Encoding chunks (this is the slow part; it only needs to run once)...")
    start = time.time()
    embeddings = embedder.encode(chunks, show_progress_bar=True, batch_size=64)
    print(f"  done in {time.time() - start:.1f}s")

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    np.save(INDEX_DIR / "embeddings.npy", np.asarray(embeddings, dtype=np.float32))
    (INDEX_DIR / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
    (INDEX_DIR / "metadatas.json").write_text(json.dumps(metadatas), encoding="utf-8")
    (INDEX_DIR / "manifest.json").write_text(
        json.dumps({
            "file_count": file_count,
            "chunk_count": len(chunks),
            "embed_model": EMBED_MODEL,
        }, indent=2),
        encoding="utf-8",
    )

    print(f"\nSaved index to {INDEX_DIR}/:")
    print("  embeddings.npy, chunks.json, metadatas.json, manifest.json")
    print("Commit these files so app.py can load the index directly instead of rebuilding it.")


if __name__ == "__main__":
    main()
