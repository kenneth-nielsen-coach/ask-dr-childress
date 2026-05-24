"""
Dr. Childress Transcript Chatbot
---------------------------------
Free stack:
  - Groq API (free) for LLM
  - sentence-transformers (free) for embeddings
  - numpy cosine similarity (no ChromaDB needed)
  - Streamlit Cloud (free) for hosting

Setup:
  1. Add GROQ_API_KEY to Streamlit Cloud secrets
  2. Put transcript .md files in ./childress_transcripts/ (subfolders OK)
  3. Deploy to Streamlit Cloud
"""

import os
import re
import glob
import numpy as np
import streamlit as st
from groq import Groq
from pathlib import Path
from sentence_transformers import SentenceTransformer

TRANSCRIPTS_DIRS = [
    "childress_transcripts",   # YouTube transcripts
    "childress_blog",          # Blog posts
    "childress_substack",      # Substack posts
]
MODEL = "llama-3.3-70b-versatile"   # llama-3.1-70b-versatile is deprecated on Groq
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
MAX_CONTEXT_CHARS = 6000            # cap injected transcript text to avoid token limit errors


# ── Load and index transcripts ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Indexing transcripts — takes ~30 seconds on first load...")
def build_index():
    embedder = SentenceTransformer(EMBED_MODEL)
    chunks, metadatas = [], []

    files = []
    for d in TRANSCRIPTS_DIRS:
        files += glob.glob(f"{d}/**/*.md", recursive=True)

    for path in files:
        text = open(path, encoding="utf-8", errors="replace").read()

        title, url, date, playlist = "Unknown", "", "", ""
        for line in text.splitlines()[:10]:
            if line.startswith("# "):
                title = line[2:].strip()
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
        for chunk in chunk_text(body):
            chunks.append(chunk)
            metadatas.append({"title": title, "url": url, "date": date, "playlist": playlist})

    embeddings = embedder.encode(chunks, show_progress_bar=False, batch_size=64)
    return embedder, np.array(embeddings), chunks, metadatas, len(files)


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    chunks, start = [], 0
    while start < len(text):
        chunks.append(text[start:start + size])
        start += size - overlap
    return chunks or [text]


def search(query: str, embedder, embeddings, chunks, metadatas) -> list[dict]:
    if len(embeddings) == 0:
        return []
    q_vec = embedder.encode([query])
    norms = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(q_vec)
    scores = (embeddings @ q_vec.T).flatten() / np.maximum(norms, 1e-9)
    top_idx = np.argsort(scores)[::-1][:TOP_K]
    return [{"document": chunks[i], "metadata": metadatas[i], "score": scores[i]} for i in top_idx]


# ── Groq LLM call ──────────────────────────────────────────────────────────

def ask_groq(question: str, context_chunks: list[dict], history: list) -> str:
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ No GROQ_API_KEY found. Add it in Streamlit Cloud → Settings → Secrets."

    client = Groq(api_key=api_key)

    # Build context with hard character cap to prevent token limit errors
    context_text = ""
    seen = set()
    for chunk in context_chunks:
        m = chunk["metadata"]
        if m["title"] not in seen:
            seen.add(m["title"])
            context_text += f"\n\n---\nVideo: {m['title']} ({m['date']})\nURL: {m['url']}\n"
        context_text += chunk["document"] + "\n"
        if len(context_text) >= MAX_CONTEXT_CHARS:
            context_text = context_text[:MAX_CONTEXT_CHARS] + "\n[...truncated for length...]"
            break

    system_prompt = f"""You are a helpful assistant that answers questions based exclusively on content from Dr. Craig Childress, a clinical psychologist specializing in parental alienation and attachment-based family therapy.

Your sources include YouTube video transcripts (with timestamps), blog posts, and Substack articles. Answer clearly and accurately using only the content provided. If the answer is not in the sources, say so honestly.

When citing a video, always include the timestamp link in the format [MM:SS](url) so the user can jump directly to that moment. If multiple timestamps are relevant, include them all.

LANGUAGE RULE: Detect the language of the user's question and respond in that same language. The source material is in English — translate your answer into the user's language while keeping any cited titles and URLs in their original English form.

SOURCE CONTENT:
{context_text}
"""

    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        # Skip messages with missing or empty content to avoid BadRequestError
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Read title and caption from README.md ──────────────────────────────────

def read_readme() -> tuple[str, str]:
    """
    Read title and caption from README.md.
    Expected format:
        # Your App Title
        > Your caption text here
    Falls back to defaults if README.md is missing or fields not found.
    """
    default_title   = "🧠 Dr. Childress – Q&A"
    default_caption = "Ask any question in your language and get answers drawn from Dr. Childress's video transcripts and his blog posts."
    try:
        readme = Path("README.md").read_text(encoding="utf-8")
        title_match   = re.search(r"^#\s+(.+)$", readme, re.MULTILINE)
        caption_match = re.search(r"^>\s+(.+)$", readme, re.MULTILINE)
        title   = title_match.group(1).strip()   if title_match   else default_title
        caption = caption_match.group(1).strip() if caption_match else default_caption
        return title, caption
    except FileNotFoundError:
        return default_title, default_caption


# ── Streamlit UI ───────────────────────────────────────────────────────────

APP_TITLE, APP_CAPTION = read_readme()

st.set_page_config(page_title=APP_TITLE, page_icon="🧠", layout="centered")
st.title(APP_TITLE)
st.caption(APP_CAPTION)

embedder, embeddings, chunks, metadatas, file_count = build_index()

if file_count == 0:
    st.error("⚠️ No transcript files found. Check that your transcript folders exist and contain .md files.")
    st.stop()

st.sidebar.success(f"✅ {file_count} files indexed")
st.sidebar.markdown(
    "**Sources**\n\n"
    "- 📺 [Dr. Childress YouTube](https://www.youtube.com/@dr.c.a.childress673)\n"
    "- 📝 [Dr. Childress Blog](https://drcraigchildressblog.com)\n"
    "- 📧 [Dr. Childress Substack](https://drcachildress.substack.com)\n"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question in any language / Pregunta en cualquier idioma / Posez votre question..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = search(question, embedder, embeddings, chunks, metadatas)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                answer = ask_groq(question, results, st.session_state.messages)
            except Exception as e:
                answer = f"**Error:** {e}"
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
