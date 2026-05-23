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
import glob
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer

TRANSCRIPTS_DIRS = [
    "childress_transcripts",   # YouTube transcripts
    "childress_blog",          # Blog posts
]
MODEL = "llama-3.3-70b-versatile"   # FIX: llama-3.1-70b-versatile is deprecated on Groq
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
MAX_CONTEXT_CHARS = 6000            # FIX: cap injected transcript text to avoid token limit errors

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

        # Parse header metadata
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

    # Build context text with a hard character cap to prevent token limit errors
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

Your sources include YouTube video transcripts and blog posts. Answer clearly and accurately using only the content provided. If the answer is not in the sources, say so honestly. Always cite which video or blog post your answer comes from, including the URL.

SOURCE CONTENT:
{context_text}
"""

    # Build message list: system prompt + last 6 history turns + current question
    messages = [{"role": "system", "content": system_prompt}]

    for msg in history[-6:]:
        # FIX: skip any messages with missing/empty content to avoid BadRequestError
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


# ── Streamlit UI ───────────────────────────────────────────────────────────

st.set_page_config(page_title="Dr. Childress Q&A", page_icon="🧠", layout="centered")
st.title("🧠 Dr. Childress – Video Q&A")
st.caption("Ask any question and get answers drawn from Dr. Childress's video transcripts.")

embedder, embeddings, chunks, metadatas, file_count = build_index()
st.sidebar.success(f"✅ {file_count} files indexed")
st.sidebar.markdown(
    "**Sources**\n\n"
    "- 📺 [Dr. Childress YouTube](https://www.youtube.com/@dr.c.a.childress673)\n"
    "- 📝 [Dr. Childress Blog](https://drcraigchildressblog.com)\n"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if question := st.chat_input("Ask a question about Dr. Childress's work..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    results = search(question, embedder, embeddings, chunks, metadatas)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_groq(question, results, st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
