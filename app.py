"""
Setup:
  1. Add GROQ_API_KEY to Streamlit Cloud secrets (free key from console.groq.com)
  2. Put transcript .md files in ./childress_transcripts/ (subfolders OK)
  3. Deploy to Streamlit Cloud
"""

import os
import re
import json
import numpy as np
import streamlit as st
import groq
from groq import Groq
from pathlib import Path
from sentence_transformers import SentenceTransformer

from indexing import EMBED_MODEL, build_corpus

MODEL = "llama-3.3-70b-versatile"   # free on Groq
TOP_K = 12
# Groq's free tier limits tokens per minute, and the request counts both the
# prompt and the requested answer length. The old Groq version used 6000 chars;
# 9000 chars (~2,300 tokens) gives richer answers while staying well inside it.
# If you see "rate limit" messages, lower these. If answers feel thin and you
# never hit limits, raise them gradually.
MAX_CONTEXT_CHARS = 9000
MAX_ANSWER_TOKENS = 1500
MAX_HISTORY_CHARS = 1500    # per earlier message, so long chats don't eat the token budget

# Chunks scoring below this cosine similarity are treated as "not actually
# about the question" and left out of the model's context, so it says it
# couldn't find anything instead of answering from a weak/irrelevant match.
# This is a heuristic for all-MiniLM-L6-v2 — if real questions get rejected
# that shouldn't be, or weak matches still slip through, tune it up or down.
MIN_SIMILARITY = 0.25

INDEX_DIR = Path("data/index")


# ── Load and index transcripts ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading index...")
def build_index():
    embedder = SentenceTransformer(EMBED_MODEL)

    precomputed = _load_precomputed_index()
    if precomputed is not None:
        embeddings, chunks, metadatas, file_count = precomputed
        return embedder, embeddings, chunks, metadatas, file_count

    # Fallback: no precomputed index on disk, so build it live (slow — this
    # is meant for local development only). Run `python build_embeddings.py`
    # once and commit data/index/ so deploys never have to hit this path.
    with st.spinner("No precomputed index found — building one now, this can take a few minutes..."):
        chunks, metadatas, file_count = build_corpus()
        embeddings = (
            np.asarray(embedder.encode(chunks, show_progress_bar=False, batch_size=64), dtype=np.float32)
            if chunks else np.array([])
        )
    return embedder, embeddings, chunks, metadatas, file_count


def _load_precomputed_index():
    """Load embeddings/chunks/metadatas from data/index/ if they exist."""
    manifest_path = INDEX_DIR / "manifest.json"
    if not manifest_path.exists():
        return None
    try:
        # Saved as float16 to keep the file small; compute in float32.
        embeddings = np.load(INDEX_DIR / "embeddings.npy").astype(np.float32)
        chunks = json.loads((INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        metadatas = json.loads((INDEX_DIR / "metadatas.json").read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        return embeddings, chunks, metadatas, manifest["file_count"]
    except Exception as e:
        st.warning(f"Found data/index/ but couldn't load it ({e}) — rebuilding live instead.")
        return None


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

    # Drop chunks that aren't actually similar enough to the question. Without
    # this, an off-topic or oddly-phrased question still gets its "top 5"
    # chunks stuffed into context regardless of how weak the match is, which
    # invites the model to answer from irrelevant material instead of admitting
    # the sources don't cover it.
    relevant_chunks = [c for c in context_chunks if c["score"] >= MIN_SIMILARITY]

    # Build context with hard character cap to prevent token limit errors
    context_text = ""
    seen = set()
    for chunk in relevant_chunks:
        m = chunk["metadata"]
        if m["title"] not in seen:
            seen.add(m["title"])
            context_text += f"\n\n---\nVideo: {m['title']} ({m['date']})\nURL: {m['url']}\n"
        context_text += chunk["document"] + "\n"
        if len(context_text) >= MAX_CONTEXT_CHARS:
            context_text = context_text[:MAX_CONTEXT_CHARS] + "\n[...truncated for length...]"
            break

    source_block = (
        f"SOURCE CONTENT:\n{context_text}"
        if context_text
        else "SOURCE CONTENT:\n(No passages in the corpus were a close enough match to this question.)"
    )

    system_prompt = f"""You are a helpful assistant that answers questions based exclusively on content from Dr. Craig Childress, a clinical psychologist specializing in parental alienation and attachment-based family therapy.

Your sources include YouTube video transcripts (with timestamps), blog posts, and Substack articles. Answer clearly and accurately using only the content provided. If the source content below is empty or doesn't address the question, say plainly that you couldn't find anything from Dr. Childress on that topic — do not answer from general knowledge or guess.

DEPTH AND PRECISION:
- Give a thorough, well-developed answer, not a short summary. Aim to actually teach the concept, not just gesture at it.
- Synthesize across ALL the relevant source passages provided below, not just the first or most obvious one — different passages often cover different facets of the same topic, and a good answer draws connections between them.
- Explain the underlying clinical concept or mechanism Dr. Childress is describing (e.g. the attachment dynamic, the psychological process, the diagnostic reasoning), not merely what he said in passing.
- Where multiple videos, blog posts, or papers address the same topic, say so explicitly and note any differences in framing or emphasis between them.
- Prefer being concrete: use Dr. Childress's own terminology, examples, and reasoning from the source material rather than paraphrasing it into generic language.
- If the retrieved passages only partially answer the question, answer the part they do cover thoroughly, and say plainly which part of the question isn't addressed by the available material — don't pad with generic filler to seem complete.

When citing a video, always include the timestamp link in the format [MM:SS](url) so the user can jump directly to that moment. If multiple timestamps are relevant, include them all.

LANGUAGE RULE: Detect the language of the user's question and respond in that same language. The source material is in English — translate your answer into the user's language while keeping any cited titles and URLs in their original English form.

{source_block}
"""

    # Groq uses the OpenAI-style format: the system prompt is the first message.
    messages = [{"role": "system", "content": system_prompt}]
    for msg in history[-6:]:
        # Skip messages with missing or empty content to avoid BadRequestError
        role = msg.get("role")
        content = msg.get("content")
        if role in ("user", "assistant") and isinstance(content, str) and content.strip():
            messages.append({"role": role, "content": content[:MAX_HISTORY_CHARS]})
    messages.append({"role": "user", "content": question})

    try:
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            temperature=0.3,
            max_tokens=MAX_ANSWER_TOKENS,
        )
    except groq.RateLimitError:
        return "⚠️ The free AI service is busy right now (rate limit). Please wait a minute and try again."
    return response.choices[0].message.content


# ── Read title and caption from README.md ──────────────────────────────────


def read_readme() -> tuple[str, str]:
    default_title   = "🧠 Dr. Childress – Q&A"
    default_caption = "Ask any question in your language and get answers drawn from Dr. Childress's video transcripts, his blogs and substack posts."
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
    "- 🧪 Kenneth's Folder (containing files to include in the search.)"
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
