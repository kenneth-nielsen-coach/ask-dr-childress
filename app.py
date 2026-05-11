"""
Dr. Childress Transcript Chatbot
---------------------------------
Free stack:
  - Groq API (free) for LLM
  - sentence-transformers (free) for embeddings
  - ChromaDB (in-memory) for vector search
  - Streamlit Cloud (free) for hosting

Setup:
  1. Add GROQ_API_KEY to Streamlit Cloud secrets
  2. Put transcript .md files in ./transcripts/ (subfolders OK)
  3. Deploy to Streamlit Cloud
"""

import os
import glob
import streamlit as st
from groq import Groq
import chromadb
from chromadb.utils import embedding_functions

TRANSCRIPTS_DIR = "transcripts"
MODEL = "llama-3.1-70b-versatile"
COLLECTION_NAME = "childress"
TOP_K = 5  # number of transcript chunks to include per answer


# ── Load and index transcripts ─────────────────────────────────────────────

@st.cache_resource(show_spinner="Indexing transcripts — this takes ~30 seconds on first load...")
def build_index():
    """Load all .md files, split into chunks, embed and store in ChromaDB."""
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"  # small, fast, runs on CPU
    )
    client = chromadb.Client()
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=ef,
    )

    docs, ids, metadatas = [], [], []
    files = glob.glob(f"{TRANSCRIPTS_DIR}/**/*.md", recursive=True)

    for path in files:
        text = open(path, encoding="utf-8", errors="replace").read()
        # Extract metadata from file header
        title = "Unknown"
        url = ""
        date = ""
        playlist = ""
        for line in text.splitlines()[:10]:
            if line.startswith("# "):
                title = line[2:].strip()
            elif line.startswith("**Link:**"):
                url = line.replace("**Link:**", "").strip()
            elif line.startswith("**Date:**"):
                date = line.replace("**Date:**", "").strip()
            elif line.startswith("**Playlist:**"):
                playlist = line.replace("**Playlist:**", "").strip()

        # Split into chunks of ~800 chars with overlap
        body = text.split("---\n", 1)[-1].strip()
        chunks = chunk_text(body, size=800, overlap=100)

        for i, chunk in enumerate(chunks):
            doc_id = f"{os.path.basename(path)}__chunk{i}"
            docs.append(chunk)
            ids.append(doc_id)
            metadatas.append({
                "title": title,
                "url": url,
                "date": date,
                "playlist": playlist,
                "file": path,
            })

    if docs:
        # ChromaDB has a max batch size of 5000
        batch = 500
        for i in range(0, len(docs), batch):
            collection.add(
                documents=docs[i:i+batch],
                ids=ids[i:i+batch],
                metadatas=metadatas[i:i+batch],
            )

    return collection, len(files)


def chunk_text(text: str, size: int = 800, overlap: int = 100) -> list[str]:
    """Split text into overlapping chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start += size - overlap
    return chunks or [text]


# ── Groq LLM call ──────────────────────────────────────────────────────────

def ask_groq(question: str, context_chunks: list[dict], history: list) -> str:
    """Send question + retrieved context to Groq and return the answer."""
    api_key = st.secrets.get("GROQ_API_KEY") or os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "⚠️ No GROQ_API_KEY found. Add it in Streamlit Cloud → Settings → Secrets."

    client = Groq(api_key=api_key)

    # Build context block from retrieved chunks
    context_text = ""
    seen_titles = set()
    for chunk in context_chunks:
        meta = chunk["metadata"]
        title = meta.get("title", "Unknown")
        url = meta.get("url", "")
        date = meta.get("date", "")
        if title not in seen_titles:
            seen_titles.add(title)
            context_text += f"\n\n---\nVideo: {title} ({date})\nURL: {url}\n"
        context_text += chunk["document"] + "\n"

    system_prompt = f"""You are a helpful assistant that answers questions based exclusively on transcripts from Dr. Craig Childress, a clinical psychologist specializing in parental alienation and attachment-based family therapy.

Answer clearly and accurately using only the transcript content provided below. If the answer is not in the transcripts, say so honestly. Always cite which video(s) your answer comes from.

TRANSCRIPT CONTEXT:
{context_text}
"""

    messages = [{"role": "system", "content": system_prompt}]
    # Include recent chat history (last 6 messages)
    for msg in history[-6:]:
        messages.append({"role": msg["role"], "content": msg["content"]})
    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ── Streamlit UI ───────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Dr. Childress Q&A",
    page_icon="🧠",
    layout="centered",
)

st.title("🧠 Dr. Childress – Video Q&A")
st.caption(
    "Ask any question and get answers based on Dr. Childress's video transcripts. "
    "Answers include links to the source videos."
)

# Build/load index
collection, file_count = build_index()
st.sidebar.success(f"✅ {file_count} transcript files indexed")
st.sidebar.markdown(
    "**About**\n\nThis chatbot answers questions using transcripts from "
    "[Dr. Craig Childress](https://www.youtube.com/@dr.c.a.childress673)'s YouTube channel."
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if question := st.chat_input("Ask a question about Dr. Childress's work..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve relevant chunks
    results = collection.query(
        query_texts=[question],
        n_results=TOP_K,
        include=["documents", "metadatas"],
    )
    chunks = [
        {"document": doc, "metadata": meta}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]

    # Get answer from Groq
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = ask_groq(question, chunks, st.session_state.messages)
        st.markdown(answer)

    st.session_state.messages.append({"role": "assistant", "content": answer})
