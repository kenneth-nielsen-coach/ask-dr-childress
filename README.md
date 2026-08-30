# 🧠 Ask Dr. Childress

> Ask any question and get answers drawn from Dr. Childress's video transcripts and blog posts.

## About

This app answers questions based on content from [Dr. Craig Childress](https://www.youtube.com/@dr.c.a.childress673).
The worlds leading clinical psychologist specializing in parental alienation and attachment-based family therapy.
This is a collection of transcrips for the videos that Dr. Childress has published
on his youtube channel: https://www.youtube.com/@dr.c.a.childress673 
and it also includes  the his blog https://drcraigchildressblog.com/
and his substack https://drchildress.substack.com/

Try using this to find the video where he talks about a topic that your question adresses:
https://ask-dr-childress.streamlit.app/


## Sources

- 📺 [YouTube channel](https://www.youtube.com/@dr.c.a.childress673)
- 📝 [Blog](https://drcraigchildressblog.com)
-  [substack](https://drchildress.substack.com/)
- 📄 Papers and book chapters (`childress_papers/`)
- 🌐 Consulting site pages (`childress_pages/`) — converted from PDF with `convert_page_pdfs.py`

## Updating the search index

Whenever transcripts, blog posts, Substack posts, papers, or site pages are
added or changed, regenerate the embedding index and commit the result so
the deployed app doesn't have to rebuild it from scratch on every cold
start:

```
pip install -r requirements.txt
python build_embeddings.py
git add data/index
git commit -m "Update embedding index"
```

This writes `data/index/embeddings.npy`, `chunks.json`, `metadatas.json`,
and `manifest.json`. `app.py` loads these directly at startup; if they're
missing it falls back to building the index live, which is much slower
and is meant for local development only.

New PDFs dropped into `childress_pages/` need converting to `.md` first —
run `python convert_page_pdfs.py` (requires `pip install pypdf`), then
re-run `build_embeddings.py`.