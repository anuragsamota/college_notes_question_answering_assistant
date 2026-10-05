"""Streamlit web UI: upload notes, then chat with them.

    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from notes_qa import Assistant, Settings
from notes_qa.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileError, load_file

st.set_page_config(page_title="College Notes Q&A", page_icon="📚", layout="wide")


@st.cache_resource
def get_assistant() -> Assistant:
    return Assistant(Settings())


assistant = get_assistant()

with st.sidebar:
    st.header("Your notes")
    uploads = st.file_uploader(
        "Upload lecture notes",
        type=sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS),
        accept_multiple_files=True,
    )
    if uploads and st.button("Index uploaded files", type="primary"):
        with st.spinner("Reading and indexing…"):
            for up in uploads:
                suffix = Path(up.name).suffix
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(up.getbuffer())
                    tmp_path = Path(tmp.name)
                try:
                    chunks = assistant.ingest_pages(load_file(tmp_path, display_name=up.name))
                    st.success(f"{up.name}: {len(chunks)} chunks")
                except (UnsupportedFileError, ValueError) as exc:
                    st.error(f"{up.name}: {exc}")
                finally:
                    tmp_path.unlink(missing_ok=True)

    docs = assistant.documents()
    if docs:
        st.caption(f"{len(docs)} documents · {len(assistant.chunks)} chunks")
        for d in docs:
            col1, col2 = st.columns([5, 1])
            pages = f" · {d.pages} pp." if d.pages else ""
            col1.write(f"**{d.source}**  \n{d.chunks} chunks{pages}")
            if col2.button("✕", key=f"rm-{d.doc_id}", help=f"Remove {d.source}"):
                assistant.remove(d.source)
                st.rerun()
    else:
        st.info("No notes indexed yet.")

    show_quotes = st.toggle("Show cited passages", value=True)
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("📚 College Notes Q&A")
st.caption("Answers come only from the notes you upload, with citations to the "
           "exact passages used.")

if "messages" not in st.session_state:
    st.session_state.messages = []


def render_answer(answer) -> None:
    if answer.status == "answered":
        st.markdown(answer.text)
    else:
        st.warning(answer.text)
    for w in answer.warnings:
        st.caption(f"⚠️ {w}")
    if answer.cited_sources:
        with st.expander("Sources", expanded=False):
            for n, hit in answer.cited_sources:
                st.markdown(f"**[{n}] {hit.chunk.location}**")
                if show_quotes:
                    for c in answer.citations:
                        if c.number == n and c.cited_text.strip():
                            st.markdown(f"> {c.cited_text.strip()}")


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_answer(msg["answer"])

if question := st.chat_input("Ask a question about your notes"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    history = [
        (m_q["content"], m_a["answer"].text)
        for m_q, m_a in zip(st.session_state.messages[:-1:2], st.session_state.messages[1::2])
        if m_a["answer"].status == "answered"
    ][-3:]
    with st.chat_message("assistant"):
        with st.spinner("Searching your notes…"):
            answer = assistant.ask(question, history)
        render_answer(answer)
    st.session_state.messages.append({"role": "assistant", "answer": answer})
