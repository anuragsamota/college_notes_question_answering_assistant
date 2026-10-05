"""Streamlit web UI: upload notes, then chat with them.

    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from notes_qa import Assistant, Settings
from notes_qa.loaders import SUPPORTED_EXTENSIONS, UnsupportedFileError, load_file
from notes_qa.servers import AUTO, NoServerAvailable, OllamaServer, ServerConfigError

st.set_page_config(page_title="College Notes Q&A", page_icon="📚", layout="wide")


@st.cache_resource
def get_assistant() -> Assistant:
    return Assistant(Settings())


assistant = get_assistant()
registry = assistant.servers
router = assistant.router


def refresh_status() -> None:
    st.session_state.server_status = {
        s.server.name: s for s in router.check_all(timeout=assistant.settings.connect_timeout)}


def server_panel() -> None:
    st.header("Ollama server")
    if "server_status" not in st.session_state:
        refresh_status()
    status = st.session_state.server_status
    read_only = registry.from_env

    def label(name: str) -> str:
        if name == AUTO:
            order = " → ".join(s.name for s in registry.servers if s.enabled)
            return f"Auto · failover ({order})" if order else "Auto · failover"
        st_ = status.get(name)
        return f"{'🟢' if st_ and st_.reachable else '🔴'} {name}"

    options = [AUTO] + registry.names
    current = router.selection or registry.default
    choice = st.selectbox("Answer with", options, format_func=label,
                          index=options.index(current) if current in options else 0)
    if choice != current:
        registry.select(choice)
        router.selection = None
        registry.save()
        st.rerun()

    for server in registry.servers:
        st_ = status.get(server.name)
        with st.expander(f"{'🟢' if st_ and st_.reachable else '🔴'} {server.name}"
                         + ("" if server.enabled else " (disabled)")):
            st.caption(server.host + (" · this computer" if server.is_local else " · network"))
            model = router.chat_model(server)
            if st_ and st_.reachable:
                st.caption(f"Reachable · {st_.latency_ms:.0f} ms · {len(st_.models)} models")
                models = st_.models or [model]
                if model not in models and not st_.has_model(model):
                    st.warning(f"`{model}` is not installed here. "
                               f"Run `ollama pull {model}` on that machine.")
                    models = [model] + models
                picked = st.selectbox("Chat model", models,
                                      index=models.index(model) if model in models else 0,
                                      key=f"model-{server.name}", disabled=read_only)
                if picked != model and not read_only:
                    server.model = picked
                    registry.save()
            else:
                st.caption(f"Unavailable: {st_.error if st_ else 'unknown'}")
                st.caption(f"Chat model: `{model}`")
            if not read_only:
                c1, c2, c3 = st.columns(3)
                if c1.button("Disable" if server.enabled else "Enable", key=f"en-{server.name}"):
                    server.enabled = not server.enabled
                    registry.save()
                    st.rerun()
                if c2.button("Prefer", key=f"up-{server.name}",
                             help="Try this server first in auto mode"):
                    registry.move(server.name, 0)
                    registry.save()
                    st.rerun()
                if c3.button("Remove", key=f"rm-srv-{server.name}"):
                    registry.remove(server.name)
                    if router.selection == server.name:
                        router.selection = None
                    registry.save()
                    st.rerun()

    if st.button("Refresh status"):
        refresh_status()
        st.rerun()

    if read_only:
        st.caption("Servers are set by NOTES_QA_OLLAMA_SERVERS; edit that variable to change them.")
        return
    with st.expander("Add a server"):
        with st.form("add-server", clear_on_submit=True):
            name = st.text_input("Name", placeholder="lab-gpu")
            host = st.text_input("Address", placeholder="192.168.1.50 or http://gpu-box:11434")
            model = st.text_input("Chat model (optional)", placeholder=assistant.settings.model)
            first = st.checkbox("Try this server first")
            if st.form_submit_button("Add"):
                try:
                    registry.upsert(OllamaServer(name, host, model or None))
                    if first:
                        registry.move(name.strip(), 0)
                    registry.save()
                    refresh_status()
                    st.rerun()
                except ServerConfigError as exc:
                    st.error(str(exc))
        st.caption("For a LAN server, start Ollama on that machine with "
                   "`OLLAMA_HOST=0.0.0.0 ollama serve` so it accepts network connections.")


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
                except (UnsupportedFileError, ValueError, NoServerAvailable) as exc:
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

    st.divider()
    server_panel()
    st.divider()
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
    if answer.server:
        st.caption(f"Answered by `{answer.model}` on **{answer.server}**")
    if answer.unsupported:
        with st.expander("Statements not matched to your notes"):
            for sentence in answer.unsupported:
                st.markdown(f"- {sentence}")
    if answer.cited_sources:
        with st.expander("Sources", expanded=False):
            for n, hit in answer.cited_sources:
                st.markdown(f"**[{n}] {hit.chunk.location}**")
                if show_quotes:
                    quotes = dict.fromkeys(c.cited_text.strip() for c in answer.citations
                                           if c.number == n and c.cited_text.strip())
                    for quote in quotes:
                        st.markdown(f"> {quote}")


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
