"""On-disk persistence for the chunk index."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from notes_qa.chunking import Chunk

_CHUNKS = "chunks.json"
_EMBEDDINGS = "embeddings.npy"


@dataclass
class DocumentInfo:
    doc_id: str
    source: str
    chunks: int
    pages: int


class IndexStore:
    """A directory holding ``chunks.json`` and, for dense retrieval, ``embeddings.npy``."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def load(self, embedder_name: str) -> tuple[list[Chunk], np.ndarray | None]:
        path = self.directory / _CHUNKS
        if not path.exists():
            return [], None
        data = json.loads(path.read_text(encoding="utf-8"))
        chunks = [Chunk.from_dict(c) for c in data["chunks"]]
        embeddings = None
        emb_path = self.directory / _EMBEDDINGS
        if data.get("embedder") == embedder_name and emb_path.exists():
            embeddings = np.load(emb_path)
            if len(embeddings) != len(chunks):
                embeddings = None
        return chunks, embeddings

    def save(self, chunks: list[Chunk], embedder_name: str,
             embeddings: np.ndarray | None) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        payload = {"version": 1, "embedder": embedder_name,
                   "chunks": [c.to_dict() for c in chunks]}
        tmp = self.directory / (_CHUNKS + ".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.directory / _CHUNKS)
        emb_path = self.directory / _EMBEDDINGS
        if embeddings is not None:
            np.save(emb_path, embeddings)
        elif emb_path.exists():
            emb_path.unlink()


def summarize_documents(chunks: list[Chunk]) -> list[DocumentInfo]:
    docs: dict[str, DocumentInfo] = {}
    pages: dict[str, set] = {}
    for c in chunks:
        info = docs.setdefault(c.doc_id, DocumentInfo(c.doc_id, c.source, 0, 0))
        info.chunks += 1
        pages.setdefault(c.doc_id, set()).add(c.page)
    for doc_id, info in docs.items():
        info.pages = len({p for p in pages[doc_id] if p is not None})
    return sorted(docs.values(), key=lambda d: d.source.lower())
