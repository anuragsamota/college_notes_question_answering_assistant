"""High-level assistant: ingest notes, then ask questions about them."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from notes_qa.chunking import Chunk, Chunker
from notes_qa.config import Settings
from notes_qa.generator import Answer, AnswerGenerator
from notes_qa.loaders import Page, iter_note_files, load_file
from notes_qa.retrieval import (Embedder, HybridRetriever, RetrievalResult,
                                OllamaEmbedder, SentenceTransformerEmbedder, analyze)
from notes_qa.store import DocumentInfo, IndexStore, summarize_documents


class Assistant:
    def __init__(self, settings: Settings | None = None, *,
                 generator: AnswerGenerator | None = None,
                 embedder: Embedder | None = None):
        self.settings = settings or Settings()
        s = self.settings
        self.store = IndexStore(s.index_dir)
        self.chunker = Chunker(s.chunk_chars, s.chunk_overlap_sentences)
        self.embedder = embedder
        if self.embedder is None and s.embedder == "sentence-transformers":
            self.embedder = SentenceTransformerEmbedder(s.resolved_embed_model)
        elif self.embedder is None and s.embedder == "ollama":
            self.embedder = OllamaEmbedder(s.resolved_embed_model, s.ollama_host)
        self.generator = generator or AnswerGenerator(
            model=s.model, host=s.ollama_host, temperature=s.temperature,
            num_ctx=s.num_ctx, max_tokens=s.max_tokens,
            min_overlap=s.min_citation_overlap)
        self.chunks, embeddings = self.store.load(self._embedder_name)
        self._rebuild(embeddings)

    @property
    def _embedder_name(self) -> str:
        if self.embedder is None:
            return "tfidf"
        # Include the model so switching models forces fresh embeddings.
        return f"{self.settings.embedder}:{self.settings.resolved_embed_model}"

    def _rebuild(self, embeddings: np.ndarray | None = None) -> None:
        self.retriever = HybridRetriever(self.chunks, embedder=self.embedder,
                                         embeddings=embeddings)

    def _persist(self) -> None:
        self.store.save(self.chunks, self._embedder_name, self.retriever.embeddings)

    # ------------------------------------------------------------ documents
    def documents(self) -> list[DocumentInfo]:
        return summarize_documents(self.chunks)

    def ingest_pages(self, pages: list[Page]) -> list[Chunk]:
        """Index one document's pages, replacing an earlier version of the same file."""
        new_chunks = self.chunker.chunk(pages)
        if not new_chunks:
            return []
        source = new_chunks[0].source
        self.chunks = [c for c in self.chunks if c.source != source] + new_chunks
        self._rebuild()
        self._persist()
        return new_chunks

    def ingest_paths(self, paths: Iterable[str | Path]) -> dict[str, int]:
        added: dict[str, int] = {}
        for path in iter_note_files(paths):
            chunks = self.ingest_pages(load_file(path))
            added[path.name] = len(chunks)
        return added

    def remove(self, source: str) -> bool:
        before = len(self.chunks)
        self.chunks = [c for c in self.chunks if c.source != source]
        if len(self.chunks) == before:
            return False
        self._rebuild()
        self._persist()
        return True

    def clear(self) -> None:
        self.chunks = []
        self._rebuild()
        self._persist()

    # ------------------------------------------------------------ questions
    def retrieve(self, question: str, history: Sequence[tuple[str, str]] = ()) -> RetrievalResult:
        s = self.settings
        query = question
        # Short follow-ups ("what about its disadvantages?") rarely retrieve well
        # alone, so borrow the topic from the previous question.
        if history and len(analyze(question, query=True)) < 4:
            query = f"{history[-1][0]} {question}"
        return self.retriever.search(
            query, top_k=s.top_k, candidates=s.candidates, mmr_lambda=s.mmr_lambda,
            min_term_coverage=s.min_term_coverage,
            min_dense_similarity=s.min_dense_similarity)

    def ask(self, question: str, history: Sequence[tuple[str, str]] = ()) -> Answer:
        question = question.strip()
        if not self.chunks:
            return Answer(question, "No notes are indexed yet. Upload or ingest some notes "
                          "first.", "no_relevant_notes")
        result = self.retrieve(question, history)
        if not result.sufficient:
            msg = "I couldn't find anything in your notes that covers this question."
            if result.missing_terms:
                msg += " Terms not found: " + ", ".join(result.missing_terms[:8]) + "."
            return Answer(question, msg, "no_relevant_notes", sources=result.hits)
        return self.generator.generate(question, result.hits, history)
