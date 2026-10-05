"""Runtime settings, overridable through environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(f"NOTES_QA_{name}", default)


@dataclass
class Settings:
    # Where the index (chunks + optional embeddings) is persisted.
    index_dir: Path = field(default_factory=lambda: Path(_env("INDEX_DIR", ".notes_index")))

    # Chunking: target size in characters and sentence overlap between chunks.
    chunk_chars: int = field(default_factory=lambda: int(_env("CHUNK_CHARS", "1200")))
    chunk_overlap_sentences: int = field(default_factory=lambda: int(_env("CHUNK_OVERLAP", "2")))

    # Retrieval.
    candidates: int = 30          # fused candidates considered before reranking
    top_k: int = field(default_factory=lambda: int(_env("TOP_K", "6")))
    mmr_lambda: float = 0.7       # 1.0 = pure relevance, lower = more diverse context
    # Relevance gate: when the top chunks cover less than this (IDF-weighted)
    # share of the question's terms, the assistant declines without calling the
    # model instead of letting it answer from general knowledge.
    min_term_coverage: float = field(default_factory=lambda: float(_env("MIN_COVERAGE", "0.5")))
    # With dense embeddings, a top hit at or above this cosine also passes the gate.
    min_dense_similarity: float = field(default_factory=lambda: float(_env("MIN_DENSE_SIM", "0.45")))
    # Dense embeddings added to hybrid retrieval: "tfidf" (none, no extra deps),
    # "ollama" (an Ollama embedding model) or "sentence-transformers".
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "tfidf"))
    # Embedding model; defaults to nomic-embed-text (ollama) / all-MiniLM-L6-v2.
    embed_model: str = field(default_factory=lambda: _env("EMBED_MODEL", ""))

    # Ollama servers (local and/or LAN), see notes_qa/servers.py.
    servers_file: Path = field(
        default_factory=lambda: Path(_env("SERVERS_FILE", "ollama_servers.json")))
    # "name=host,name=host" defines the server list for this run instead of the file.
    ollama_servers: str = field(default_factory=lambda: _env("OLLAMA_SERVERS", ""))
    # Server name or "auto" (failover in list order); empty = the file's default.
    ollama_server: str = field(default_factory=lambda: _env("SERVER", ""))
    # Address of the built-in "local" server when no servers are configured.
    ollama_host: str | None = field(default_factory=lambda: os.environ.get("OLLAMA_HOST"))
    # Fail fast on a switched-off LAN machine, but allow slow generation.
    connect_timeout: float = field(default_factory=lambda: float(_env("CONNECT_TIMEOUT", "3")))
    request_timeout: float = field(default_factory=lambda: float(_env("REQUEST_TIMEOUT", "300")))

    # Generation. Default chat model for servers that don't name their own.
    model: str = field(default_factory=lambda: _env("MODEL", "llama3.1:8b"))
    temperature: float = field(default_factory=lambda: float(_env("TEMPERATURE", "0.1")))
    # Ollama's default context window is small; 6 excerpts + prompt need ~3-4k tokens.
    num_ctx: int = field(default_factory=lambda: int(_env("NUM_CTX", "8192")))
    max_tokens: int = field(default_factory=lambda: int(_env("MAX_TOKENS", "1024")))
    # A cited sentence counts as verified when this share of its content words
    # appears in the excerpt(s) it cites.
    min_citation_overlap: float = field(
        default_factory=lambda: float(_env("MIN_CITATION_OVERLAP", "0.6")))

    @property
    def resolved_embed_model(self) -> str:
        if self.embed_model:
            return self.embed_model
        return "nomic-embed-text" if self.embedder == "ollama" else "all-MiniLM-L6-v2"
