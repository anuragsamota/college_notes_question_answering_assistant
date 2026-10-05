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
    # "tfidf" (no extra deps) or "sentence-transformers" (needs the dense extra).
    embedder: str = field(default_factory=lambda: _env("EMBEDDER", "tfidf"))
    dense_model: str = field(default_factory=lambda: _env("DENSE_MODEL", "all-MiniLM-L6-v2"))

    # Generation.
    model: str = field(default_factory=lambda: _env("MODEL", "claude-opus-5-5"))
    effort: str = field(default_factory=lambda: _env("EFFORT", "medium"))
    max_tokens: int = 16000
    # Server-side refusal fallback (Claude API only; disable on Bedrock/Vertex/Foundry).
    use_fallbacks: bool = field(default_factory=lambda: _env("FALLBACKS", "1") == "1")
