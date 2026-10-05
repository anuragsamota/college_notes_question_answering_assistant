"""Hybrid retrieval over note chunks.

Pipeline for a question:

1. Score every chunk with BM25 (exact keyword match) and TF-IDF cosine
   similarity, plus dense embeddings when ``sentence-transformers`` is enabled.
2. Fuse the rankings with Reciprocal Rank Fusion so no single scorer dominates.
3. Pick the final context with Maximal Marginal Relevance, so near-duplicate
   chunks (overlaps, repeated slides) don't crowd out other relevant sections.
4. Run a relevance gate: if even the best chunk barely relates to the question,
   report the context as insufficient so the assistant can decline instead of
   letting the model answer from general knowledge.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Protocol, Sequence

import numpy as np

from notes_qa.chunking import Chunk

STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been
before being below between both but by can could did do does doing down during
each few for from further had has have having he her here hers herself him
himself his how i if in into is it its itself just me more most my myself no nor
not now of off on once only or other our ours ourselves out over own same she
should so some such than that the their theirs them themselves then there these
they this those through to too under until up very was we were what when where
which while who whom why will with would you your yours yourself yourselves
also may might must shall us via etc ie eg
""".split())

# Words that frame a question but say nothing about its topic.
QUERY_FILLER = frozenset("""
explain describe define definition tell briefly brief mean meaning notes note
lecture lectures give please according discuss elaborate short summary summarize
compute computed calculate calculated find happen happens
""".split())

_TOKEN = re.compile(r"[a-z0-9]+(?:[+#]+|(?:'[a-z]+))?")


def stem(word: str) -> str:
    """Tiny suffix stripper: enough to match 'processes'/'process', 'paging'/'page'."""
    if len(word) <= 3 or word.isdigit():
        return word
    for suffix, repl in (("ies", "y"), ("sses", "ss"), ("ing", ""), ("edly", ""),
                         ("ed", ""), ("ly", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + repl
            break
    else:
        if word.endswith("es") and word[-3:-2] in "sxz" and len(word) > 4:
            word = word[:-2]
        elif word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
    if word.endswith("e") and len(word) >= 4:
        word = word[:-1]
    return word


def _surface_terms(text: str, query: bool) -> list[tuple[str, str]]:
    """(stem, original word) pairs for the content words in ``text``."""
    out = []
    for t in _TOKEN.findall(text.lower().replace("’", "'")):
        t = t.split("'", 1)[0]
        if t in STOPWORDS or (query and t in QUERY_FILLER):
            continue
        if len(t) == 1 and not t.isdigit():
            continue
        out.append((stem(t), t))
    return out


def analyze(text: str, *, query: bool = False) -> list[str]:
    return [s for s, _ in _surface_terms(text, query)]


def _surface_words(query: str, stems: list[str]) -> list[str]:
    surface = dict(reversed(_surface_terms(query, True)))
    return sorted({surface.get(t, t) for t in stems})


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    """Optional dense embedder; requires ``pip install sentence-transformers``."""

    def __init__(self, model_name: str):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "EMBEDDER=sentence-transformers needs `pip install sentence-transformers`"
            ) from exc
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs = self.model.encode(list(texts), normalize_embeddings=True,
                                 show_progress_bar=False)
        return np.asarray(vecs, dtype=np.float32)


class OllamaEmbedder:
    """Dense embedder backed by an Ollama embedding model (e.g. nomic-embed-text)."""

    def __init__(self, model_name: str, host: str | None = None, batch_size: int = 64):
        import ollama

        self.client = ollama.Client(host=host)
        self.model = model_name
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        vecs: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start:start + self.batch_size])
            vecs.extend(self.client.embed(model=self.model, input=batch).embeddings)
        arr = np.asarray(vecs, dtype=np.float32)
        return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-9)


@dataclass
class Hit:
    chunk: Chunk
    score: float        # fused (RRF) score, normalised to [0, 1]
    similarity: float   # TF-IDF cosine with the question


@dataclass
class RetrievalResult:
    query: str
    hits: list[Hit]
    max_similarity: float   # best TF-IDF cosine among hits (diagnostic)
    term_coverage: float    # IDF-weighted share of question terms found in top hits
    sufficient: bool
    missing_terms: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(self, chunks: list[Chunk], *, embedder: Embedder | None = None,
                 embeddings: np.ndarray | None = None, k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.embedder = embedder
        self._build_lexical()
        self.embeddings = embeddings
        if embedder is not None and embeddings is None and chunks:
            self.embeddings = embedder.encode([c.search_text for c in chunks])

    # ---------------------------------------------------------------- indexing
    def _build_lexical(self) -> None:
        self.doc_tokens = [analyze(c.search_text) for c in self.chunks]
        n = len(self.chunks)
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=np.float64)
        self.avg_len = float(self.doc_len.mean()) if n else 0.0
        df: Counter[str] = Counter()
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, toks in enumerate(self.doc_tokens):
            for term, tf in Counter(toks).items():
                df[term] += 1
                self.postings[term].append((i, tf))
        self.df = df
        self.bm25_idf = {t: math.log(1 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.tfidf_idf = {t: math.log((1 + n) / (1 + d)) + 1 for t, d in df.items()}
        self.doc_vecs = [self._tfidf_vector(Counter(t)) for t in self.doc_tokens]

    def _tfidf_vector(self, counts: Counter[str]) -> dict[str, float]:
        vec = {t: (1 + math.log(tf)) * self.tfidf_idf[t]
               for t, tf in counts.items() if t in self.tfidf_idf}
        norm = math.sqrt(sum(v * v for v in vec.values()))
        return {t: v / norm for t, v in vec.items()} if norm else {}

    # ---------------------------------------------------------------- scoring
    def _bm25(self, q_terms: list[str]) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        for term in set(q_terms):
            idf = self.bm25_idf.get(term)
            if idf is None:
                continue
            for i, tf in self.postings[term]:
                denom = tf + self.k1 * (1 - self.b + self.b * self.doc_len[i] / self.avg_len)
                scores[i] += idf * tf * (self.k1 + 1) / denom
        return scores

    def _tfidf(self, q_vec: dict[str, float]) -> np.ndarray:
        scores = np.zeros(len(self.chunks))
        for term, w in q_vec.items():
            for i, _ in self.postings[term]:
                scores[i] += w * self.doc_vecs[i].get(term, 0.0)
        return scores

    @staticmethod
    def _sparse_cos(a: dict[str, float], b: dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(t, 0.0) for t, v in a.items())

    # ---------------------------------------------------------------- search
    def search(self, query: str, *, top_k: int = 6, candidates: int = 30,
               mmr_lambda: float = 0.7, min_term_coverage: float = 0.5,
               min_dense_similarity: float = 0.45) -> RetrievalResult:
        q_terms = analyze(query, query=True) or analyze(query)
        if not self.chunks or not q_terms:
            return RetrievalResult(query, [], 0.0, 0.0, False, _surface_words(query, q_terms))

        q_vec = self._tfidf_vector(Counter(q_terms))
        rankings = [self._bm25(q_terms), self._tfidf(q_vec)]
        dense_scores = None
        if self.embedder is not None and self.embeddings is not None:
            q_emb = self.embedder.encode([query])[0]
            dense_scores = self.embeddings @ q_emb
            rankings.append(dense_scores)
        tfidf_scores = rankings[1]

        fused = self._rrf(rankings, candidates)
        if not fused:
            return RetrievalResult(query, [], 0.0, 0.0, False, _surface_words(query, q_terms))

        selected = self._mmr(fused, top_k, mmr_lambda, dense_scores is not None)
        top = max(fused.values())
        hits = [Hit(self.chunks[i], fused[i] / top, float(tfidf_scores[i])) for i in selected]

        # Relevance gate: how much of the question's specific vocabulary appears
        # in the best few chunks we pass on. Terms are weighted by IDF, and a
        # term the notes never mention gets the highest weight, so a single
        # generic overlap ("control" in "TCP congestion control") can't pass.
        covered = set().union(*(set(self.doc_tokens[h_i]) for h_i in selected[:3]))
        unique_q = sorted(set(q_terms))
        unseen_weight = max(self.tfidf_idf.values())
        weights = {t: self.tfidf_idf.get(t, unseen_weight) for t in unique_q}
        missing_stems = [t for t in unique_q if t not in covered]
        coverage = 1 - sum(weights[t] for t in missing_stems) / sum(weights.values())
        missing = _surface_words(query, missing_stems)
        max_sim = max((h.similarity for h in hits), default=0.0)
        dense_ok = False
        if dense_scores is not None:
            # Dense embeddings catch paraphrases with no shared keywords, but
            # their cosine scale differs, so only a clearly high score counts.
            dense_ok = float(max(dense_scores[i] for i in selected)) >= min_dense_similarity
        sufficient = bool(hits) and (coverage >= min_term_coverage or dense_ok)
        return RetrievalResult(query, hits, max_sim, coverage, sufficient, missing)

    @staticmethod
    def _rrf(rankings: list[np.ndarray], candidates: int, k: int = 60) -> dict[int, float]:
        fused: dict[int, float] = defaultdict(float)
        for scores in rankings:
            order = np.argsort(-scores)[:candidates]
            rank = 0
            for i in order:
                if scores[i] <= 0:
                    break
                rank += 1
                fused[int(i)] += 1.0 / (k + rank)
        return dict(fused)

    def _mmr(self, fused: dict[int, float], top_k: int, lam: float, dense: bool) -> list[int]:
        top = max(fused.values())
        relevance = {i: s / top for i, s in fused.items()}
        remaining = sorted(relevance, key=relevance.get, reverse=True)
        selected: list[int] = []
        while remaining and len(selected) < top_k:
            def mmr_score(i: int) -> float:
                if not selected:
                    return relevance[i]
                redundancy = max(self._similarity(i, j, dense) for j in selected)
                return lam * relevance[i] - (1 - lam) * redundancy
            best = max(remaining, key=mmr_score)
            selected.append(best)
            remaining.remove(best)
        return selected

    def _similarity(self, i: int, j: int, dense: bool) -> float:
        if dense and self.embeddings is not None:
            return float(self.embeddings[i] @ self.embeddings[j])
        return self._sparse_cos(self.doc_vecs[i], self.doc_vecs[j])
