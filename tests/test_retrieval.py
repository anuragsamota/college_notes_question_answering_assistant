import numpy as np
import pytest

from notes_qa.chunking import Chunker
from notes_qa.loaders import load_file
from notes_qa.retrieval import HybridRetriever, analyze, stem


@pytest.fixture
def retriever(sample_path):
    return HybridRetriever(Chunker(600, 1).chunk(load_file(sample_path)))


@pytest.mark.parametrize("a,b", [("page", "paging"), ("processes", "process"),
                                 ("schedule", "scheduling"), ("memories", "memory")])
def test_stemmer_conflates_inflections(a, b):
    assert stem(a) == stem(b)


def test_query_filler_removed():
    assert analyze("Explain the definition of a deadlock", query=True) == ["deadlock"]


@pytest.mark.parametrize("question,expected_section", [
    ("What are the four necessary conditions for deadlock?", "Necessary Conditions"),
    ("How is the effective access time computed with a TLB?", "Translation Lookaside Buffer"),
    ("Which page replacement algorithm suffers from Belady's anomaly?", "Page Replacement"),
    ("What problem does aging solve?", "Priority Scheduling"),
    ("What happens when the time quantum expires?", "Round Robin (RR)"),
    ("What does the PCB contain?", "Process Control Block"),
])
def test_top_hit_is_relevant_section(retriever, question, expected_section):
    result = retriever.search(question, top_k=4)
    assert result.sufficient
    assert result.hits[0].chunk.section.endswith(expected_section)


@pytest.mark.parametrize("question", [
    "Who won the FIFA World Cup in 2018?",
    "What is the capital of Australia?",
    "Explain photosynthesis in plants",
    # Shares one generic word ("control") with the notes; must still be gated.
    "What is TCP congestion control?",
])
def test_off_topic_questions_are_gated(retriever, question):
    result = retriever.search(question)
    assert not result.sufficient


def test_missing_terms_reported_as_written(retriever):
    result = retriever.search("photosynthesis in chloroplasts")
    assert "photosynthesis" in result.missing_terms


def test_mmr_avoids_duplicates(sample_path):
    chunks = Chunker(600, 1).chunk(load_file(sample_path))
    dup = Chunker(600, 1).chunk(load_file(sample_path))
    for c in dup:
        c.id += "-copy"
    retriever = HybridRetriever(chunks + dup)
    result = retriever.search("Banker's algorithm safe state", top_k=3, mmr_lambda=0.5)
    texts = [h.chunk.text for h in result.hits]
    assert len(set(texts)) == len(texts)


class FakeEmbedder:
    """Deterministic bag-of-letters embedder, to exercise the dense code path."""

    def encode(self, texts):
        vecs = np.zeros((len(texts), 26), dtype=np.float32)
        for i, t in enumerate(texts):
            for ch in t.lower():
                if "a" <= ch <= "z":
                    vecs[i, ord(ch) - 97] += 1
        return vecs / np.maximum(np.linalg.norm(vecs, axis=1, keepdims=True), 1e-9)


def test_dense_scores_are_fused(sample_path):
    chunks = Chunker(600, 1).chunk(load_file(sample_path))
    retriever = HybridRetriever(chunks, embedder=FakeEmbedder())
    assert retriever.embeddings.shape == (len(chunks), 26)
    result = retriever.search("Banker's algorithm safe sequence")
    assert result.hits[0].chunk.section.endswith("Banker's Algorithm")


def test_empty_index():
    result = HybridRetriever([]).search("anything")
    assert result.hits == [] and not result.sufficient
