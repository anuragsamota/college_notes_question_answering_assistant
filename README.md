# College Notes Question-Answering Assistant

A retrieval-augmented generation (RAG) assistant for asking natural-language
questions about your own college lecture notes and course material. It runs
entirely on your machine, with a local LLM served by [Ollama](https://ollama.com).
Answers come only from the notes you upload. Every claim is cited to the passage, page,
and section it came from. If the notes don't cover a question, the assistant
says so instead of answering from general knowledge.

```
$ notes-qa ask "How does the Banker's algorithm decide whether to grant a request?"

The system grants a request only if the resulting state is safe, meaning there is a
safe sequence in which every process can obtain its maximum need and finish. [1]

Sources:
  [1] os_lecture_notes.md · CS301 Operating Systems — Lecture Notes > Unit 3: Deadlocks > Banker's Algorithm
```

## Features

- **Upload PDFs, Word documents (.docx), Markdown and text files.** The assistant keeps page numbers and section headings.
- **Hybrid retrieval.** BM25 and TF-IDF scoring, with optional dense embeddings, combined using Reciprocal Rank Fusion. MMR then picks a varied set of passages.
- **Local, grounded generation with Ollama.** The model must cite a numbered passage for each statement, and the app checks every citation against the passage it points to.
- **Three safeguards against unsupported answers** (see below).
- **Follow-up questions in chat.** Short follow-ups like "and LRU?" use the topic of the previous question.
- **CLI and Streamlit web UI**, plus a retrieval evaluation script that runs without an LLM.

## Architecture

```
 notes (pdf/docx/md/txt)
        │  loaders.py      text per page; DOCX headings → Markdown headings
        ▼
   chunking.py             split by headings → paragraphs → sentences;
        │                  ~1200-char chunks, 2-sentence overlap,
        │                  each chunk tagged with file · page · section path
        ▼
   store.py                .notes_index/chunks.json (+ embeddings.npy)
        │
 question ──► retrieval.py
        │       BM25 ─┐
        │      TF-IDF ├─► Reciprocal Rank Fusion ─► MMR (top 6, de-duplicated)
        │  [dense emb]┘                                   │
        │                                   relevance gate (IDF-weighted
        │                                   coverage of question terms)
        │                                     │ fails → "not in your notes"
        ▼                                     ▼ passes
   generator.py            numbered excerpts → Ollama chat model; it answers
        │                  from them with [n] citations, or NOT_IN_NOTES
        ▼
   citation check          drop [n] that point nowhere; each cited sentence must
        │                  share most content words with the excerpt it cites
        ▼
   Answer                  text with [n] markers, cited passages, verified share,
                           sentences that failed the check
```

### How unsupported answers are reduced

1. **A relevance gate runs before generation.** The gate measures how much of
   the question's vocabulary appears in the top retrieved chunks. Each term is
   weighted by IDF, and a term that never appears in the notes gets the highest
   weight. "What is TCP congestion control?" shares only the generic word
   "control" with OS notes, so it is declined without calling the model. The
   user is told which terms were missing.
2. **Generation is limited to the excerpts.** The system prompt tells the
   model not to add facts from general knowledge, to end each factual sentence
   with the `[n]` of the excerpt it came from, to answer only the parts the
   notes cover, to point out where sources disagree, and to reply
   `NOT_IN_NOTES` when the excerpts don't contain the answer. Temperature is
   kept low (0.1).
3. **Citations are checked after generation.** Small local models sometimes
   cite carelessly, so the citations aren't taken on trust:
   - A citation to an excerpt number that doesn't exist is removed and reported.
   - A cited sentence counts as verified only when at least 60% of its content
     words appear in the excerpt it cites. This catches a real citation
     attached to an invented claim, such as "It was discovered at IBM in 1969
     [2]".
   - Factual sentences with no citation are counted as unsupported.

   The answer reports the verified share and lists the sentences that failed,
   so the student knows exactly what to double-check.

## Setup

Requires Python 3.10+ and [Ollama](https://ollama.com/download).

```bash
ollama pull llama3.1:8b                  # the default model; any chat model works
ollama serve                             # if Ollama isn't already running
pip install -r requirements.txt          # or: pip install -e ".[ui,dev]"
```

Choosing a model: models with 7–8B parameters or more (`llama3.1:8b`,
`qwen2.5:7b`, `mistral-nemo`) follow the citation instructions well. Smaller
ones (`llama3.2:3b`, `qwen2.5:3b`) run on modest laptops but produce more
answers that fail the citation check. Reasoning models such as `qwen3` or
`deepseek-r1` also work; their `<think>` output is removed from the answer.

## Usage

### CLI

```bash
python -m notes_qa.cli ingest sample_notes/ ~/semester3/os/*.pdf
python -m notes_qa.cli list
python -m notes_qa.cli ask "What is Belady's anomaly?" --quotes
python -m notes_qa.cli chat                       # interactive, supports follow-ups
python -m notes_qa.cli search "safe state"        # inspect retrieval only, no LLM call
python -m notes_qa.cli remove os_lecture_notes.md
```

If you ran `pip install -e .`, `notes-qa` works in place of `python -m notes_qa.cli`.
Re-ingesting a file with the same name replaces the earlier version.

### Web UI

```bash
streamlit run app.py
```

Upload notes in the sidebar, click **Index uploaded files**, then ask questions
in the chat. Expand **Sources** under an answer to see the passages it cites,
and **Statements not matched to your notes** to see what failed the citation check.

### Python

```python
from notes_qa import Assistant

assistant = Assistant()
assistant.ingest_paths(["lectures/"])
answer = assistant.ask("Why does FCFS suffer from the convoy effect?")
print(answer.status, answer.text)
for n, hit in answer.cited_sources:
    print(n, hit.chunk.location)
```

`answer.status` is one of `answered`, `not_in_notes` (the model found no
answer in the excerpts), `no_relevant_notes` (declined by the relevance gate),
or `error` (for example, Ollama isn't running or the model isn't pulled).
`answer.support_ratio` is the verified share of the answer, and
`answer.unsupported` lists the sentences that failed the citation check.

## Configuration

Environment variables (see `notes_qa/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `NOTES_QA_INDEX_DIR` | `.notes_index` | where the index is stored |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server address |
| `NOTES_QA_MODEL` | `llama3.1:8b` | Ollama chat model used for answers |
| `NOTES_QA_TEMPERATURE` | `0.1` | sampling temperature |
| `NOTES_QA_NUM_CTX` | `8192` | context window requested from Ollama (its default is too small for 6 excerpts) |
| `NOTES_QA_MAX_TOKENS` | `1024` | maximum answer length in tokens |
| `NOTES_QA_MIN_CITATION_OVERLAP` | `0.6` | share of a cited sentence's content words that must appear in the cited excerpt |
| `NOTES_QA_TOP_K` | `6` | passages sent to the model |
| `NOTES_QA_CHUNK_CHARS` | `1200` | target chunk size |
| `NOTES_QA_CHUNK_OVERLAP` | `2` | sentences repeated between consecutive chunks |
| `NOTES_QA_MIN_COVERAGE` | `0.5` | relevance gate threshold (lower = answer more, decline less) |
| `NOTES_QA_EMBEDDER` | `tfidf` | `ollama` or `sentence-transformers` adds dense embeddings to retrieval |
| `NOTES_QA_EMBED_MODEL` | `nomic-embed-text` / `all-MiniLM-L6-v2` | embedding model for the chosen embedder |
| `NOTES_QA_MIN_DENSE_SIM` | `0.45` | dense cosine that also passes the gate |

Dense embeddings help with paraphrased questions that share few words with
the notes, such as "How does the OS decide which process runs next?" for notes
that say "CPU scheduling". The simplest way to enable them is through Ollama:

```bash
ollama pull nomic-embed-text
export NOTES_QA_EMBEDDER=ollama
python -m notes_qa.cli ingest sample_notes/     # re-ingest to compute embeddings
```

Embeddings are stored with the index. They are recomputed when you switch the
embedder or the embedding model and re-ingest. The minimum dense similarity
that passes the relevance gate (`NOTES_QA_MIN_DENSE_SIM`) depends on the
embedding model, so check it with the eval script.

## Evaluation

```bash
python eval/eval_retrieval.py
```

The script runs a labelled question set against the sample notes. It needs no
LLM. Current results with the default lexical retriever:

```
answerable (17):  hit@1 100%  hit@6 100%  wrongly declined 0%
unanswerable (4): declined before generation 75%
```

The unanswerable question that gets through, "Describe the structure of a
neuron", matches "data structures" in the notes. On word overlap alone it looks
the same as a genuine question, so the gate passes it on and the model's
`NOT_IN_NOTES` instruction handles it. Add your own questions to
`eval/retrieval_questions.json` to tune `NOTES_QA_MIN_COVERAGE` for your course.

## Tests

```bash
python -m pytest
```

The tests cover chunking, loaders, ranking, the relevance gate, MMR,
de-duplication, citation checking (invalid numbers, invented claims with real
citations), abstention, `<think>` stripping, Ollama errors, follow-up history
and index persistence. `tests/test_ollama_http.py` runs the real `ollama`
client against a local stub of Ollama's HTTP API, so no Ollama install or
model is needed.

## Limitations

- Scanned PDFs need OCR first. Only text-based PDFs can be read.
- Equations and diagrams come through as whatever text the PDF provides.
- The default lexical retriever does not match synonyms. Enable dense
  embeddings if your questions often use different words from your notes.
- The citation check compares words, not meaning. A sentence that paraphrases
  its source heavily can be flagged even though it is correct. A sentence that
  reuses the excerpt's words but changes what they say (for example, swapping
  "eliminates" for "causes") can pass.
