# College Notes Question-Answering Assistant

A retrieval-augmented generation (RAG) assistant for asking natural-language
questions about your own college lecture notes and course material. Answers
come only from the notes you upload. Every claim is cited to the passage, page,
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
- **Grounded generation with Claude.** Retrieved passages go to Claude through the [Citations API](https://platform.claude.com/docs/en/build-with-claude/citations), so each citation points to an actual passage rather than a reference the model wrote itself.
- **Three safeguards against unsupported answers** (see below).
- **Follow-up questions in chat.** Short follow-ups like "and LRU?" use the topic of the previous question.
- **CLI and Streamlit web UI**, plus a retrieval evaluation script that needs no API key.

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
   generator.py            each chunk → its own `document` block, citations on
        │                  Claude answers from the excerpts only, or NOT_IN_NOTES
        ▼
   Answer                  text with [n] markers, cited passages, support ratio,
                           warnings for uncited or partly cited answers
```

### How unsupported answers are reduced

1. **A relevance gate runs before generation.** The gate measures how much of
   the question's vocabulary appears in the top retrieved chunks. Each term is
   weighted by IDF, and a term that never appears in the notes gets the highest
   weight. "What is TCP congestion control?" shares only the generic word
   "control" with OS notes, so it is declined without calling the model. The
   user is told which terms were missing.
2. **Generation is limited to the excerpts.** The system prompt tells Claude
   not to add facts from general knowledge. Claude must answer only the parts
   the notes cover, point out where sources disagree, and reply `NOT_IN_NOTES`
   when the excerpts don't contain the answer. The citations come from the
   Citations API, so they always point at real passages.
3. **Answers are checked after generation.** The assistant measures what share
   of the answer is backed by citations. An answer with no citations is flagged
   as unverified. One where less than half is cited gets a warning.

## Setup

Requires Python 3.10+ and a Claude API key.

```bash
pip install -r requirements.txt          # or: pip install -e ".[ui,dev]"
export ANTHROPIC_API_KEY=sk-ant-...
```

## Usage

### CLI

```bash
python -m notes_qa.cli ingest sample_notes/ ~/semester3/os/*.pdf
python -m notes_qa.cli list
python -m notes_qa.cli ask "What is Belady's anomaly?" --quotes
python -m notes_qa.cli chat                       # interactive, supports follow-ups
python -m notes_qa.cli search "safe state"        # inspect retrieval only, no API call
python -m notes_qa.cli remove os_lecture_notes.md
```

If you ran `pip install -e .`, `notes-qa` works in place of `python -m notes_qa.cli`.
Re-ingesting a file with the same name replaces the earlier version.

### Web UI

```bash
streamlit run app.py
```

Upload notes in the sidebar, click **Index uploaded files**, then ask questions
in the chat. Expand **Sources** under an answer to see the exact cited passages.

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

`answer.status` is one of `answered`, `not_in_notes` (Claude found no answer in
the excerpts), `no_relevant_notes` (declined by the relevance gate), `refused`,
or `error`.

## Configuration

Environment variables (see `notes_qa/config.py`):

| Variable | Default | Meaning |
|---|---|---|
| `NOTES_QA_INDEX_DIR` | `.notes_index` | where the index is stored |
| `NOTES_QA_MODEL` | `claude-opus-5-5` | Claude model used for answers |
| `NOTES_QA_EFFORT` | `medium` | reasoning effort (`low` … `max`); empty to omit |
| `NOTES_QA_FALLBACKS` | `1` | server-side refusal fallback (Claude API only; set `0` on Bedrock/Vertex/Foundry) |
| `NOTES_QA_TOP_K` | `6` | passages sent to Claude |
| `NOTES_QA_CHUNK_CHARS` | `1200` | target chunk size |
| `NOTES_QA_CHUNK_OVERLAP` | `2` | sentences repeated between consecutive chunks |
| `NOTES_QA_MIN_COVERAGE` | `0.5` | relevance gate threshold (lower = answer more, decline less) |
| `NOTES_QA_EMBEDDER` | `tfidf` | set to `sentence-transformers` for dense hybrid retrieval |
| `NOTES_QA_DENSE_MODEL` | `all-MiniLM-L6-v2` | embedding model when dense retrieval is enabled |
| `NOTES_QA_MIN_DENSE_SIM` | `0.45` | dense cosine that also passes the gate |

Dense embeddings (`pip install sentence-transformers`) help with paraphrased
questions that share few words with the notes, such as "How does the OS decide
which process runs next?" for notes that say "CPU scheduling". After switching
embedders, re-run `ingest` so the embeddings are computed.

## Evaluation

```bash
python eval/eval_retrieval.py
```

The script runs a labelled question set against the sample notes. It needs no
API key. Current results:

```
answerable (17):  hit@1 100%  hit@6 100%  wrongly declined 0%
unanswerable (4): declined before generation 75%
```

The unanswerable question that gets through, "Describe the structure of a
neuron", matches "data structures" in the notes. On word overlap alone it looks
the same as a genuine question, so the gate passes it on and Claude's
`NOT_IN_NOTES` instruction handles it. Add your own questions to
`eval/retrieval_questions.json` to tune `NOTES_QA_MIN_COVERAGE` for your course.

## Tests

```bash
python -m pytest
```

The tests cover chunking, loaders, ranking, the relevance gate, MMR,
de-duplication, citation parsing, abstention, refusals, follow-up history and
index persistence. They use a fake Claude client, so no network access is
needed.

## Limitations

- Scanned PDFs need OCR first. Only text-based PDFs can be read.
- Equations and diagrams come through as whatever text the PDF provides.
- The default lexical retriever does not match synonyms. Enable dense
  embeddings if your questions often use different words from your notes.
