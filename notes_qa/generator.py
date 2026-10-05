"""Grounded answer generation with a local LLM served by Ollama.

Retrieved chunks are given to the model as numbered excerpts, and it must cite
them inline as ``[n]``. Local models are less dependable at following such
instructions than hosted ones, so the citations are checked in code rather
than trusted. Three layers keep answers tied to the notes:

1. The system prompt restricts the model to the excerpts and gives it an
   explicit way to say the notes don't cover the question (``NOT_IN_NOTES``).
2. Every ``[n]`` is validated: numbers that point at no excerpt are removed,
   and each cited sentence must share most of its content words with the
   excerpt it cites, which catches citations attached to invented claims.
3. The share of the answer that is cited *and* verified is reported, and
   sentences that fail are listed so the student knows what to double-check.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import ollama

from notes_qa.retrieval import Hit, analyze
from notes_qa.servers import NoServerAvailable, OllamaRouter

ABSTAIN_TOKEN = "NOT_IN_NOTES"

SYSTEM_PROMPT = f"""You are a study assistant that answers students' questions \
using only their own college lecture notes and course material.

The user's message contains numbered excerpts from the notes, retrieved for \
this question. Follow these rules:

- Base every statement on the excerpts. Do not add facts, formulas, examples or \
definitions from general knowledge, even if you are confident they are correct; \
the student needs answers that match what their course teaches.
- End every sentence that states a fact with the number of the excerpt it comes \
from in square brackets, for example: "Paging removes external fragmentation [2]." \
Use [1][3] when a sentence draws on two excerpts. Only cite excerpt numbers that exist.
- Stay close to the wording of the excerpts; keep their terminology and notation.
- If the excerpts answer only part of the question, answer that part and say \
plainly which part the notes do not cover.
- If the excerpts do not contain the answer at all, reply with exactly \
{ABSTAIN_TOKEN} on the first line, followed by one sentence naming what is \
missing. Do not guess.
- If excerpts disagree with each other, point out the disagreement and cite both.
- Lead with the direct answer, then the supporting detail. Be concise.
"""

_MARKER = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z*(\"'])")
_CHUNK_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class Citation:
    number: int            # 1-based index of the excerpt, as shown to the user
    hit: Hit
    cited_text: str        # passage of the excerpt that best matches the claim


@dataclass
class Answer:
    question: str
    text: str
    status: str            # answered | not_in_notes | no_relevant_notes | error
    citations: list[Citation] = field(default_factory=list)
    sources: list[Hit] = field(default_factory=list)
    support_ratio: float = 0.0   # share of answer text that is cited and verified
    unsupported: list[str] = field(default_factory=list)  # sentences that failed checks
    warnings: list[str] = field(default_factory=list)
    model: str | None = None
    server: str | None = None    # name of the Ollama server that answered

    @property
    def grounded(self) -> bool:
        return self.status == "answered" and bool(self.citations)

    @property
    def cited_sources(self) -> list[tuple[int, Hit]]:
        seen: dict[int, Hit] = {}
        for c in self.citations:
            seen.setdefault(c.number, c.hit)
        return sorted(seen.items())


def build_prompt(question: str, hits: Sequence[Hit]) -> str:
    parts = ["Excerpts from the student's notes:\n"]
    for n, hit in enumerate(hits, start=1):
        parts.append(f"[{n}] ({hit.chunk.location})\n{hit.chunk.text}\n")
    parts.append(f"Question: {question}")
    return "\n".join(parts)


class AnswerGenerator:
    def __init__(self, router: OllamaRouter, *, temperature: float = 0.1,
                 num_ctx: int = 8192, max_tokens: int = 1024, min_overlap: float = 0.6):
        self.router = router
        self.options = {"temperature": temperature, "num_ctx": num_ctx,
                        "num_predict": max_tokens}
        self.min_overlap = min_overlap

    def generate(self, question: str, hits: Sequence[Hit],
                 history: Sequence[tuple[str, str]] = ()) -> Answer:
        messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        for prev_q, prev_a in history:
            messages.append({"role": "user", "content": prev_q})
            # Citation markers refer to an earlier set of excerpts; drop them.
            messages.append({"role": "assistant", "content": _strip_markers(prev_a)})
        messages.append({"role": "user", "content": build_prompt(question, hits)})

        def error(msg: str) -> Answer:
            return Answer(question, msg, "error", sources=list(hits))

        try:
            response, server, model = self.router.chat(messages, self.options)
        except NoServerAvailable as exc:
            return error(_unavailable_message(self.router, exc))
        except ollama.ResponseError as exc:
            server = self.router.last_attempt
            if exc.status_code == 404:
                model = self.router.chat_model(server)
                return error(f"Model '{model}' is not installed on Ollama server "
                             f"'{server.name}'. Run `ollama pull {model}` on that machine, "
                             "or pick another model for it.")
            return error(f"Ollama error from '{server.name}' ({exc.status_code}): {exc.error}")

        answer = parse_response(question, hits, response.message.content or "",
                                done_reason=response.done_reason, model=model,
                                min_overlap=self.min_overlap)
        answer.server = server.name
        return answer


def _unavailable_message(router: OllamaRouter, exc: NoServerAvailable) -> str:
    servers = router.registry.candidates(router.selection)
    if len(servers) == 1 and len(exc.errors) == 1:
        s = servers[0]
        hint = ("Start it with `ollama serve`." if s.is_local else
                "Check the machine is on and Ollama listens on the network "
                "(OLLAMA_HOST=0.0.0.0 on that machine, port 11434 open).")
        return f"Could not reach Ollama server '{s.name}' at {s.host}. {hint}"
    return "No Ollama server could answer: " + "; ".join(exc.errors) + "."


def _split_sentences(text: str) -> list[str]:
    """Split an answer into sentences, keeping trailing [n] markers attached."""
    out: list[str] = []
    for line in text.splitlines(keepends=True):
        pieces = _SENTENCE_END.split(line)
        for i, piece in enumerate(pieces):
            # A marker placed after the full stop ("... wait. [2] Next") belongs
            # to the sentence before it.
            m = re.match(r"\s*((?:\[[\d,;\s]+\]\s*)+)", piece)
            if m and out:
                out[-1] = out[-1].rstrip() + " " + m.group(1).strip()
                piece = piece[m.end():]
            if piece:
                out.append(piece if i == len(pieces) - 1 else piece + " ")
    return out


def _tidy(text: str) -> str:
    """Remove the space left before punctuation when a marker is dropped."""
    return re.sub(r" +([.!?,;:])", r"\1", text)


def _strip_markers(text: str) -> str:
    return _tidy(_MARKER.sub("", text))


def _valid_marks(count: int):
    def repl(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"[,;]", m.group(1))]
        return "".join(f"[{n}]" for n in dict.fromkeys(nums) if 1 <= n <= count)
    return repl


def _best_passage(claim_terms: set[str], text: str) -> str:
    best, best_score = "", -1.0
    for sent in _CHUNK_SENTENCE.split(text):
        sent = sent.strip()
        if not sent:
            continue
        score = len(claim_terms & set(analyze(sent)))
        if score > best_score:
            best, best_score = sent, score
    return best


def parse_response(question: str, hits: Sequence[Hit], content: str, *,
                   done_reason: str | None = None, model: str | None = None,
                   min_overlap: float = 0.6) -> Answer:
    body = _THINK.sub("", content).strip()

    if body.startswith(ABSTAIN_TOKEN) or (ABSTAIN_TOKEN in body and len(body) < 300):
        reason = body.replace(ABSTAIN_TOKEN, "").strip(" :\n")
        msg = "I couldn't find the answer to this in your notes."
        if reason:
            msg += f" {reason}"
        return Answer(question, msg, "not_in_notes", sources=list(hits), model=model)

    chunk_terms = [set(analyze(h.chunk.search_text)) for h in hits]
    citations: list[Citation] = []
    unsupported: list[str] = []
    invalid: set[int] = set()
    rendered: list[str] = []
    total_chars = supported_chars = 0

    for sentence in _split_sentences(body):
        numbers: list[int] = []
        for group in _MARKER.findall(sentence):
            for num in re.split(r"[,;]", group):
                n = int(num)
                if 1 <= n <= len(hits):
                    if n not in numbers:
                        numbers.append(n)
                else:
                    invalid.add(n)
        claim = _strip_markers(sentence)
        terms = set(analyze(claim))
        # Keep only valid excerpt numbers in the displayed markers.
        rendered.append(_tidy(_MARKER.sub(_valid_marks(len(hits)), sentence)))

        if len(terms) < 3:
            continue  # headings, connectors and other non-claims
        size = len(claim.strip())
        total_chars += size
        if not numbers:
            unsupported.append(claim.strip())
            continue
        covered = set().union(*(chunk_terms[n - 1] for n in numbers))
        overlap = len(terms & covered) / len(terms)
        if overlap >= min_overlap:
            supported_chars += size
        else:
            unsupported.append(claim.strip())
        for n in numbers:
            citations.append(Citation(n, hits[n - 1],
                                      _best_passage(terms, hits[n - 1].chunk.text)))

    text = "".join(rendered).strip()
    support = supported_chars / total_chars if total_chars else 0.0
    answer = Answer(question, text, "answered", citations, list(hits), support,
                    unsupported, model=model)
    if done_reason == "length":
        answer.warnings.append("The answer was cut off by the output token limit.")
    if invalid:
        answer.warnings.append("Removed citations to excerpts that don't exist: "
                               + ", ".join(f"[{n}]" for n in sorted(invalid)) + ".")
    if not citations:
        answer.warnings.append(
            "This answer has no citations to your notes; treat it as unverified.")
    elif unsupported:
        answer.warnings.append(
            f"{len(unsupported)} statement(s) could not be matched to your notes "
            f"({support:.0%} of the answer is verified); double-check them.")
    return answer
