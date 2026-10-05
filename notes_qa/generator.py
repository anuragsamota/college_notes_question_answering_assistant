"""Grounded answer generation with Claude.

Retrieved chunks are sent as separate ``document`` blocks with the Citations
API enabled, so every claim in the answer comes back linked to the exact
passage it was drawn from. Three layers keep answers tied to the notes:

1. The system prompt restricts Claude to the supplied excerpts and gives it an
   explicit way to say the notes don't cover the question (``NOT_IN_NOTES``).
2. Citations are produced by the API itself, pointing at specific chunks,
   rather than being free-text references the model could invent.
3. After generation, the share of the answer backed by citations is measured;
   an answer with no citations is flagged as unsupported.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

import anthropic

from notes_qa.retrieval import Hit

ABSTAIN_TOKEN = "NOT_IN_NOTES"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

SYSTEM_PROMPT = f"""You are a study assistant that answers students' questions \
using only their own college lecture notes and course material.

The user's message contains numbered excerpts from the notes, retrieved for \
this question. Follow these rules:

- Base every statement on the excerpts. Do not add facts, formulas, examples or \
definitions from general knowledge, even if you are confident they are correct; \
the student needs answers that match what their course teaches.
- If the excerpts answer only part of the question, answer that part and say \
plainly which part the notes do not cover.
- If the excerpts do not contain the answer at all, reply with exactly \
{ABSTAIN_TOKEN} on the first line, followed by one sentence naming what is \
missing. Do not guess.
- If excerpts disagree with each other, point out the disagreement and cite both.
- Explain clearly, as a good teaching assistant would: lead with the direct \
answer, then the supporting detail. Keep the notes' terminology and notation.
"""


@dataclass
class Citation:
    number: int            # 1-based index of the excerpt, as shown to the user
    hit: Hit
    cited_text: str


@dataclass
class Answer:
    question: str
    text: str
    status: str            # answered | not_in_notes | no_relevant_notes | refused | error
    citations: list[Citation] = field(default_factory=list)
    sources: list[Hit] = field(default_factory=list)
    support_ratio: float = 0.0   # share of answer text backed by a citation
    warnings: list[str] = field(default_factory=list)
    model: str | None = None

    @property
    def grounded(self) -> bool:
        return self.status == "answered" and bool(self.citations)

    @property
    def cited_sources(self) -> list[tuple[int, Hit]]:
        seen: dict[int, Hit] = {}
        for c in self.citations:
            seen.setdefault(c.number, c.hit)
        return sorted(seen.items())


def build_user_content(question: str, hits: Sequence[Hit]) -> list[dict[str, Any]]:
    content: list[dict[str, Any]] = []
    for n, hit in enumerate(hits, start=1):
        content.append({
            "type": "document",
            "source": {"type": "content",
                       "content": [{"type": "text", "text": hit.chunk.text}]},
            "title": f"[{n}] {hit.chunk.location}",
            "citations": {"enabled": True},
        })
    content.append({"type": "text", "text": f"Question: {question}"})
    return content


class AnswerGenerator:
    def __init__(self, client: anthropic.Anthropic | None = None, *,
                 model: str = "claude-opus-5-5", effort: str | None = "medium",
                 max_tokens: int = 16000, use_fallbacks: bool = True):
        self._client = client
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens
        self.use_fallbacks = use_fallbacks

    @property
    def client(self) -> anthropic.Anthropic:
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def generate(self, question: str, hits: Sequence[Hit],
                 history: Sequence[tuple[str, str]] = ()) -> Answer:
        messages: list[dict[str, Any]] = []
        for prev_q, prev_a in history:
            messages.append({"role": "user", "content": prev_q})
            # Citation markers refer to an earlier set of excerpts; drop them.
            messages.append({"role": "assistant", "content": re.sub(r" ?\[\d+\]", "", prev_a)})
        messages.append({"role": "user", "content": build_user_content(question, hits)})

        request: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "messages": messages,
        }
        if self.effort:
            request["output_config"] = {"effort": self.effort}

        try:
            if self.use_fallbacks:
                response = self.client.beta.messages.create(
                    **request, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                response = self.client.messages.create(**request)
        except anthropic.AuthenticationError:
            return Answer(question, "Claude API authentication failed. Set ANTHROPIC_API_KEY "
                          "(or run `ant auth login`).", "error", sources=list(hits))
        except anthropic.RateLimitError:
            return Answer(question, "Rate limited by the Claude API; try again shortly.",
                          "error", sources=list(hits))
        except anthropic.APIStatusError as exc:
            return Answer(question, f"Claude API error ({exc.status_code}): {exc.message}",
                          "error", sources=list(hits))
        except anthropic.APIConnectionError:
            return Answer(question, "Could not reach the Claude API (network error).",
                          "error", sources=list(hits))

        return parse_response(question, hits, response)


def parse_response(question: str, hits: Sequence[Hit], response: Any) -> Answer:
    model = getattr(response, "model", None)
    if response.stop_reason == "refusal":
        return Answer(question, "The model declined to answer this question.", "refused",
                      sources=list(hits), model=model)

    parts: list[str] = []
    citations: list[Citation] = []
    total_chars = cited_chars = 0
    for block in response.content:
        if block.type != "text":
            continue  # thinking / fallback markers are not part of the answer
        text = block.text
        numbers: list[int] = []
        for cit in block.citations or []:
            idx = getattr(cit, "document_index", None)
            if idx is None or not 0 <= idx < len(hits):
                continue
            number = idx + 1
            citations.append(Citation(number, hits[idx], getattr(cit, "cited_text", "")))
            if number not in numbers:
                numbers.append(number)
        stripped = text.strip()
        total_chars += len(stripped)
        if numbers:
            cited_chars += len(stripped)
            text = text.rstrip() + " " + "".join(f"[{n}]" for n in numbers) + \
                text[len(text.rstrip()):]
        parts.append(text)

    body = "".join(parts).strip()
    support = cited_chars / total_chars if total_chars else 0.0

    if body.startswith(ABSTAIN_TOKEN):
        reason = body[len(ABSTAIN_TOKEN):].strip(" :\n")
        msg = "I couldn't find the answer to this in your notes."
        if reason:
            msg += f" {reason}"
        return Answer(question, msg, "not_in_notes", sources=list(hits), model=model)

    answer = Answer(question, body, "answered", citations, list(hits), support, model=model)
    if response.stop_reason == "max_tokens":
        answer.warnings.append("The answer was cut off by the output token limit.")
    if not citations:
        answer.warnings.append(
            "This answer has no citations to your notes; treat it as unverified.")
    elif support < 0.5:
        answer.warnings.append(
            f"Only {support:.0%} of this answer is directly backed by cited passages.")
    return answer
