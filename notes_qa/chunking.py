"""Structure-aware chunking of lecture notes.

Notes are split along their own structure first (headings -> paragraphs ->
sentences) and then packed into chunks of roughly ``chunk_chars`` characters
with a small sentence overlap. Each chunk remembers its file, page and section
heading so retrieval can match on headings and answers can cite precisely.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass
from typing import Iterable

from notes_qa.loaders import Page

_MD_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_NUMBERED_HEADING = re.compile(r"^((?:\d+\.)*\d+)\.?\s+([A-Z][^.!?]{1,70})$")
_KEYWORD_HEADING = re.compile(
    r"^(chapter|unit|module|lecture|topic|part|section)\s+[\dIVXivx]+\b.{0,70}$", re.IGNORECASE
)
_BULLET = re.compile(r"^\s*(?:[-*•▪●]|\d+[.)]|[a-z][.)])\s+")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'“])")


@dataclass
class Chunk:
    id: str
    doc_id: str
    source: str
    page: int | None
    section: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**data)

    @property
    def location(self) -> str:
        parts = [self.source]
        if self.page is not None:
            parts.append(f"p. {self.page}")
        if self.section:
            parts.append(self.section)
        return " · ".join(parts)

    @property
    def search_text(self) -> str:
        """Text used for retrieval: the section heading is a strong topical signal."""
        return f"{self.section}\n{self.text}" if self.section else self.text


def document_id(source: str, pages: Iterable[Page]) -> str:
    h = hashlib.sha1(source.encode())
    for p in pages:
        h.update(p.text.encode("utf-8", errors="replace"))
    return h.hexdigest()[:12]


def _detect_heading(line: str) -> tuple[int, str] | None:
    """Return (level, title) if ``line`` looks like a heading."""
    stripped = line.strip()
    if not stripped:
        return None
    m = _MD_HEADING.match(stripped)
    if m:
        return len(m.group(1)), m.group(2).strip()
    if len(stripped) > 80:
        return None
    m = _NUMBERED_HEADING.match(stripped)
    if m:
        return m.group(1).count(".") + 1, stripped
    if _KEYWORD_HEADING.match(stripped):
        return 1, stripped
    words = stripped.split()
    if (
        2 <= len(words) <= 8
        and stripped.isupper()
        and sum(c.isalpha() for c in stripped) >= 6
    ):
        return 1, stripped.title()
    return None


def _split_units(paragraph: str) -> list[str]:
    """Split a paragraph into sentence-like units, keeping bullet items whole."""
    lines = [ln.strip() for ln in paragraph.splitlines() if ln.strip()]
    units: list[str] = []
    buf: list[str] = []
    for ln in lines:
        if _BULLET.match(ln):
            if buf:
                units.extend(_SENTENCE_SPLIT.split(" ".join(buf)))
                buf = []
            units.append(ln)
        else:
            buf.append(ln)
    if buf:
        units.extend(_SENTENCE_SPLIT.split(" ".join(buf)))
    return [u.strip() for u in units if u.strip()]


def _hard_wrap(unit: str, limit: int) -> list[str]:
    """Split a single over-long sentence (e.g. a table row dump) on word boundaries."""
    if len(unit) <= limit:
        return [unit]
    out, cur = [], ""
    for word in unit.split():
        if cur and len(cur) + 1 + len(word) > limit:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}" if cur else word
    if cur:
        out.append(cur)
    return out


class Chunker:
    def __init__(self, chunk_chars: int = 1200, overlap_sentences: int = 2):
        if chunk_chars < 200:
            raise ValueError("chunk_chars must be at least 200")
        self.chunk_chars = chunk_chars
        self.overlap_sentences = overlap_sentences

    def chunk(self, pages: list[Page]) -> list[Chunk]:
        if not pages:
            return []
        source = pages[0].source
        doc_id = document_id(source, pages)
        chunks: list[Chunk] = []
        headings: dict[int, str] = {}  # level -> title, persists across PDF pages

        for page in pages:
            section_units: list[str] = []
            section = self._section_path(headings)
            paragraph: list[str] = []

            def flush_paragraph() -> None:
                if paragraph:
                    section_units.extend(_split_units("\n".join(paragraph)))
                    paragraph.clear()

            for line in page.text.splitlines():
                heading = _detect_heading(line)
                if heading:
                    flush_paragraph()
                    self._emit(chunks, doc_id, page, section, section_units)
                    section_units = []
                    level, title = heading
                    headings = {k: v for k, v in headings.items() if k < level}
                    headings[level] = title
                    section = self._section_path(headings)
                elif not line.strip():
                    flush_paragraph()
                else:
                    paragraph.append(line)
            flush_paragraph()
            self._emit(chunks, doc_id, page, section, section_units)

        for i, c in enumerate(chunks):
            c.id = f"{doc_id}:{i}"
        return chunks

    @staticmethod
    def _section_path(headings: dict[int, str]) -> str:
        return " > ".join(headings[k] for k in sorted(headings))

    def _emit(self, chunks: list[Chunk], doc_id: str, page: Page, section: str,
              units: list[str]) -> None:
        units = [piece for u in units for piece in _hard_wrap(u, self.chunk_chars)]
        if not units:
            return
        max_overlap_chars = self.chunk_chars // 3
        current: list[str] = []
        size = 0
        fresh = 0  # units in `current` that were not carried over as overlap
        for unit in units:
            if current and fresh and size + len(unit) + 1 > self.chunk_chars:
                chunks.append(self._make(doc_id, page, section, current))
                overlap: list[str] = []
                if self.overlap_sentences:
                    for u in reversed(current[-self.overlap_sentences:]):
                        if sum(map(len, overlap)) + len(u) > max_overlap_chars:
                            break
                        overlap.insert(0, u)
                current = overlap
                size = sum(len(u) + 1 for u in current)
                fresh = 0
            current.append(unit)
            size += len(unit) + 1
            fresh += 1
        if fresh:
            chunks.append(self._make(doc_id, page, section, current))

    @staticmethod
    def _make(doc_id: str, page: Page, section: str, units: list[str]) -> Chunk:
        text = ""
        for u in units:
            if not text:
                text = u
            else:
                text += ("\n" if _BULLET.match(u) or _BULLET.match(text.rsplit("\n", 1)[-1]) else " ") + u
        return Chunk(id="", doc_id=doc_id, source=page.source, page=page.page,
                     section=section, text=text)
