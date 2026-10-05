"""Measure retrieval quality and the relevance gate on a labelled question set.

    python eval/eval_retrieval.py [eval/retrieval_questions.json]

For answerable questions it reports hit@1 / hit@k (the expected section is the
top chunk / among the chunks sent to Claude) and how often the gate wrongly
declines. For unanswerable questions (section = null) it reports how often the
gate correctly declines before any model call. No API key needed.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from notes_qa import Assistant, Settings  # noqa: E402


def main(path: str) -> int:
    spec = json.loads(Path(path).read_text())
    settings = Settings()
    settings.index_dir = Path(tempfile.mkdtemp()) / "index"
    assistant = Assistant(settings)
    assistant.ingest_paths([ROOT / p for p in spec["notes"]])

    answerable = [q for q in spec["questions"] if q["section"]]
    unanswerable = [q for q in spec["questions"] if not q["section"]]
    hit1 = hitk = false_declines = 0
    for item in answerable:
        result = assistant.retrieve(item["q"])
        sections = [h.chunk.section for h in result.hits]
        ok1 = bool(sections) and sections[0].endswith(item["section"])
        okk = any(s.endswith(item["section"]) for s in sections)
        hit1 += ok1
        hitk += okk
        false_declines += not result.sufficient
        flag = "ok " if okk and result.sufficient else "MISS"
        print(f"{flag} cov={result.term_coverage:.2f} {item['q']}")
    correct_declines = 0
    for item in unanswerable:
        result = assistant.retrieve(item["q"])
        correct_declines += not result.sufficient
        flag = "ok " if not result.sufficient else "LEAK"
        print(f"{flag} cov={result.term_coverage:.2f} {item['q']}")

    n, m = len(answerable), len(unanswerable)
    print(f"\nanswerable ({n}): hit@1 {hit1 / n:.0%}  hit@{settings.top_k} {hitk / n:.0%}  "
          f"wrongly declined {false_declines / n:.0%}")
    if m:
        print(f"unanswerable ({m}): declined before generation {correct_declines / m:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else str(ROOT / "eval/retrieval_questions.json")))
