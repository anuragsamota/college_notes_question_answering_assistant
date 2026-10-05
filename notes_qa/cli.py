"""Command-line interface.

    notes-qa ingest lectures/ syllabus.pdf
    notes-qa ask "What is the difference between paging and segmentation?"
    notes-qa chat
    notes-qa search "deadlock conditions"     # retrieval only, no API call
    notes-qa list | remove <file> | clear
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from notes_qa.config import Settings
from notes_qa.generator import Answer
from notes_qa.pipeline import Assistant


def _print_answer(answer: Answer, show_sources: bool) -> None:
    print()
    print(answer.text)
    if answer.cited_sources:
        print("\nSources:")
        for n, hit in answer.cited_sources:
            print(f"  [{n}] {hit.chunk.location}")
    for w in answer.warnings:
        print(f"\n! {w}")
    for sentence in answer.unsupported:
        print(f"  - {textwrap.shorten(sentence, 160)}")
    if show_sources and answer.citations:
        print("\nCited passages:")
        for c in answer.citations:
            quote = textwrap.shorten(c.cited_text.replace("\n", " "), 220)
            print(f"  [{c.number}] \"{quote}\"")
    print()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="notes-qa", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--index-dir", help="index location (default .notes_index)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="index note files or folders")
    p.add_argument("paths", nargs="+")
    p = sub.add_parser("ask", help="ask one question")
    p.add_argument("question", nargs="+")
    p.add_argument("--quotes", action="store_true", help="show the cited passages")
    p = sub.add_parser("chat", help="interactive question loop with follow-ups")
    p.add_argument("--quotes", action="store_true")
    p = sub.add_parser("search", help="show retrieved chunks without calling the LLM")
    p.add_argument("query", nargs="+")
    sub.add_parser("list", help="list indexed documents")
    p = sub.add_parser("remove", help="remove a document by file name")
    p.add_argument("source")
    sub.add_parser("clear", help="delete the whole index")

    args = parser.parse_args(argv)
    settings = Settings()
    if args.index_dir:
        from pathlib import Path
        settings.index_dir = Path(args.index_dir)
    assistant = Assistant(settings)

    if args.command == "ingest":
        for name, n in assistant.ingest_paths(args.paths).items():
            print(f"indexed {name}: {n} chunks")
        print(f"{len(assistant.chunks)} chunks across {len(assistant.documents())} documents")
    elif args.command == "list":
        docs = assistant.documents()
        if not docs:
            print("No documents indexed.")
        for d in docs:
            pages = f", {d.pages} pages" if d.pages else ""
            print(f"{d.source}  ({d.chunks} chunks{pages})")
    elif args.command == "remove":
        print("removed" if assistant.remove(args.source) else f"not found: {args.source}")
    elif args.command == "clear":
        assistant.clear()
        print("index cleared")
    elif args.command == "search":
        result = assistant.retrieve(" ".join(args.query))
        print(f"sufficient={result.sufficient} max_similarity={result.max_similarity:.3f} "
              f"term_coverage={result.term_coverage:.2f} missing={result.missing_terms}")
        for n, hit in enumerate(result.hits, 1):
            print(f"\n[{n}] score={hit.score:.2f} sim={hit.similarity:.3f}  {hit.chunk.location}")
            print(textwrap.indent(textwrap.shorten(hit.chunk.text, 400), "    "))
    elif args.command == "ask":
        _print_answer(assistant.ask(" ".join(args.question)), args.quotes)
    elif args.command == "chat":
        history: list[tuple[str, str]] = []
        print("Ask about your notes (empty line or Ctrl-D to quit).")
        while True:
            try:
                question = input("> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if not question:
                break
            answer = assistant.ask(question, history[-3:])
            _print_answer(answer, args.quotes)
            if answer.status == "answered":
                history.append((question, answer.text))
    return 0


if __name__ == "__main__":
    sys.exit(main())
