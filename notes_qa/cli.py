"""Command-line interface.

    notes-qa ingest lectures/ syllabus.pdf
    notes-qa ask "What is the difference between paging and segmentation?"
    notes-qa chat
    notes-qa search "deadlock conditions"     # retrieval only, no LLM call
    notes-qa list | remove <file> | clear

Ollama servers (a local instance, LAN machines, or both):
    notes-qa servers                                  # list with live status
    notes-qa servers add lab 192.168.1.50 --model llama3.1:70b --first
    notes-qa servers use auto | lab | local           # auto = fail over in order
    notes-qa servers prefer lab | remove lab | disable lab | enable lab
    notes-qa --server local ask "..."                 # one-off override
"""

from __future__ import annotations

import argparse
import sys
import textwrap

from notes_qa.config import Settings
from notes_qa.generator import Answer
from notes_qa.pipeline import Assistant
from notes_qa.servers import (AUTO, NoServerAvailable, OllamaRouter, OllamaServer,
                              ServerConfigError, ServerRegistry)


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
    if answer.server:
        print(f"\n(answered by {answer.model} on {answer.server})")
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
    parser.add_argument("--server", help="Ollama server name or 'auto' for this command")
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

    p = sub.add_parser("servers", help="configure local / LAN Ollama servers")
    ssub = p.add_subparsers(dest="action")
    ssub.add_parser("list", help="show servers and whether they are reachable")
    a = ssub.add_parser("add", help="add or update a server")
    a.add_argument("name")
    a.add_argument("host", help="e.g. 192.168.1.50, gpu-box:11434 or http://host:port")
    a.add_argument("--model", help="chat model to use on this server")
    a.add_argument("--first", action="store_true", help="try this server first in auto mode")
    for action, help_text in [("remove", "delete a server"),
                              ("use", "select a server, or 'auto' for failover"),
                              ("prefer", "move a server to the front of the auto order"),
                              ("enable", "include a server in auto mode"),
                              ("disable", "skip a server in auto mode")]:
        ssub.add_parser(action, help=help_text).add_argument("name")

    args = parser.parse_args(argv)
    settings = Settings()
    if args.index_dir:
        from pathlib import Path
        settings.index_dir = Path(args.index_dir)
    if args.server:
        settings.ollama_server = args.server
    try:
        if args.command == "servers":
            return _servers(args, settings)
        assistant = Assistant(settings)
        return _run(args, assistant)
    except ServerConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except NoServerAvailable as exc:
        print(f"error: no Ollama server could handle the request: {exc}", file=sys.stderr)
        return 1


def _run(args: argparse.Namespace, assistant: Assistant) -> int:
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


def _servers(args: argparse.Namespace, settings: Settings) -> int:
    registry = ServerRegistry.load(settings.servers_file, env_servers=settings.ollama_servers,
                                   local_host=settings.ollama_host)
    action = args.action or "list"
    if action != "list":
        if registry.from_env:
            raise ServerConfigError("servers come from NOTES_QA_OLLAMA_SERVERS; "
                                    "edit that variable or unset it to use the file")
        if action == "add":
            existing = args.name in registry.names
            registry.upsert(OllamaServer(args.name, args.host, args.model))
            if args.first:
                registry.move(args.name, 0)
            print(f"{'updated' if existing else 'added'} {registry.get(args.name).host}")
        elif action == "remove":
            registry.remove(args.name)
        elif action == "use":
            registry.select(args.name)
        elif action == "prefer":
            registry.move(args.name, 0)
        elif action in ("enable", "disable"):
            registry.get(args.name).enabled = action == "enable"
        registry.save()
        print(f"saved {registry.path}")

    router = OllamaRouter(registry, default_model=settings.model)
    selected = settings.ollama_server or registry.default
    print(f"selected: {selected}" + ("  (servers tried in this order)" if selected == AUTO else ""))
    for status in router.check_all(timeout=settings.connect_timeout):
        srv = status.server
        model = router.chat_model(srv)
        marker = "*" if selected in (srv.name, AUTO) and srv.enabled else " "
        line = f" {marker} {srv.name:<12} {srv.host:<32} model={model}"
        if not srv.enabled:
            line += "  [disabled]"
        if status.reachable:
            line += f"  up ({status.latency_ms:.0f} ms)"
            if not status.has_model(model):
                line += f"  ! model missing: run `ollama pull {model}` there"
        else:
            line += f"  DOWN ({status.error})"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
