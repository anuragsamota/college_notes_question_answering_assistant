"""Exercise the real ``ollama`` client against a local stub of Ollama's HTTP API."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from notes_qa.config import Settings
from notes_qa.pipeline import Assistant


class StubOllama(BaseHTTPRequestHandler):
    requests: list = []
    installed_models = {"llama3.1:8b", "nomic-embed-text"}

    def log_message(self, *args):
        pass

    def _send(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        StubOllama.requests.append((self.path, req))
        if req["model"] not in self.installed_models:
            return self._send(404, {"error": f"model '{req['model']}' not found"})
        if self.path == "/api/embed":
            # Bag-of-letters vectors: deterministic and good enough to rank.
            vecs = []
            for text in req["input"]:
                v = [0.0] * 26
                for ch in text.lower():
                    if "a" <= ch <= "z":
                        v[ord(ch) - 97] += 1
                vecs.append(v)
            return self._send(200, {"model": req["model"], "embeddings": vecs})
        if self.path == "/api/chat":
            prompt = req["messages"][-1]["content"]
            n = next(line[1:line.index("]")] for line in prompt.splitlines()
                     if line.startswith("[") and "Banker" in line)
            content = ("The Banker's algorithm grants a request only if the resulting "
                       f"state is safe [{n}].")
            return self._send(200, {"model": req["model"], "created_at": "2026-01-01T00:00:00Z",
                                    "message": {"role": "assistant", "content": content},
                                    "done": True, "done_reason": "stop"})
        self._send(404, {"error": "unknown path"})


@pytest.fixture
def ollama_host():
    server = HTTPServer(("127.0.0.1", 0), StubOllama)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    StubOllama.requests = []
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def test_end_to_end_with_ollama_chat_and_embeddings(tmp_path, sample_path, ollama_host):
    settings = Settings()
    settings.index_dir = tmp_path / "index"
    settings.ollama_host = ollama_host
    settings.embedder = "ollama"
    assistant = Assistant(settings)
    assistant.ingest_paths([sample_path])
    assert assistant.retriever.embeddings.shape == (len(assistant.chunks), 26)

    answer = assistant.ask("How does the Banker's algorithm decide whether to grant a request?")
    assert answer.status == "answered" and answer.grounded
    assert answer.support_ratio == 1.0
    assert answer.cited_sources[0][1].chunk.section.endswith("Banker's Algorithm")

    paths = [p for p, _ in StubOllama.requests]
    assert paths.count("/api/embed") >= 2 and paths[-1] == "/api/chat"
    chat = StubOllama.requests[-1][1]
    assert chat["stream"] is False
    assert chat["options"]["num_ctx"] == 8192 and chat["options"]["temperature"] == 0.1

    # Embeddings are cached on disk with the index and reused on reload.
    before = len(StubOllama.requests)
    reloaded = Assistant(settings)
    assert reloaded.retriever.embeddings.shape == assistant.retriever.embeddings.shape
    assert len(StubOllama.requests) == before


def test_missing_model_message(tmp_path, sample_path, ollama_host):
    settings = Settings()
    settings.index_dir = tmp_path / "index"
    settings.ollama_host = ollama_host
    settings.model = "mistral:7b"
    assistant = Assistant(settings)
    assistant.ingest_paths([sample_path])
    answer = assistant.ask("What is Belady's anomaly?")
    assert answer.status == "error"
    assert "ollama pull mistral:7b" in answer.text


def test_ollama_not_running(tmp_path, sample_path):
    settings = Settings()
    settings.index_dir = tmp_path / "index"
    settings.ollama_host = "http://127.0.0.1:9"
    assistant = Assistant(settings)
    assistant.ingest_paths([sample_path])
    answer = assistant.ask("What is Belady's anomaly?")
    assert answer.status == "error" and "ollama serve" in answer.text
