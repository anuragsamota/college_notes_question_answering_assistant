from pathlib import Path

import pytest

from notes_qa.config import Settings

SAMPLE = Path(__file__).resolve().parent.parent / "sample_notes" / "os_lecture_notes.md"


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.index_dir = tmp_path / "index"
    s.embedder = "tfidf"
    s.servers_file = tmp_path / "ollama_servers.json"
    s.ollama_servers = ""
    s.ollama_server = ""
    return s


@pytest.fixture
def sample_path() -> Path:
    return SAMPLE
