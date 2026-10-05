import json
from types import SimpleNamespace as NS

import httpx
import ollama
import pytest

from notes_qa.cli import main as cli
from notes_qa.servers import (AUTO, NoServerAvailable, OllamaRouter, OllamaServer,
                              ServerConfigError, ServerRegistry, normalize_host,
                              parse_server_list)


@pytest.mark.parametrize("given,expected", [
    ("localhost", "http://localhost:11434"),
    ("192.168.1.50", "http://192.168.1.50:11434"),
    ("gpu-box:8080", "http://gpu-box:8080"),
    ("https://ollama.lab.example/", "https://ollama.lab.example:11434"),
    ("http://[fe80::1]:11434", "http://[fe80::1]:11434"),
])
def test_normalize_host(given, expected):
    assert normalize_host(given) == expected


@pytest.mark.parametrize("bad", ["", "ftp://x", "http://host:notaport"])
def test_normalize_host_rejects(bad):
    with pytest.raises(ServerConfigError):
        normalize_host(bad)


def test_server_names_validated():
    with pytest.raises(ServerConfigError):
        OllamaServer("auto", "localhost")
    with pytest.raises(ServerConfigError):
        ServerRegistry([OllamaServer("a", "x"), OllamaServer("a", "y")])


def test_default_registry_is_local(tmp_path):
    reg = ServerRegistry.load(tmp_path / "s.json")
    assert [(s.name, s.host) for s in reg.servers] == [("local", "http://localhost:11434")]
    assert reg.servers[0].is_local and reg.default == AUTO
    reg = ServerRegistry.load(tmp_path / "s.json", local_host="http://127.0.0.1:12000")
    assert reg.servers[0].host == "http://127.0.0.1:12000"


def test_registry_round_trip_and_editing(tmp_path):
    path = tmp_path / "s.json"
    reg = ServerRegistry.load(path)
    reg.upsert(OllamaServer("lab", "192.168.1.50", model="llama3.1:70b"))
    reg.move("lab", 0)
    reg.select("lab")
    reg.save()
    data = json.loads(path.read_text())
    assert data["default"] == "lab" and data["servers"][0]["model"] == "llama3.1:70b"

    reg = ServerRegistry.load(path)
    assert reg.names == ["lab", "local"] and reg.default == "lab"
    assert not reg.get("lab").is_local
    assert reg.candidates() == [reg.get("lab")]
    assert reg.candidates(AUTO) == reg.servers
    reg.get("local").enabled = False
    assert reg.candidates(AUTO) == [reg.get("lab")]
    reg.remove("lab")
    assert reg.default == AUTO
    with pytest.raises(ServerConfigError):
        reg.select("lab")


def test_env_server_list_overrides_file(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"servers": [{"name": "x", "host": "x"}]}))
    reg = ServerRegistry.load(path, env_servers="lab=10.0.0.5, local=localhost")
    assert reg.names == ["lab", "local"] and reg.from_env
    reg.save()                                        # never written back
    assert json.loads(path.read_text())["servers"][0]["name"] == "x"
    assert [s.name for s in parse_server_list("10.0.0.5,10.0.0.6")] == ["local", "server2"]


class FakeClient:
    def __init__(self, behaviour):
        self.behaviour = behaviour   # "ok", "down", "timeout", "nomodel"
        self.calls = 0

    def _maybe_fail(self):
        self.calls += 1
        if self.behaviour == "down":
            raise ConnectionError("down")
        if self.behaviour == "timeout":
            raise httpx.ConnectTimeout("timed out")
        if self.behaviour == "nomodel":
            raise ollama.ResponseError("model not found", 404)

    def chat(self, model, messages, options):
        self._maybe_fail()
        return NS(model=model)

    def embed(self, model, input):
        self._maybe_fail()
        return NS(embeddings=[[1.0, 0.0]] * len(input))

    def list(self):
        self._maybe_fail()
        return NS(models=[NS(model="llama3.1:latest"), NS(model="nomic-embed-text:latest")])


def router_for(behaviours, **kw):
    servers = [OllamaServer(name, f"10.0.0.{i + 1}") for i, name in enumerate(behaviours)]
    clients = {s.host: FakeClient(b) for s, b in zip(servers, behaviours.values())}
    router = OllamaRouter(ServerRegistry(servers), default_model="llama3.1",
                          client_factory=lambda host, timeout: clients[host], **kw)
    return router, {s.name: clients[s.host] for s in servers}


def test_failover_in_order_and_cooldown():
    router, clients = router_for({"lab": "timeout", "local": "ok"})
    _, server, model = router.chat([], {})
    assert server.name == "local" and model == "llama3.1"
    # The failed LAN server is now cooling down, so it is tried last.
    router.chat([], {})
    assert clients["lab"].calls == 1 and clients["local"].calls == 2


def test_cooldown_expires():
    router, clients = router_for({"lab": "down", "local": "ok"}, cooldown=0)
    router.chat([], {})
    router.chat([], {})
    assert clients["lab"].calls == 2


def test_missing_model_fails_over_but_bad_request_does_not():
    router, _ = router_for({"lab": "nomodel", "local": "ok"})
    assert router.chat([], {})[1].name == "local"
    router, _ = router_for({"only": "nomodel"})
    with pytest.raises(ollama.ResponseError):
        router.chat([], {})


def test_all_down_and_explicit_selection():
    router, _ = router_for({"lab": "down", "local": "timeout"})
    with pytest.raises(NoServerAvailable) as exc:
        router.chat([], {})
    assert "lab (http://10.0.0.1:11434): not reachable" in str(exc.value)
    assert "local (http://10.0.0.2:11434): timed out" in str(exc.value)

    router, clients = router_for({"lab": "ok", "local": "ok"})
    router.selection = "local"
    assert router.chat([], {})[1].name == "local" and clients["lab"].calls == 0


def test_per_server_model_and_embed():
    router, _ = router_for({"lab": "down", "local": "ok"})
    router.registry.get("lab").model = "llama3.1:70b"
    assert router.chat_model(router.registry.get("lab")) == "llama3.1:70b"
    vectors, server = router.embed(["a", "b"])
    assert len(vectors) == 2 and server.name == "local"


def test_check_reports_models():
    router, _ = router_for({"lab": "down", "local": "ok"})
    lab, local = router.check_all()
    assert not lab.reachable and lab.error == "not reachable"
    assert local.reachable and local.has_model("llama3.1") and local.has_model("llama3.1:latest")


def test_cli_server_management(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("NOTES_QA_SERVERS_FILE", str(tmp_path / "servers.json"))
    monkeypatch.delenv("NOTES_QA_OLLAMA_SERVERS", raising=False)
    monkeypatch.setenv("NOTES_QA_CONNECT_TIMEOUT", "0.5")
    assert cli(["servers", "add", "lab", "127.0.0.1:9", "--model", "llama3.1:70b",
                "--first"]) == 0
    assert cli(["servers", "use", "lab"]) == 0
    out = capsys.readouterr().out
    assert "selected: lab" in out and "lab" in out and "DOWN" in out
    data = json.loads((tmp_path / "servers.json").read_text())
    assert [s["name"] for s in data["servers"]] == ["lab", "local"]
    assert data["default"] == "lab"
    assert cli(["servers", "use", "nope"]) == 2
    assert "unknown Ollama server 'nope'" in capsys.readouterr().err
    assert cli(["servers", "remove", "lab"]) == 0
    assert json.loads((tmp_path / "servers.json").read_text())["default"] == AUTO
