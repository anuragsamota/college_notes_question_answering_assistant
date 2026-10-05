"""Ollama server configuration: a local instance, LAN-hosted instances, or both.

Servers are kept in a small JSON file (``ollama_servers.json`` by default)::

    {
      "default": "auto",
      "servers": [
        {"name": "lab-gpu", "host": "http://192.168.1.50:11434", "model": "llama3.1:70b"},
        {"name": "local",   "host": "http://localhost:11434",    "model": "llama3.2:3b"}
      ]
    }

``default`` is either a server name or ``"auto"``. In auto mode servers are
tried in list order: if one is unreachable, times out, or lacks the model, the
next one answers. Each server can name its own chat model, so a powerful LAN
machine can run a larger model than the laptop fallback. The embedding model
(when dense retrieval is on) is the same everywhere, because vectors from
different models can't be mixed in one index.

Instead of the file, ``NOTES_QA_OLLAMA_SERVERS="lab=192.168.1.50,local=localhost"``
defines the list for one session.
"""

from __future__ import annotations

import json
import time
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence

import httpx
import ollama

AUTO = "auto"
DEFAULT_PORT = 11434


class ServerConfigError(ValueError):
    pass


class NoServerAvailable(RuntimeError):
    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors) or "no Ollama server configured")


def normalize_host(host: str) -> str:
    """Accept '192.168.1.5', 'gpu-box:11434' or full URLs; return 'http://host:port'."""
    host = host.strip()
    if not host:
        raise ServerConfigError("server address is empty")
    if "://" not in host:
        host = "http://" + host
    parsed = urllib.parse.urlsplit(host)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        raise ServerConfigError(f"invalid Ollama server address: {host!r}")
    try:
        port = parsed.port or DEFAULT_PORT
    except ValueError as exc:
        raise ServerConfigError(f"invalid port in {host!r}") from exc
    hostname = f"[{parsed.hostname}]" if ":" in parsed.hostname else parsed.hostname
    path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{hostname}:{port}{path}"


@dataclass
class OllamaServer:
    name: str
    host: str
    model: str | None = None   # chat model on this server (else the global default)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.name = self.name.strip()
        if not self.name or self.name == AUTO or "," in self.name or "=" in self.name:
            raise ServerConfigError(f"invalid server name: {self.name!r}")
        self.host = normalize_host(self.host)
        self.model = self.model or None

    @property
    def is_local(self) -> bool:
        hostname = urllib.parse.urlsplit(self.host).hostname or ""
        return hostname in ("localhost", "127.0.0.1", "::1")


@dataclass
class ServerStatus:
    server: OllamaServer
    reachable: bool
    models: list[str] = field(default_factory=list)
    latency_ms: float | None = None
    error: str | None = None

    def has_model(self, model: str) -> bool:
        # "llama3.1" means "llama3.1:latest" to Ollama.
        names = set(self.models) | {m.removesuffix(":latest") for m in self.models}
        return model in names


class ServerRegistry:
    """The configured servers plus which one (or ``auto``) is selected."""

    def __init__(self, servers: Sequence[OllamaServer] = (), default: str = AUTO,
                 path: Path | None = None, from_env: bool = False):
        self.servers: list[OllamaServer] = list(servers)
        self.default = default
        self.path = path
        self.from_env = from_env   # defined by an env var: don't write it back to disk
        self._check_unique()
        if default != AUTO and default not in self.names:
            self.default = AUTO

    @property
    def names(self) -> list[str]:
        return [s.name for s in self.servers]

    def _check_unique(self) -> None:
        if len(set(self.names)) != len(self.names):
            raise ServerConfigError("server names must be unique")

    # ---------------------------------------------------------------- loading
    @classmethod
    def load(cls, path: str | Path, *, env_servers: str = "",
             local_host: str | None = None) -> "ServerRegistry":
        path = Path(path)
        if env_servers.strip():
            return cls(parse_server_list(env_servers), path=path, from_env=True)
        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                servers = [OllamaServer(**s) for s in data.get("servers", [])]
            except (json.JSONDecodeError, TypeError) as exc:
                raise ServerConfigError(f"could not read {path}: {exc}") from exc
            return cls(servers, data.get("default", AUTO), path)
        return cls([OllamaServer("local", local_host or "localhost")], AUTO, path)

    def save(self) -> None:
        if self.path is None or self.from_env:
            return
        payload = {"default": self.default, "servers": [asdict(s) for s in self.servers]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp.replace(self.path)

    # ---------------------------------------------------------------- editing
    def get(self, name: str) -> OllamaServer:
        for s in self.servers:
            if s.name == name:
                return s
        raise ServerConfigError(f"unknown Ollama server {name!r}; known: {', '.join(self.names)}")

    def upsert(self, server: OllamaServer) -> None:
        for i, s in enumerate(self.servers):
            if s.name == server.name:
                self.servers[i] = server
                return
        self.servers.append(server)

    def remove(self, name: str) -> None:
        self.servers.remove(self.get(name))
        if self.default == name:
            self.default = AUTO

    def move(self, name: str, position: int) -> None:
        server = self.get(name)
        self.servers.remove(server)
        self.servers.insert(max(0, min(position, len(self.servers))), server)

    def select(self, name: str) -> None:
        if name != AUTO:
            self.get(name)
        self.default = name

    def candidates(self, selection: str | None = None) -> list[OllamaServer]:
        """Servers to try for a request, in order."""
        selection = selection or self.default
        if selection == AUTO:
            return [s for s in self.servers if s.enabled]
        return [self.get(selection)]


def parse_server_list(spec: str) -> list[OllamaServer]:
    """Parse ``name=host[,name=host...]``; a bare host is named after its position."""
    servers = []
    for i, item in enumerate(p.strip() for p in spec.split(",") if p.strip()):
        name, sep, host = item.partition("=")
        if not sep:
            name, host = ("local" if i == 0 else f"server{i + 1}"), item
        servers.append(OllamaServer(name.strip(), host.strip()))
    return servers


ClientFactory = Callable[[str, httpx.Timeout], Any]


def _default_client(host: str, timeout: httpx.Timeout) -> ollama.Client:
    return ollama.Client(host=host, timeout=timeout)


def _describe(exc: Exception) -> str:
    if isinstance(exc, ConnectionError):
        return "not reachable"
    if isinstance(exc, httpx.TimeoutException):
        return "timed out"
    if isinstance(exc, ollama.ResponseError):
        return f"error {exc.status_code}: {exc.error}"
    return f"{type(exc).__name__}: {exc}"


# Errors that mean "try the next server" rather than "the request is bad".
_FAILOVER_ERRORS = (ConnectionError, httpx.TransportError)


class OllamaRouter:
    """Sends chat and embedding requests to the selected server, failing over in auto mode.

    A server that fails is skipped for ``cooldown`` seconds so that a switched-off
    LAN machine doesn't add a connect timeout to every question.
    """

    def __init__(self, registry: ServerRegistry, *, default_model: str,
                 embed_model: str = "nomic-embed-text",
                 connect_timeout: float = 3.0, read_timeout: float = 300.0,
                 cooldown: float = 30.0, client_factory: ClientFactory = _default_client):
        self.registry = registry
        self.selection: str | None = None   # None -> registry.default
        self.default_model = default_model
        self.embed_model_name = embed_model
        self.timeout = httpx.Timeout(read_timeout, connect=connect_timeout)
        self.cooldown = cooldown
        self.client_factory = client_factory
        self._clients: dict[tuple[str, float], Any] = {}
        self._down_until: dict[str, float] = {}
        self.last_server: OllamaServer | None = None      # last server that answered
        self.last_attempt: OllamaServer | None = None     # last server tried

    # ---------------------------------------------------------------- helpers
    def client(self, server: OllamaServer, timeout: httpx.Timeout | None = None):
        timeout = timeout or self.timeout
        key = (server.host, timeout.read or 0.0)
        if key not in self._clients:
            self._clients[key] = self.client_factory(server.host, timeout)
        return self._clients[key]

    def chat_model(self, server: OllamaServer) -> str:
        return server.model or self.default_model

    def _ordered(self) -> list[OllamaServer]:
        servers = self.registry.candidates(self.selection)
        if len(servers) <= 1:
            return servers
        now = time.monotonic()
        healthy = [s for s in servers if self._down_until.get(s.host, 0) <= now]
        cooling = [s for s in servers if s not in healthy]
        return healthy + cooling   # still try cooling servers last rather than give up

    def _run(self, action: Callable[[OllamaServer], Any]) -> tuple[Any, OllamaServer]:
        servers = self._ordered()
        if not servers:
            raise NoServerAvailable(["no Ollama server is enabled"])
        errors: list[str] = []
        for server in servers:
            self.last_attempt = server
            try:
                result = action(server)
            except ollama.ResponseError as exc:
                # A missing model on one server is worth a retry elsewhere.
                if exc.status_code == 404 and len(servers) > 1:
                    errors.append(f"{server.name}: {_describe(exc)}")
                    continue
                raise
            except _FAILOVER_ERRORS as exc:
                self._down_until[server.host] = time.monotonic() + self.cooldown
                errors.append(f"{server.name} ({server.host}): {_describe(exc)}")
                continue
            self._down_until.pop(server.host, None)
            self.last_server = server
            return result, server
        raise NoServerAvailable(errors)

    # ---------------------------------------------------------------- requests
    def chat(self, messages: list[dict], options: dict) -> tuple[Any, OllamaServer, str]:
        def action(server: OllamaServer):
            return self.client(server).chat(model=self.chat_model(server),
                                            messages=messages, options=options)
        response, server = self._run(action)
        return response, server, self.chat_model(server)

    def embed(self, texts: list[str]) -> tuple[list[list[float]], OllamaServer]:
        def action(server: OllamaServer):
            return self.client(server).embed(model=self.embed_model_name,
                                             input=texts).embeddings
        return self._run(action)

    def check(self, server: OllamaServer, timeout: float = 3.0) -> ServerStatus:
        """Probe a server: reachable? which models does it have?"""
        client = self.client(server, httpx.Timeout(timeout))
        start = time.monotonic()
        try:
            listing = client.list()
        except (ollama.ResponseError, *_FAILOVER_ERRORS) as exc:
            return ServerStatus(server, False, error=_describe(exc))
        latency = (time.monotonic() - start) * 1000
        models = sorted(m.model for m in listing.models if m.model)
        return ServerStatus(server, True, models, latency)

    def check_all(self, timeout: float = 3.0) -> list[ServerStatus]:
        from concurrent.futures import ThreadPoolExecutor

        servers = self.registry.servers
        if not servers:
            return []
        with ThreadPoolExecutor(max_workers=min(8, len(servers))) as pool:
            return list(pool.map(lambda s: self.check(s, timeout), servers))
