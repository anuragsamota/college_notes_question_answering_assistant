from types import SimpleNamespace as NS

import ollama

from notes_qa.generator import ABSTAIN_TOKEN, SYSTEM_PROMPT, parse_response
from notes_qa.pipeline import Assistant
from notes_qa.servers import OllamaRouter, OllamaServer, ServerRegistry


def reply(content, done_reason="stop"):
    return NS(message=NS(role="assistant", content=content), done_reason=done_reason)


class FakeOllama:
    def __init__(self, resp=None, error=None):
        self.resp = resp if resp is not None else reply("")
        self.error = error
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.resp


def fake_router(client, model="llama3.1:8b"):
    registry = ServerRegistry([OllamaServer("local", "localhost")])
    return OllamaRouter(registry, default_model=model,
                        client_factory=lambda host, timeout: client)


def make_assistant(settings, sample_path, resp=None, error=None):
    client = FakeOllama(resp, error)
    assistant = Assistant(settings, router=fake_router(client))
    assistant.ingest_paths([sample_path])
    return assistant, client


def excerpt_number(client, section):
    """Find which [n] the prompt assigned to the chunk from ``section``."""
    prompt = client.calls[0]["messages"][-1]["content"]
    for line in prompt.splitlines():
        if line.startswith("[") and section in line:
            return int(line[1:line.index("]")])
    raise AssertionError(f"{section} not in prompt")


def test_grounded_answer_is_verified(settings, sample_path):
    assistant, client = make_assistant(settings, sample_path)
    assistant.ask("What are the necessary conditions for deadlock?")
    n = excerpt_number(client, "Necessary Conditions")
    client.resp = reply(
        "**Answer:**\nThe four Coffman conditions are mutual exclusion, hold and wait, "
        f"no preemption and circular wait [{n}]. A deadlock can arise only if all four "
        f"conditions hold simultaneously. [{n}]")
    answer = assistant.ask("What are the necessary conditions for deadlock?")

    assert answer.status == "answered" and answer.grounded
    assert answer.server == "local" and answer.model == "llama3.1:8b"
    assert answer.support_ratio == 1.0
    assert answer.unsupported == [] and answer.warnings == []
    assert [num for num, _ in answer.cited_sources] == [n]
    assert "Coffman conditions" in answer.citations[0].cited_text

    call = client.calls[-1]
    assert call["model"] == "llama3.1:8b"
    assert call["options"]["num_ctx"] == 8192
    assert call["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}
    prompt = call["messages"][-1]["content"]
    assert prompt.startswith("Excerpts from the student's notes:")
    assert prompt.endswith("Question: What are the necessary conditions for deadlock?")


def test_hallucinated_claim_with_citation_is_caught(settings, sample_path):
    assistant, client = make_assistant(settings, sample_path)
    assistant.ask("What is Belady's anomaly?")
    n = excerpt_number(client, "Page Replacement")
    client.resp = reply(
        f"Belady's anomaly is when adding frames increases the number of page faults [{n}]. "
        f"It was discovered by Laszlo Belady at IBM Research in 1969 [{n}].")
    answer = assistant.ask("What is Belady's anomaly?")
    assert answer.status == "answered"
    assert answer.unsupported == ["It was discovered by Laszlo Belady at IBM Research in 1969."]
    assert 0 < answer.support_ratio < 1
    assert any("could not be matched" in w for w in answer.warnings)


def test_invalid_and_missing_citations(settings, sample_path):
    assistant, client = make_assistant(settings, sample_path,
                                       reply("Paging uses frames and pages [9]. "
                                             "It is used by every modern operating system."))
    answer = assistant.ask("What is paging?")
    assert "[9]" not in answer.text
    assert answer.text.startswith("Paging uses frames and pages.")
    assert not answer.grounded
    assert any("don't exist: [9]" in w for w in answer.warnings)
    assert any("no citations" in w for w in answer.warnings)
    assert len(answer.unsupported) == 2


def test_marker_after_full_stop_and_grouped():
    hit = NS(chunk=NS(text="Round Robin gives each process a fixed time quantum.",
                      search_text="Round Robin gives each process a fixed time quantum.",
                      location="x"))
    answer = parse_response("q", [hit, hit],
                            "Round Robin gives each process a fixed time quantum. [1, 2]")
    assert answer.text == "Round Robin gives each process a fixed time quantum. [1][2]"
    assert answer.support_ratio == 1.0


def test_model_abstention_and_think_tags(settings, sample_path):
    resp = reply(f"<think>The notes don't mention Linux.</think>\n{ABSTAIN_TOKEN}\n"
                 "The notes do not give the RR quantum used by Linux.")
    assistant, _ = make_assistant(settings, sample_path, resp)
    answer = assistant.ask("What round robin quantum does Linux use?")
    assert answer.status == "not_in_notes"
    assert answer.text.startswith("I couldn't find the answer")
    assert "Linux" in answer.text


def test_off_topic_question_skips_the_model(settings, sample_path):
    assistant, client = make_assistant(settings, sample_path)
    answer = assistant.ask("Who won the FIFA World Cup in 2018?")
    assert answer.status == "no_relevant_notes"
    assert client.calls == []


def test_ollama_errors_are_reported(settings, sample_path):
    assistant, _ = make_assistant(settings, sample_path,
                                  error=ollama.ResponseError("model not found", 404))
    answer = assistant.ask("What is paging?")
    assert answer.status == "error" and "ollama pull llama3.1:8b" in answer.text
    assert "'local'" in answer.text

    assistant, _ = make_assistant(settings, sample_path, error=ConnectionError("down"))
    answer = assistant.ask("What is paging?")
    assert answer.status == "error" and "ollama serve" in answer.text


def test_truncation_warning(settings, sample_path):
    assistant, _ = make_assistant(settings, sample_path,
                                  reply("Paging divides memory into frames [1]", "length"))
    answer = assistant.ask("What is paging?")
    assert any("cut off" in w for w in answer.warnings)


def test_follow_up_uses_history(settings, sample_path):
    assistant, client = make_assistant(settings, sample_path, reply("LRU does not [1]."))
    history = [("Which page replacement algorithm suffers from Belady's anomaly?",
                "FIFO does [1].")]
    result = assistant.retrieve("and LRU?", history)
    assert result.hits[0].chunk.section.endswith("Page Replacement")
    assistant.ask("and LRU?", history)
    msgs = client.calls[0]["messages"]
    assert msgs[2] == {"role": "assistant", "content": "FIFO does."}


def test_index_persists_and_replaces(settings, sample_path):
    assistant, _ = make_assistant(settings, sample_path)
    n = len(assistant.chunks)
    assistant.ingest_paths([sample_path])          # re-ingest replaces, not duplicates
    assert len(assistant.chunks) == n
    reloaded = Assistant(settings, router=fake_router(FakeOllama()))
    assert len(reloaded.chunks) == n
    assert [d.source for d in reloaded.documents()] == ["os_lecture_notes.md"]
    assert reloaded.remove("os_lecture_notes.md")
    assert reloaded.ask("What is paging?").status == "no_relevant_notes"
