from types import SimpleNamespace as NS

from notes_qa.generator import ABSTAIN_TOKEN, FALLBACK_BETA, AnswerGenerator
from notes_qa.pipeline import Assistant


def text_block(text, doc_indexes=()):
    cits = [NS(type="content_block_location", document_index=i, cited_text=f"quote {i}",
               start_block_index=0, end_block_index=1) for i in doc_indexes]
    return NS(type="text", text=text, citations=cits or None)


def response(*blocks, stop_reason="end_turn"):
    return NS(content=[NS(type="thinking", thinking=""), *blocks],
              stop_reason=stop_reason, model="claude-opus-5-5")


class FakeClient:
    def __init__(self, resp):
        self.resp = resp
        self.calls = []
        self.beta = NS(messages=NS(create=self._create))
        self.messages = NS(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self.resp


def make_assistant(settings, sample_path, resp):
    client = FakeClient(resp)
    assistant = Assistant(settings, generator=AnswerGenerator(client))
    assistant.ingest_paths([sample_path])
    return assistant, client


def test_grounded_answer_with_citations(settings, sample_path):
    resp = response(
        text_block("The four Coffman conditions are mutual exclusion, hold and wait, "
                   "no preemption and circular wait.", [0]),
        text_block("\n\nAll four must hold at the same time.", [0, 1]),
    )
    assistant, client = make_assistant(settings, sample_path, resp)
    answer = assistant.ask("What are the necessary conditions for deadlock?")

    assert answer.status == "answered" and answer.grounded
    assert "circular wait. [1]" in answer.text
    assert answer.text.endswith("same time. [1][2]")
    assert answer.support_ratio == 1.0
    assert answer.warnings == []
    assert [n for n, _ in answer.cited_sources] == [1, 2]

    call = client.calls[0]
    assert call["model"] == "claude-opus-5-5"
    assert call["betas"] == [FALLBACK_BETA] and call["fallbacks"] == "default"
    content = call["messages"][-1]["content"]
    docs = [b for b in content if b["type"] == "document"]
    assert docs and all(d["citations"] == {"enabled": True} for d in docs)
    assert docs[0]["title"].startswith("[1] os_lecture_notes.md")
    assert content[-1] == {"type": "text",
                           "text": "Question: What are the necessary conditions for deadlock?"}


def test_uncited_answer_is_flagged(settings, sample_path):
    assistant, _ = make_assistant(settings, sample_path,
                                  response(text_block("Deadlocks are bad.")))
    answer = assistant.ask("What is a deadlock?")
    assert answer.status == "answered" and not answer.grounded
    assert any("no citations" in w for w in answer.warnings)


def test_model_abstention(settings, sample_path):
    resp = response(text_block(f"{ABSTAIN_TOKEN}\nThe notes do not give the RR quantum "
                               "used by Linux."))
    assistant, _ = make_assistant(settings, sample_path, resp)
    answer = assistant.ask("What round robin quantum does Linux use?")
    assert answer.status == "not_in_notes"
    assert answer.text.startswith("I couldn't find the answer")
    assert "Linux" in answer.text


def test_off_topic_question_skips_the_model(settings, sample_path):
    assistant, client = make_assistant(settings, sample_path, response(text_block("x")))
    answer = assistant.ask("Who won the FIFA World Cup in 2018?")
    assert answer.status == "no_relevant_notes"
    assert client.calls == []


def test_refusal(settings, sample_path):
    assistant, _ = make_assistant(settings, sample_path, response(stop_reason="refusal"))
    assert assistant.ask("What is paging?").status == "refused"


def test_follow_up_uses_history(settings, sample_path):
    resp = response(text_block("LRU does not.", [0]))
    assistant, client = make_assistant(settings, sample_path, resp)
    history = [("Which page replacement algorithm suffers from Belady's anomaly?",
                "FIFO does. [1]")]
    result = assistant.retrieve("and LRU?", history)
    assert result.hits[0].chunk.section.endswith("Page Replacement")
    assistant.ask("and LRU?", history)
    msgs = client.calls[0]["messages"]
    assert msgs[1] == {"role": "assistant", "content": "FIFO does."}


def test_index_persists_and_replaces(settings, sample_path):
    assistant, _ = make_assistant(settings, sample_path, response())
    n = len(assistant.chunks)
    assistant.ingest_paths([sample_path])          # re-ingest replaces, not duplicates
    assert len(assistant.chunks) == n
    reloaded = Assistant(settings, generator=AnswerGenerator(FakeClient(response())))
    assert len(reloaded.chunks) == n
    assert [d.source for d in reloaded.documents()] == ["os_lecture_notes.md"]
    assert reloaded.remove("os_lecture_notes.md")
    assert reloaded.ask("What is paging?").status == "no_relevant_notes"


def test_fallbacks_can_be_disabled(settings, sample_path):
    client = FakeClient(response(text_block("Frames and pages.", [0])))
    assistant = Assistant(settings, generator=AnswerGenerator(client, use_fallbacks=False))
    assistant.ingest_paths([sample_path])
    assistant.ask("What is paging?")
    assert "betas" not in client.calls[0] and "fallbacks" not in client.calls[0]
