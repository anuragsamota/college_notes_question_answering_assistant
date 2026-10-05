from notes_qa.chunking import Chunker
from notes_qa.loaders import Page, load_file


def test_sections_are_tracked(sample_path):
    chunks = Chunker(600, 1).chunk(load_file(sample_path))
    sections = {c.section for c in chunks}
    assert any(s.endswith("Unit 3: Deadlocks > Banker's Algorithm") for s in sections)
    banker = next(c for c in chunks if "Banker's Algorithm" in c.section)
    assert "Need = Max - Allocation" in banker.text
    assert all(c.id.startswith(c.doc_id + ":") for c in chunks)


def test_chunks_respect_size_and_overlap():
    sentences = [f"Sentence number {i} talks about topic {i}." for i in range(80)]
    page = Page("long.txt", " ".join(sentences), None)
    chunks = Chunker(300, 1).chunk([page])
    assert len(chunks) > 5
    assert all(len(c.text) <= 300 + 60 for c in chunks)
    # The last sentence of one chunk is repeated at the start of the next.
    for a, b in zip(chunks, chunks[1:]):
        assert b.text.startswith(a.text.split(". ")[-1].rstrip("."))


def test_pdf_style_headings_and_pages():
    pages = [
        Page("os.pdf", "UNIT TWO SCHEDULING\n2.1 Round Robin\nEach process gets a quantum.", 1),
        Page("os.pdf", "Short quanta increase context switches.\n2.2 Priority Scheduling\n"
                       "Aging prevents starvation.", 2),
    ]
    chunks = Chunker(400, 0).chunk(pages)
    assert [(c.page, c.section) for c in chunks] == [
        (1, "Unit Two Scheduling > 2.1 Round Robin"),
        (2, "Unit Two Scheduling > 2.1 Round Robin"),
        (2, "Unit Two Scheduling > 2.2 Priority Scheduling"),
    ]
    assert chunks[1].location == "os.pdf · p. 2 · Unit Two Scheduling > 2.1 Round Robin"


def test_bullets_kept_on_separate_lines(sample_path):
    chunks = Chunker(1200, 0).chunk(load_file(sample_path))
    states = next(c for c in chunks if c.section.endswith("Process States"))
    assert "\n- Ready: the process is waiting" in states.text
