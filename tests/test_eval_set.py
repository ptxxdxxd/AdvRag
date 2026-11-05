"""The eval set must stay consistent with the corpus it scores."""

from advrag.corpus import load_corpus
from advrag.evaluate import load_questions
from advrag.index import DEFAULT_CORPUS


def test_every_gold_chunk_exists_and_holds_the_evidence():
    chunks = {c.id: c for c in load_corpus(DEFAULT_CORPUS)}
    questions = load_questions()
    assert len(questions) >= 40
    assert len({q.id for q in questions}) == len(questions)
    for q in questions:
        for gold in q.gold:
            assert gold in chunks, f"{q.id}: unknown gold chunk {gold}"
        assert any(q.evidence in chunks[g].text for g in q.gold), q.id
