from advrag.corpus import load_corpus, slugify
from advrag.index import DEFAULT_CORPUS


def test_ids_are_unique_and_stable():
    chunks = load_corpus(DEFAULT_CORPUS)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert "pep-0008#maximum-line-length" in ids


def test_boilerplate_sections_are_dropped():
    sections = {c.section.lower() for c in load_corpus(DEFAULT_CORPUS)}
    assert "copyright" not in sections
    assert "references" not in sections


def test_long_sections_are_split():
    chunks = load_corpus(DEFAULT_CORPUS)
    assert any("/p2" in c.id for c in chunks)
    # Single paragraphs can run past the limit, but nothing should be huge.
    assert max(len(c.text.split()) for c in chunks) < 1000


def test_slugify():
    assert slugify("Should a Line Break Before or After a Binary Operator?") == (
        "should-a-line-break-before-or-after-a-binary-operator"
    )
