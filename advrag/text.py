"""Tokenization shared by every retriever, so scores are comparable."""

import re

_TOKEN = re.compile(r"[a-z0-9_]+")

STOPWORDS = frozenset(
    """a an and are as at be but by can could did do does for from had has have how i if in
    into is it its may might must not of on or should so such than that the their them then
    there these they this those to was were what when where which who why will with would
    you your we our us""".split()
)


def normalize(token: str) -> str:
    """Very light stemming: fold a plural or -ing/-ed ending onto the base form."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 5 and token.endswith("ing"):
        return token[:-3]
    if len(token) > 4 and token.endswith("ed") and not token.endswith("eed"):
        return token[:-2]
    if len(token) > 3 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    return [normalize(t) for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS]
