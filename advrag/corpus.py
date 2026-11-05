"""Load PEP reStructuredText files and split them into section-level chunks.

Each chunk keeps a stable id of the form ``pep-0008#indentation`` so answers can
cite exactly where a fact came from, and the eval set can name its gold chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Underline characters PEPs use for section headings, in rough order of level.
_UNDERLINE = re.compile(r"^([=\-~^\"'`+*#])\1{2,}\s*$")
_SKIP_SECTIONS = {"copyright", "references", "footnotes", "acknowledgements", "acknowledgments"}
MAX_WORDS = 400  # long sections are split into parts of about this many words


@dataclass(frozen=True)
class Chunk:
    id: str
    pep: str
    pep_title: str
    section: str
    text: str

    @property
    def label(self) -> str:
        return f"PEP {int(self.pep)}: {self.pep_title} > {self.section}".replace("``", "`")


def slugify(heading: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", heading.lower()).strip("-")
    return slug or "section"


def _parse_header(lines: list[str]) -> tuple[dict[str, str], int]:
    """Read the RFC 822 style header block at the top of a PEP."""
    header: dict[str, str] = {}
    key = None
    i = 0
    for i, line in enumerate(lines):
        if not line.strip():
            break
        if line[:1].isspace() and key:
            header[key] += " " + line.strip()
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            header[key.strip()] = value.strip()
    return header, i + 1


def _sections(lines: list[str]) -> list[tuple[str, list[str]]]:
    """Split body lines into (heading, body lines) pairs on underlined headings."""
    sections: list[tuple[str, list[str]]] = []
    heading = "Preamble"
    body: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        is_heading = (
            line.strip()
            and not line[:1].isspace()
            and _UNDERLINE.match(nxt)
            and len(nxt.rstrip()) >= len(line.rstrip())
        )
        if is_heading:
            sections.append((heading, body))
            heading, body = line.strip(), []
            i += 2
            continue
        # Overlined headings (=== / Title / ===) also occur; drop the bare rule lines.
        if not _UNDERLINE.match(line):
            body.append(line)
        i += 1
    sections.append((heading, body))
    return sections


def _split_words(text: str, max_words: int) -> list[str]:
    paragraphs = [p.strip("\n") for p in re.split(r"\n\s*\n", text) if p.strip()]
    parts: list[str] = []
    current: list[str] = []
    count = 0
    for para in paragraphs:
        words = len(para.split())
        if current and count + words > max_words:
            parts.append("\n\n".join(current))
            current, count = [], 0
        current.append(para)
        count += words
    if current:
        parts.append("\n\n".join(current))
    return parts


def load_pep(path: Path) -> list[Chunk]:
    raw = path.read_text(encoding="utf-8")
    lines = raw.splitlines()
    header, start = _parse_header(lines)
    pep = header.get("PEP", path.stem.split("-")[-1]).zfill(4)
    title = header.get("Title", path.stem)
    chunks: list[Chunk] = []
    seen: dict[str, int] = {}
    for heading, body in _sections(lines[start:]):
        if heading.lower() in _SKIP_SECTIONS:
            continue
        text = "\n".join(body).strip()
        if len(text.split()) < 5:
            continue
        base = f"pep-{pep}#{slugify(heading)}"
        seen[base] = seen.get(base, 0) + 1
        if seen[base] > 1:
            base = f"{base}-{seen[base]}"
        parts = _split_words(text, MAX_WORDS)
        for n, part in enumerate(parts, start=1):
            cid = base if len(parts) == 1 else f"{base}/p{n}"
            chunks.append(Chunk(id=cid, pep=pep, pep_title=title, section=heading, text=part))
    return chunks


def load_corpus(directory: Path | str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for path in sorted(Path(directory).glob("pep-*.rst")):
        chunks.extend(load_pep(path))
    ids = [c.id for c in chunks]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate chunk ids in corpus")
    return chunks
