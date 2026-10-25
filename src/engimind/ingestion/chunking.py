"""Recursive character splitter.

Interview talking point: we split on large semantic boundaries first
(paragraphs, then lines, then sentences, then spaces) so chunks stay
coherent. Overlap keeps a sentence that straddles a boundary in both
neighbors so retrieval does not drop it.

Other strategies we did not implement (see INTERVIEW.md):
- Semantic chunking (split where embedding similarity drops)
- Parent-document / small-to-big (retrieve small, pass parent to LLM)
- Late chunking (embed long context, then pool token vectors per span)
"""

import re
from dataclasses import dataclass

from engimind.ingestion.pdf_loader import Page

SEPARATORS = ["\n\n", "\n", ". ", " ", ""]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    text: str
    page: int
    source: str
    section: str = ""
    end_page: int | None = None


def chunk_pages(
    pages: list[Page],
    *,
    source: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")
    chunks: list[Chunk] = []
    seq = 0
    for page in pages:
        if not page.text:
            continue
        for section, paragraph in _sections_for_page(page.text):
            for part in recursive_split(paragraph, chunk_size, chunk_overlap):
                chunks.append(
                    Chunk(
                        chunk_id=f"{source}:{page.page_number}:{seq}",
                        text=part,
                        page=page.page_number,
                        source=source,
                        section=section,
                        end_page=page.end_page_number,
                    )
                )
                seq += 1
    if not chunks:
        raise ValueError("Document produced no chunks after splitting.")
    return chunks


def _looks_like_heading(line: str) -> bool:
    text = line.strip()
    if not text or len(text) > 80 or text.endswith((".", "!", "?")):
        return False
    if text.isupper() or text.endswith(":"):
        return True

    words = text.split()
    if len(words) > 8:
        return False

    allowed = {"-", "&", "/", "(", ")"}
    return all(
        w.isupper() or w.isdigit() or w in allowed or w[:1].isupper()
        for w in words
    )


def _clean_heading(line: str) -> str:
    return re.sub(r"\s+", " ", line.strip()).strip(" -:")


def _sections_for_page(text: str) -> list[tuple[str, str]]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n+", text) if b.strip()]
    if not blocks:
        return [("", re.sub(r"\s+", " ", text).strip())]

    current_section = ""
    sections: list[tuple[str, str]] = []
    for block in blocks:
        lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
        if not lines:
            continue

        heading_lines: list[str] = []
        body_lines: list[str] = []
        for line in lines:
            if _looks_like_heading(line):
                heading_lines.append(_clean_heading(line))
            else:
                body_lines.append(line)

        if heading_lines:
            current_section = " | ".join(heading_lines)
            if body_lines:
                sections.append((current_section, " ".join(body_lines)))
            continue

        sections.append((current_section, " ".join(lines)))
    return sections if sections else [("", re.sub(r"\s+", " ", text).strip())]


def recursive_split(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = re.sub(r"\s+", " ", text.strip())
    if not text:
        return []

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n+", text) if p.strip()]
    if not paragraphs:
        return []

    units: list[str] = []
    for paragraph in paragraphs:
        units.extend(_split_paragraph(paragraph, chunk_size))
    return _merge_units(units, chunk_size, chunk_overlap)


def _split_paragraph(paragraph: str, chunk_size: int) -> list[str]:
    if len(paragraph) <= chunk_size:
        return [paragraph]

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", paragraph) if s.strip()]
    if len(sentences) > 1:
        parts: list[str] = []
        buffer = ""
        for sentence in sentences:
            candidate = f"{buffer} {sentence}".strip() if buffer else sentence
            if len(candidate) <= chunk_size:
                buffer = candidate
                continue
            if buffer:
                parts.append(buffer)
            if len(sentence) <= chunk_size:
                buffer = sentence
            else:
                parts.extend(_split_by_words(sentence, chunk_size))
                buffer = ""
        if buffer:
            parts.append(buffer)
        return parts

    return _split_by_words(paragraph, chunk_size)


def _split_by_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current = ""
    for word in words:
        if len(word) > chunk_size:
            if current:
                chunks.append(current)
                current = ""
            chunks.extend(
                word[index : index + chunk_size]
                for index in range(0, len(word), chunk_size)
            )
            continue

        candidate = f"{current} {word}".strip() if current else word
        if len(candidate) <= chunk_size:
            current = candidate
            continue
        if current:
            chunks.append(current)
        current = word
    if current:
        chunks.append(current)
    return chunks


def _split_to_units(text: str, separators: list[str], chunk_size: int) -> list[str]:
    if len(text) <= chunk_size:
        return [text] if text else []
    sep, *rest = separators
    if sep == "":
        return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]

    parts = text.split(sep)
    units: list[str] = []
    for i, part in enumerate(parts):
        glued = part if i == 0 or not part else f"{sep}{part}"
        if not glued:
            continue
        if len(glued) <= chunk_size:
            units.append(glued)
        else:
            units.extend(_split_to_units(glued, rest, chunk_size))
    return units


def _merge_units(units: list[str], chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    buf = ""
    for unit in units:
        candidate = f"{buf} {unit}".strip() if buf else unit
        if len(candidate) <= chunk_size or not buf:
            buf = candidate
            if len(buf) >= chunk_size:
                chunks.append(buf.strip())
                buf = buf[-chunk_overlap:] if chunk_overlap else ""
            continue
        chunks.append(buf.strip())
        overlap = buf[-chunk_overlap:] if chunk_overlap else ""
        buf = (f"{overlap} {unit}".strip() if overlap else unit).strip()
        if len(buf) > chunk_size:
            extra = recursive_split(buf, chunk_size, 0)
            chunks.extend(extra[:-1])
            buf = extra[-1] if extra else ""
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]
