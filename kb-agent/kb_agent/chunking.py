"""标题感知的 Markdown 分块器。"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

MAX_CHARS = 600
OVERLAP_CHARS = 80
_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
_SENTENCE_END = "。！？!?；;"


@dataclass(frozen=True)
class Chunk:
    text: str
    doc_id: str
    heading_path: tuple[str, ...]
    chunk_index: int

    def retrieval_text(self) -> str:
        """检索时同时利用标题路径，正文保持原样。"""
        return "\n".join((*self.heading_path, self.text))

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "doc_id": self.doc_id,
            "heading_path": list(self.heading_path),
            "chunk_index": self.chunk_index,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            text=data["text"],
            doc_id=data["doc_id"],
            heading_path=tuple(data["heading_path"]),
            chunk_index=int(data["chunk_index"]),
        )


def _sentences(text: str) -> list[str]:
    parts: list[str] = []
    start = 0
    for i, char in enumerate(text):
        if char in _SENTENCE_END:
            if text[start : i + 1].strip():
                parts.append(text[start : i + 1].strip())
            start = i + 1
    if text[start:].strip():
        parts.append(text[start:].strip())
    return parts or [text.strip()]


def _split_oversize(text: str, limit: int) -> list[str]:
    """优先在逗号、空白处切开超长句，避免生硬截断。"""
    pieces: list[str] = []
    rest = text.strip()
    while len(rest) > limit:
        window = rest[:limit]
        cut = max(window.rfind(ch) for ch in "，,、：: \n")
        cut = cut if cut > limit // 2 else limit
        pieces.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        pieces.append(rest)
    return pieces


def _units(body: str) -> list[str]:
    units: list[str] = []
    for block in re.split(r"\n\s*\n", body.strip()):
        block = block.strip()
        if not block:
            continue
        if len(block) <= MAX_CHARS:
            units.append(block)
            continue
        for sentence in _sentences(block):
            units.extend(_split_oversize(sentence, MAX_CHARS))
    return units


def _overlap_tail(text: str, target: int = OVERLAP_CHARS) -> str:
    tail = ""
    for sentence in reversed(_sentences(text)):
        if tail and len(tail) + len(sentence) > target * 2:
            break
        tail = sentence + tail
        if len(tail) >= target:
            break
    if not tail:
        tail = text[-target:]
    if len(tail) > target * 2:
        tail = tail[-target:]
        cut = max(tail.rfind(ch) for ch in "，,、：: \n")
        if cut > 10:
            tail = tail[cut + 1 :]
    return tail.strip()


def _sections(markdown: str) -> list[tuple[tuple[str, ...], str]]:
    sections: list[tuple[tuple[str, ...], str]] = []
    path: list[str] = []
    body: list[str] = []

    def flush() -> None:
        text = "\n".join(body).strip()
        if text:
            sections.append((tuple(path), text))
        body.clear()

    for line in markdown.splitlines():
        match = _HEADING.match(line)
        if match:
            flush()
            level = len(match.group(1))
            path = path[: level - 1] + [match.group(2).strip()]
        elif line.strip():
            body.append(line.rstrip())
    flush()
    return sections


def chunk_markdown(markdown: str, doc_id: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    for heading_path, body in _sections(markdown):
        current = ""
        for unit in _units(body):
            if current and len(current) + len(unit) + 1 > MAX_CHARS:
                chunks.append(Chunk(current.strip(), doc_id, heading_path, len(chunks)))
                room = MAX_CHARS - len(unit) - 1
                overlap = _overlap_tail(current) if room > 20 else ""
                if overlap and len(overlap) > room:
                    overlap = _overlap_tail(current, max(20, room))
                current = (overlap + "\n" + unit).strip() if overlap else unit
            else:
                current = (current + "\n" + unit).strip() if current else unit
        if current.strip():
            chunks.append(Chunk(current.strip(), doc_id, heading_path, len(chunks)))
    return chunks


def chunk_document(path: Path) -> list[Chunk]:
    return chunk_markdown(path.read_text(encoding="utf-8"), path.stem)


def chunk_corpus(directory: str | Path) -> list[Chunk]:
    directory = Path(directory)
    chunks: list[Chunk] = []
    for path in sorted(directory.glob("*.md")):
        chunks.extend(chunk_document(path))
    return chunks
