"""纯 Python 向量索引与落盘格式。"""

from __future__ import annotations

import json
from pathlib import Path

from .chunking import Chunk, chunk_corpus
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder


class VectorIndex:
    def __init__(self, embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.vectors: list[list[float]] = []

    def add(self, chunk: Chunk, vector: list[float] | None = None) -> None:
        self.chunks.append(chunk)
        self.vectors.append(vector or self.embedder.embed_query(chunk.retrieval_text()))

    def search(self, query_vector: list[float], k: int = 10) -> list[tuple[Chunk, float]]:
        if self.vectors and len(query_vector) != len(self.vectors[0]):
            raise ValueError("查询向量维度与索引不一致，请用同一 Embedder 重新 ingest")
        scored = [
            (chunk, sum(a * b for a, b in zip(query_vector, vector)))
            for chunk, vector in zip(self.chunks, self.vectors)
        ]
        scored.sort(key=lambda item: (-item[1], item[0].doc_id, item[0].chunk_index))
        return scored[:k]

    @classmethod
    def build_from_corpus(cls, directory: str | Path, embedder) -> "VectorIndex":
        index = cls(embedder)
        chunks = chunk_corpus(directory)
        vectors = embedder.embed_documents([chunk.retrieval_text() for chunk in chunks])
        for chunk, vector in zip(chunks, vectors):
            index.add(chunk, vector)
        return index

    def save(self, index_dir: str | Path) -> None:
        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)
        with (index_dir / "chunks.jsonl").open("w", encoding="utf-8", newline="\n") as handle:
            for chunk, vector in zip(self.chunks, self.vectors):
                row = {**chunk.to_dict(), "vector": vector}
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        if isinstance(self.embedder, HashingEmbedder):
            embedder_meta = {"kind": "hashing", "dim": self.embedder.dim}
        else:
            embedder_meta = {
                "kind": "openai-compatible",
                "model": getattr(self.embedder, "model", "unknown"),
                "dim": getattr(self.embedder, "dim", len(self.vectors[0]) if self.vectors else 0),
            }
        meta = {"embedder": embedder_meta, "chunks": len(self.chunks), "docs": len({c.doc_id for c in self.chunks})}
        (index_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, index_dir: str | Path, embedder=None) -> "VectorIndex":
        index_dir = Path(index_dir)
        meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
        if embedder is None:
            if meta["embedder"]["kind"] == "hashing":
                embedder = HashingEmbedder(int(meta["embedder"]["dim"]))
            else:
                embedder = OpenAICompatibleEmbedder.from_env()
        index = cls(embedder)
        with (index_dir / "chunks.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                index.add(Chunk.from_dict(row), [float(x) for x in row["vector"]])
        return index
