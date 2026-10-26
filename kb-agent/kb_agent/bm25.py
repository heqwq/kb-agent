"""BM25 检索：面向中英文混合文本。"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

_TOKEN = re.compile(r"[a-zA-Z0-9]+|[\u4e00-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """ASCII 单词保留原词，中文串切成字符 bigram。"""
    tokens: list[str] = []
    for match in _TOKEN.findall(text):
        if match[0].isascii():
            tokens.append(match.lower())
        elif len(match) == 1:
            tokens.append(match)
        else:
            tokens.extend(match[i : i + 2] for i in range(len(match) - 1))
    return tokens


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_ids: list[object] = []
        self.term_freqs: list[Counter[str]] = []
        self.lengths: list[int] = []
        self._df: Counter[str] = Counter()
        self._avg_length = 0.0

    def add(self, doc_id: object, text: str) -> None:
        tokens = tokenize(text)
        self.doc_ids.append(doc_id)
        self.term_freqs.append(Counter(tokens))
        self.lengths.append(len(tokens))
        self._df.update(set(tokens))
        self._avg_length = sum(self.lengths) / len(self.lengths)

    def idf(self, term: str) -> float:
        n = len(self.doc_ids)
        df = self._df.get(term, 0)
        return math.log(1 + (n - df + 0.5) / (df + 0.5))

    def search(self, query: str, k: int = 10) -> list[tuple[object, float]]:
        query_terms = set(tokenize(query))
        scored: list[tuple[object, float]] = []
        for index, doc_id in enumerate(self.doc_ids):
            tf = self.term_freqs[index]
            norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / (self._avg_length or 1))
            score = sum(
                self.idf(term) * tf[term] * (self.k1 + 1) / (tf[term] + norm)
                for term in query_terms
                if term in tf
            )
            if score > 0:
                scored.append((doc_id, score))
        scored.sort(key=lambda item: (-item[1], str(item[0])))
        return scored[:k]
