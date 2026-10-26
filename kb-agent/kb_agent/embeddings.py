"""确定性哈希向量与可选的 OpenAI 兼容 Embedding 客户端。"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import urllib.request
from collections import Counter
from typing import Iterable

_ASCII = re.compile(r"[a-zA-Z0-9]+")
_CJK = re.compile(r"[\u4e00-\u9fff]+")


def _features(text: str) -> Iterable[str]:
    for word in _ASCII.findall(text.lower()):
        yield "w:" + word
    for run in _CJK.findall(text):
        if len(run) == 1:
            yield "c:" + run
        else:
            for i in range(len(run) - 1):
                yield "c:" + run[i : i + 2]


def _normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


class HashingEmbedder:
    """blake2b 特征哈希；同样输入在任何进程、任何机器上结果一致。"""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        vector = [0.0] * self.dim
        for feature, count in Counter(_features(text)).items():
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign * (1.0 + math.log(count))
        return _normalize(vector)


class OpenAICompatibleEmbedder:
    """仅在显式配置 KB_EMBED_BASE_URL / KB_EMBED_API_KEY 时使用。"""

    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.dim = 0

    @classmethod
    def from_env(cls) -> "OpenAICompatibleEmbedder":
        base_url = os.getenv("KB_EMBED_BASE_URL", "")
        api_key = os.getenv("KB_EMBED_API_KEY", "")
        if not base_url or not api_key:
            raise RuntimeError("缺少 KB_EMBED_BASE_URL 或 KB_EMBED_API_KEY，无法使用远程 Embedding。")
        return cls(base_url, api_key, os.getenv("KB_EMBED_MODEL", "text-embedding-3-small"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib.request.Request(
            self.base_url + "/embeddings",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                data = json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # 不把密钥或请求头写进异常
            raise RuntimeError(f"Embedding 请求失败：{type(exc).__name__}: {exc}") from None
        vectors = [_normalize([float(x) for x in item["embedding"]]) for item in data["data"]]
        self.dim = len(vectors[0]) if vectors else self.dim
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]
