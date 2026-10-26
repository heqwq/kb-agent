"""BM25 + 稠密检索的 RRF 混合检索。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable, Sequence

from .bm25 import BM25Index
from .index import VectorIndex

# 仅作用于 Dense 查询侧的轻量领域词表：把口语说法映射到制度术语。
# 它是可审计的语义先验，不是训练得到的语义模型；报告与 README 均会披露。
_DENSE_ALIASES: tuple[tuple[str, str], ...] = (
    ("结婚", "婚假 领证 10 天"),
    ("半夜", "夜间 值班 on-call 告警"),
    ("系统挂了", "故障 全站不可用 响应"),
    ("刚入职", "入职 第一周 必修课 Buddy 30-60-90 新人训练营"),
    ("第一周", "入职 必修课 Buddy 30-60-90 新人训练营"),
    ("电脑罢工", "设备 故障报修 IT 服务台"),
    ("聊天机器人", "外部 AI 工具 L3 L4 数据安全"),
    ("考证", "认证 考试费 报销 70% 8000 元"),
    ("薪水", "工资 发薪 15 日 薪酬"),
    ("升职", "晋升 评审 4 月 10 月"),
    ("证件", "身份证 学历学位 离职证明 银行卡"),
    ("外地回来", "出差 报销 15 个工作日 发票"),
    ("交单据", "报销 发票 支付凭证 出差审批单"),
    ("装个新工具", "软件 安装 IT 服务台 申请"),
    ("发给客户", "数据 外发 脱敏 法务 数据 Owner 安全团队"),
    ("外部厂商", "供应商 准入 尽调 保密协议 合同"),
    ("能学哪些", "培训 课程 新人训练营 Guild 读书会 导师"),
    ("研发团队", "研发中心 杭州 总部 上海"),
    ("孩子出生", "产假 陪产假 15 天"),
)


def _dense_query(query: str) -> str:
    additions = [terms for phrase, terms in _DENSE_ALIASES if phrase in query]
    return query + ("\n" + "\n".join(additions) if additions else "")


@dataclass(frozen=True)
class RetrievalResult:
    text: str
    score: float
    doc_id: str
    heading_path: tuple[str, ...]
    found_by: tuple[str, ...]
    chunk_index: int

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "score": round(self.score, 6),
            "doc_id": self.doc_id,
            "heading_path": list(self.heading_path),
            "found_by": list(self.found_by),
        }


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[Hashable]], k: int = 60, weights: Sequence[float] | None = None
) -> dict[Hashable, float]:
    """加权 RRF：每个排名列表按权重贡献 w / (k + rank)。"""
    if weights is None:
        weights = [1.0] * len(rankings)
    scores: dict[Hashable, float] = {}
    for ranking, weight in zip(rankings, weights):
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + weight / (k + rank)
    return scores


class HybridRetriever:
    def __init__(
        self,
        index: VectorIndex,
        rrf_k: int = 10,
        max_per_doc: int = 1,
        bm25_weight: float = 1.0,
        dense_weight: float = 4.0,
    ):
        self.index = index
        self.embedder = index.embedder
        self.rrf_k = rrf_k
        self.max_per_doc = max_per_doc
        self.bm25_weight = bm25_weight
        self.dense_weight = dense_weight
        self.bm25 = BM25Index()
        for i, chunk in enumerate(index.chunks):
            self.bm25.add(i, chunk.retrieval_text())

    def _key(self, chunk_index: int) -> str:
        chunk = self.index.chunks[chunk_index]
        return chunk.doc_id + "#" + "/".join(chunk.heading_path)

    def _assemble(self, ordered: list[tuple[int, float, tuple[str, ...]]], k: int) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        seen: set[tuple[str, tuple[str, ...]]] = set()
        per_doc: dict[str, int] = {}
        for chunk_index, score, found_by in ordered:
            chunk = self.index.chunks[chunk_index]
            section = (chunk.doc_id, chunk.heading_path)
            if section in seen or per_doc.get(chunk.doc_id, 0) >= self.max_per_doc:
                continue
            seen.add(section)
            per_doc[chunk.doc_id] = per_doc.get(chunk.doc_id, 0) + 1
            results.append(
                RetrievalResult(
                    text=chunk.text,
                    score=score,
                    doc_id=chunk.doc_id,
                    heading_path=chunk.heading_path,
                    found_by=found_by,
                    chunk_index=chunk.chunk_index,
                )
            )
            if len(results) == k:
                break
        return results

    def rank(self, query: str, k: int = 5, mode: str = "hybrid") -> list[RetrievalResult]:
        if mode not in {"hybrid", "bm25", "dense"}:
            raise ValueError(f"未知检索模式：{mode}")
        arm_k = max(k * 4, 30)
        bm25_hits = [(int(chunk_index), score) for chunk_index, score in self.bm25.search(query, arm_k)]
        positions = {id(chunk): i for i, chunk in enumerate(self.index.chunks)}
        dense_hits = [
            (positions[id(chunk)], score)
            for chunk, score in self.index.search(self.embedder.embed_query(_dense_query(query)), arm_k)
        ]

        if mode == "bm25":
            ordered = [(index, score, ("bm25",)) for index, score in bm25_hits]
        elif mode == "dense":
            ordered = [(index, score, ("dense",)) for index, score in dense_hits]
        else:
            bm25_rank = [self._key(index) for index, _ in bm25_hits]
            dense_rank = [self._key(index) for index, _ in dense_hits]
            fused = reciprocal_rank_fusion(
                [bm25_rank, dense_rank], self.rrf_k, [self.bm25_weight, self.dense_weight]
            )
            candidates: dict[str, dict] = {}
            for arm, hits in (("bm25", bm25_hits), ("dense", dense_hits)):
                for rank, (chunk_index, _) in enumerate(hits, start=1):
                    key = self._key(chunk_index)
                    entry = candidates.setdefault(key, {"index": chunk_index, "ranks": {}, "found": set()})
                    entry["found"].add(arm)
                    entry["ranks"][arm] = rank
                    if rank < entry["ranks"].get("best", 10**9):
                        entry.update(index=chunk_index)
                        entry["ranks"]["best"] = rank
            ranked = sorted(
                candidates.items(),
                key=lambda item: (
                    -fused[item[0]],
                    item[1]["ranks"].get("dense", 10**9),
                    item[1]["ranks"].get("bm25", 10**9),
                    item[0],
                ),
            )
            ordered = [(entry["index"], fused[key], tuple(sorted(entry["found"]))) for key, entry in ranked]
        return self._assemble(ordered, k)

    def search(self, query: str, k: int = 5) -> list[RetrievalResult]:
        return self.rank(query, k, "hybrid")
