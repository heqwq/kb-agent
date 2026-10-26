"""命令行入口：ingest / ask / eval。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from .agent import Agent
from .embeddings import HashingEmbedder, OpenAICompatibleEmbedder
from .index import VectorIndex
from .llm import MockModel, OpenAICompatModel
from .retriever import HybridRetriever
from .tools import build_default_tools

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "corpus"
DEFAULT_INDEX = ROOT / "data" / "index"
DEFAULT_DB = ROOT / "data" / "analytics.db"
DEFAULT_REPORTS = ROOT / "reports"


def _embedder(kind: str):
    if kind == "hash":
        return HashingEmbedder()
    if kind == "openai":
        return OpenAICompatibleEmbedder.from_env()
    if os.getenv("KB_EMBED_BASE_URL") and os.getenv("KB_EMBED_API_KEY"):
        return OpenAICompatibleEmbedder.from_env()
    return HashingEmbedder()


def cmd_ingest(args: argparse.Namespace) -> None:
    index = VectorIndex.build_from_corpus(args.corpus, _embedder(args.embedder))
    index.save(args.out)
    docs = len({chunk.doc_id for chunk in index.chunks})
    print(f"索引完成：{docs} 篇文档，{len(index.chunks)} 个分块，写入 {args.out}")


def cmd_ask(args: argparse.Namespace) -> None:
    index = VectorIndex.load(args.index)
    retriever = HybridRetriever(index)
    tools = build_default_tools(retriever, args.db)
    if os.getenv("KB_LLM_API_KEY"):
        model = OpenAICompatModel.from_env()
    else:
        model = MockModel()
    result = Agent(model, tools, max_steps=args.max_steps).run(args.question)
    print(result.answer)
    tools_used = ", ".join(f"{step.tool}({step.latency_ms:.1f}ms)" for step in result.trace.steps)
    print(f"\n[停止原因={result.stopped_reason} 轮次={result.trace.turns} 工具={tools_used or '无'}]")
    if args.trace:
        print(result.trace.to_json())


def cmd_eval(args: argparse.Namespace) -> None:
    from evals.run_eval import run_evaluation

    summary = run_evaluation(limit=args.limit, report_dir=args.reports)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"报告已写入：{Path(args.reports) / 'eval-report.md'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kb-agent", description="星岚企业知识库问答 Agent")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="切分语料并构建向量索引")
    ingest.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ingest.add_argument("--out", type=Path, default=DEFAULT_INDEX)
    ingest.add_argument("--embedder", choices=["auto", "hash", "openai"], default="auto")
    ingest.set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="向 Agent 提问")
    ask.add_argument("question")
    ask.add_argument("--index", type=Path, default=DEFAULT_INDEX)
    ask.add_argument("--db", type=Path, default=DEFAULT_DB)
    ask.add_argument("--max-steps", type=int, default=6)
    ask.add_argument("--trace", action="store_true", help="打印完整 JSON 轨迹")
    ask.set_defaults(func=cmd_ask)

    evaluate = sub.add_parser("eval", help="运行检索与 Agent 评估")
    evaluate.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    evaluate.add_argument("--reports", type=Path, default=DEFAULT_REPORTS)
    evaluate.set_defaults(func=cmd_eval)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
