"""LangChain 参考适配器（可选，不参与离线测试）。

本文件展示如何把同一个 HybridRetriever 和同一组 ToolRegistry 工具接到
LangChain 的 Retriever / Tool / AgentExecutor 上。运行前需要：

    pip install langchain langchain-openai

本仓库的离线环境没有安装 LangChain，因此该适配器未在本机执行验证；
它不是核心运行时依赖，核心实现仍以 kb_agent/ 的 stdlib 版本为准。
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from kb_agent.index import VectorIndex
from kb_agent.retriever import HybridRetriever
from kb_agent.tools import build_default_tools

try:
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.documents import Document
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.tools import Tool
    from langchain_openai import ChatOpenAI
    from pydantic import ConfigDict
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit("缺少 LangChain，请运行：pip install langchain langchain-openai") from exc

ROOT = Path(__file__).resolve().parents[1]


class KbRetriever(BaseRetriever):
    """把 HybridRetriever 暴露成 LangChain Retriever。"""

    model_config = ConfigDict(arbitrary_types_allowed=True)
    hybrid: HybridRetriever

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> list[Document]:
        return [
            Document(
                page_content=item.text,
                metadata={
                    "doc_id": item.doc_id,
                    "heading_path": list(item.heading_path),
                    "score": item.score,
                    "found_by": list(item.found_by),
                },
            )
            for item in self.hybrid.search(query, k=5)
        ]


def _tools(registry):
    def run(name: str, arguments: dict) -> str:
        return json.dumps(registry.call(name, arguments), ensure_ascii=False)

    return [
        Tool(
            name="kb_search",
            description="检索企业知识库。",
            func=lambda query: run("kb_search", {"query": query, "k": 5}),
        ),
        Tool(
            name="sql_query",
            description="执行单条只读 SELECT。",
            func=lambda sql: run("sql_query", {"sql": sql}),
        ),
        Tool(
            name="calculator",
            description="计算纯算术表达式。",
            func=lambda expression: run("calculator", {"expression": expression}),
        ),
    ]


def build_agent() -> AgentExecutor:
    if not os.getenv("KB_LLM_API_KEY"):
        raise RuntimeError("缺少 KB_LLM_API_KEY，LangChain 适配器需要真实模型。")
    index = VectorIndex.load(ROOT / "data" / "index")
    hybrid = HybridRetriever(index)
    registry = build_default_tools(hybrid, ROOT / "data" / "analytics.db")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "你是企业知识库助手，必要时调用工具并引用来源。"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )
    llm = ChatOpenAI(
        model=os.getenv("KB_LLM_MODEL", "deepseek-chat"),
        base_url=os.getenv("KB_LLM_BASE_URL", "https://api.deepseek.com/v1"),
        api_key=os.getenv("KB_LLM_API_KEY"),
        temperature=0,
    )
    tools = _tools(registry)
    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, max_iterations=6, verbose=True)


if __name__ == "__main__":  # pragma: no cover
    executor = build_agent()
    print(executor.invoke({"input": "一线城市出差住宿标准是多少？"})["output"])
