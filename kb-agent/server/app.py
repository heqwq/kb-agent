"""可选 FastAPI 服务：仅在安装 fastapi/uvicorn 后使用。"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_APP = None


def create_app():
    try:
        from fastapi import FastAPI
        from pydantic import BaseModel
    except ImportError as exc:
        print("缺少 FastAPI。请先运行：pip install -r requirements-optional.txt")
        raise exc

    from kb_agent.agent import Agent
    from kb_agent.index import VectorIndex
    from kb_agent.llm import MockModel, OpenAICompatModel
    from kb_agent.retriever import HybridRetriever
    from kb_agent.tools import build_default_tools

    index_dir = Path(os.getenv("KB_INDEX_DIR", ROOT / "data" / "index"))
    db_path = Path(os.getenv("KB_DB_PATH", ROOT / "data" / "analytics.db"))
    index = VectorIndex.load(index_dir)
    retriever = HybridRetriever(index)
    tools = build_default_tools(retriever, db_path)
    model = OpenAICompatModel.from_env() if os.getenv("KB_LLM_API_KEY") else MockModel()

    app = FastAPI(title="星岚企业知识库 Agent", version="0.1.0")

    class AskRequest(BaseModel):
        question: str
        max_steps: int = 6

    @app.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "documents": len({chunk.doc_id for chunk in index.chunks}),
            "chunks": len(index.chunks),
            "model": type(model).__name__,
        }

    @app.post("/ask")
    def ask(request: AskRequest) -> dict:
        result = Agent(model, tools, max_steps=max(1, min(request.max_steps, 10))).run(request.question)
        return {
            "answer": result.answer,
            "stopped_reason": result.stopped_reason,
            "trace": result.trace.to_dict(),
        }

    return app


def __getattr__(name: str):
    global _APP
    if name == "app":
        if _APP is None:
            _APP = create_app()
        return _APP
    raise AttributeError(name)


if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        print("缺少 uvicorn。请先运行：pip install -r requirements-optional.txt")
        raise SystemExit(1)
    uvicorn.run(__getattr__("app"), host="127.0.0.1", port=int(os.getenv("KB_API_PORT", "8000")))
