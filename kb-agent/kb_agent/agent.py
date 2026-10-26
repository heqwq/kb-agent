"""具备工具调用、步数上限与可序列化轨迹的 Agent 循环。"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from .llm import ChatModel
from .tools import ToolError, ToolRegistry

SYSTEM_PROMPT = (
    "你是星岚科技的企业知识库助手。回答制度问题前优先调用 kb_search，并引用文档来源；"
    "统计类问题调用 sql_query，只能使用只读 SELECT；算术问题调用 calculator。"
    "不确定时明确说不知道，不要编造。最终回答使用简体中文。"
)


@dataclass
class TraceStep:
    step: int
    tool: str
    args: dict
    ok: bool
    latency_ms: float
    error: str | None = None
    result_excerpt: str = ""


@dataclass
class Trace:
    question: str
    steps: list[TraceStep] = field(default_factory=list)
    turns: int = 0
    latency_ms: float = 0.0
    stopped_reason: str = ""
    answer: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


@dataclass
class AgentResult:
    answer: str
    stopped_reason: str
    trace: Trace


class Agent:
    def __init__(self, model: ChatModel, tools: ToolRegistry, max_steps: int = 6):
        self.model = model
        self.tools = tools
        self.max_steps = max_steps

    def run(self, question: str) -> AgentResult:
        started = time.perf_counter()
        messages: list[dict] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ]
        trace = Trace(
            question=question,
            model=type(self.model).__name__,
        )
        answer = ""
        stopped_reason = "max_steps"
        last_content = ""

        for turn in range(self.max_steps):
            response = self.model.complete(messages, self.tools.schemas())
            trace.turns = turn + 1
            last_content = response.content
            if not response.tool_calls:
                answer = response.content.strip() or "模型未返回答案。"
                stopped_reason = "answered"
                break

            messages.append(
                {
                    "role": "assistant",
                    "content": response.content or None,
                    "tool_calls": [
                        {
                            "id": call.id or f"call_{turn}_{position}",
                            "type": "function",
                            "function": {
                                "name": call.name,
                                "arguments": json.dumps(call.arguments, ensure_ascii=False),
                            },
                        }
                        for position, call in enumerate(response.tool_calls)
                    ],
                }
            )
            for position, call in enumerate(response.tool_calls):
                call_id = call.id or f"call_{turn}_{position}"
                call_started = time.perf_counter()
                ok = True
                error = None
                try:
                    output = self.tools.call(call.name, call.arguments)
                    content = json.dumps(output, ensure_ascii=False)
                except ToolError as exc:
                    ok = False
                    error = str(exc)
                    content = json.dumps({"error": error}, ensure_ascii=False)
                latency = (time.perf_counter() - call_started) * 1000
                trace.steps.append(
                    TraceStep(
                        step=len(trace.steps) + 1,
                        tool=call.name,
                        args=call.arguments,
                        ok=ok,
                        latency_ms=round(latency, 3),
                        error=error,
                        result_excerpt=content[:300],
                    )
                )
                messages.append(
                    {"role": "tool", "tool_call_id": call_id, "name": call.name, "content": content}
                )

        if stopped_reason == "max_steps":
            partial = last_content.strip()
            answer = (
                f"达到最大步数 {self.max_steps}，未能形成最终答案。"
                + (f"部分内容：{partial}" if partial else "")
            )
        trace.answer = answer
        trace.stopped_reason = stopped_reason
        trace.latency_ms = round((time.perf_counter() - started) * 1000, 3)
        return AgentResult(answer=answer, stopped_reason=stopped_reason, trace=trace)
