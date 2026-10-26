"""LLM 抽象、OpenAI 兼容客户端与离线 MockModel。"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-chat"


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class ChatResponse:
    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)


class ChatModel(Protocol):
    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse: ...


class OpenAICompatModel:
    """OpenAI chat-completions 兼容客户端；密钥只从环境变量读取。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: int = 60,
    ):
        self.base_url = (base_url or os.getenv("KB_LLM_BASE_URL", DEFAULT_BASE_URL)).rstrip("/")
        self.api_key = api_key or os.getenv("KB_LLM_API_KEY", "")
        self.model = model or os.getenv("KB_LLM_MODEL", DEFAULT_MODEL)
        self.timeout = timeout

    @classmethod
    def from_env(cls) -> "OpenAICompatModel":
        if not os.getenv("KB_LLM_API_KEY"):
            raise RuntimeError("缺少 KB_LLM_API_KEY，无法调用远程模型。")
        return cls()

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        if not self.api_key:
            raise RuntimeError("缺少 KB_LLM_API_KEY，无法调用远程模型。")
        payload: dict[str, Any] = {"model": self.model, "messages": messages, "temperature": 0}
        if tools:
            payload["tools"] = tools
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:300]
            raise RuntimeError(f"LLM 请求失败：HTTP {exc.code} {detail}") from None
        except urllib.error.URLError as exc:
            raise RuntimeError(f"LLM 网络错误：{exc.reason}") from None

        message = data["choices"][0]["message"]
        calls: list[ToolCall] = []
        for call in message.get("tool_calls") or []:
            function = call["function"]
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                raise RuntimeError("模型返回了无法解析的工具参数 JSON。") from None
            calls.append(ToolCall(call.get("id", ""), function["name"], arguments))
        return ChatResponse(message.get("content") or "", calls)


_MATH_HINT = re.compile(r"[-+*/%()\d\s.]{5,}")


class MockModel:
    """离线、确定性的模型替身：可预置脚本，也可按规则选择检索工具。"""

    def __init__(
        self,
        script: list[ChatResponse] | None = None,
        planner: Callable[[list[dict], set[str]], ChatResponse | None] | None = None,
        max_tool_steps: int = 3,
    ):
        self.script = list(script or [])
        self.planner = planner
        self.max_tool_steps = max_tool_steps
        self.calls = 0

    def complete(self, messages: list[dict], tools: list[dict] | None = None) -> ChatResponse:
        self.calls += 1
        if self.script:
            return self.script.pop(0)
        available = {item["function"]["name"] for item in (tools or [])}
        tool_steps = sum(message.get("role") == "tool" for message in messages)
        if self.planner and tool_steps < self.max_tool_steps:
            planned = self.planner(messages, available)
            if planned is not None:
                return planned
        if tool_steps:
            return ChatResponse(self._final_from_tools(messages))
        question = next((m["content"] for m in reversed(messages) if m.get("role") == "user"), "")
        if "calculator" in available:
            match = _MATH_HINT.search(question)
            if match and any(op in match.group() for op in "+-*/%"):
                return ChatResponse(
                    tool_calls=[ToolCall("mock_calc", "calculator", {"expression": match.group().strip()})]
                )
        if "kb_search" in available:
            return ChatResponse(tool_calls=[ToolCall("mock_search", "kb_search", {"query": question, "k": 5})])
        return ChatResponse("离线 MockModel 没有可用工具，无法回答该问题。")

    def _final_from_tools(self, messages: list[dict]) -> str:
        last = next(message for message in reversed(messages) if message.get("role") == "tool")
        try:
            payload = json.loads(last.get("content") or "{}")
        except json.JSONDecodeError:
            return "工具返回了非 JSON 内容：" + str(last.get("content"))[:500]
        if "error" in payload:
            return "工具调用失败：" + str(payload["error"])
        if "results" in payload:
            lines = ["（离线 MockModel：以下内容直接摘自检索结果）"]
            for item in payload["results"][:2]:
                heading = " > ".join(item.get("heading_path", []))
                snippet = item.get("text", "").replace("\n", " ")[:420]
                lines.append(f"[{item.get('doc_id')} / {heading}] {snippet}")
            return "\n".join(lines)
        if "columns" in payload:
            lines = ["查询结果：", "列：" + " | ".join(payload.get("columns", []))]
            for row in payload.get("rows", []):
                lines.append(" | ".join(str(value) for value in row))
            if payload.get("truncated"):
                lines.append("（结果超过 100 行，已截断）")
            return "\n".join(lines)
        if "result" in payload:
            value = payload["result"]
            shown = f"{value:.2f}" if isinstance(value, float) else str(value)
            return f"计算结果：{shown}（表达式：{payload.get('expression', '')}）"
        return "工具执行完成：" + json.dumps(payload, ensure_ascii=False)
