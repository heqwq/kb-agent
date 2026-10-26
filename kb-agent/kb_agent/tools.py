"""Agent 工具注册表：知识库检索、只读 SQL、受限计算器。"""

from __future__ import annotations

import ast
import math
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

from .retriever import HybridRetriever


class ToolError(Exception):
    pass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict
    func: Callable[..., Any]


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schemas(self) -> list[dict]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def names(self) -> list[str]:
        return list(self._tools)

    def subset(self, names: list[str]) -> "ToolRegistry":
        registry = ToolRegistry()
        for name in names:
            if name not in self._tools:
                raise ToolError(f"未知工具：{name}")
            registry.register(self._tools[name])
        return registry

    def call(self, name: str, arguments: dict) -> Any:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolError(f"未知工具：{name}")
        if not isinstance(arguments, dict):
            raise ToolError("工具参数必须是 JSON 对象")
        try:
            return tool.func(**arguments)
        except ToolError:
            raise
        except TypeError as exc:
            raise ToolError(f"工具参数错误：{exc}") from None


def _kb_search_tool(retriever: HybridRetriever) -> Tool:
    def kb_search(query: str, k: int = 5) -> dict:
        if not isinstance(query, str) or not query.strip():
            raise ToolError("query 不能为空")
        k = max(1, min(int(k), 10))
        results = retriever.search(query.strip(), k)
        return {
            "query": query,
            "results": [
                {
                    "text": item.text,
                    "doc_id": item.doc_id,
                    "heading_path": list(item.heading_path),
                    "score": round(item.score, 6),
                    "found_by": list(item.found_by),
                }
                for item in results
            ],
        }

    return Tool(
        name="kb_search",
        description="检索星岚科技内部知识库，返回相关制度原文片段与来源。",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "检索问题或关键词"},
                "k": {"type": "integer", "description": "返回条数，1-10，默认 5"},
            },
            "required": ["query"],
        },
        func=kb_search,
    )


def _sql_tool(db_path: str | Path) -> Tool:
    path = Path(db_path).resolve()

    def sql_query(sql: str) -> dict:
        if not isinstance(sql, str):
            raise ToolError("sql 必须是字符串")
        statement = sql.strip()
        if statement.endswith(";"):
            statement = statement[:-1].strip()
        if ";" in statement:
            raise ToolError("只允许执行单条 SQL 语句")
        if not re.match(r"(?is)^(select|pragma)\b", statement):
            raise ToolError("只允许 SELECT 或 PRAGMA 只读查询")
        if "__" in statement:
            raise ToolError("SQL 中不允许出现双下划线标识符")
        if not path.is_file():
            raise ToolError(f"数据库不存在：{path}")
        uri = "file:" + quote(path.as_posix(), safe="/:") + "?mode=ro"
        try:
            connection = sqlite3.connect(uri, uri=True, timeout=5)
            connection.row_factory = sqlite3.Row
            cursor = connection.execute(statement)
            rows = cursor.fetchmany(101)
            columns = [item[0] for item in cursor.description] if cursor.description else []
            truncated = len(rows) > 100
            rows = rows[:100]
            return {
                "columns": columns,
                "rows": [[row[column] for column in columns] for row in rows],
                "truncated": truncated,
            }
        except sqlite3.Error as exc:
            raise ToolError(f"SQL 执行失败：{exc}") from None
        finally:
            if "connection" in locals():
                connection.close()

    return Tool(
        name="sql_query",
        description="对只读分析库执行单条 SELECT/PRAGMA 查询，用于订单、客户等结构化统计。",
        parameters={
            "type": "object",
            "properties": {"sql": {"type": "string", "description": "单条只读 SQL"}},
            "required": ["sql"],
        },
        func=sql_query,
    )


_BIN_OPS = {
    ast.Add: lambda a, b: a + b,
    ast.Sub: lambda a, b: a - b,
    ast.Mult: lambda a, b: a * b,
    ast.Div: lambda a, b: a / b,
    ast.FloorDiv: lambda a, b: a // b,
    ast.Mod: lambda a, b: a % b,
    ast.Pow: lambda a, b: a**b,
}


def _eval_ast(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ToolError("只允许数字常量")
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_ast(node.left), _eval_ast(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 100:
            raise ToolError("指数绝对值的上限为 100")
        try:
            return _BIN_OPS[type(node.op)](left, right)
        except (ZeroDivisionError, OverflowError, ValueError):
            raise ToolError("计算失败：除零或数值溢出") from None
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _eval_ast(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    raise ToolError("表达式只允许数字、括号与 + - * / // % ** 运算")


def _calculator_tool() -> Tool:
    def calculator(expression: str) -> dict:
        if not isinstance(expression, str) or not expression.strip():
            raise ToolError("expression 不能为空")
        if "__" in expression or len(expression) > 200:
            raise ToolError("表达式含非法内容或过长")
        try:
            tree = ast.parse(expression, mode="eval")
        except SyntaxError:
            raise ToolError("表达式语法错误") from None
        result = _eval_ast(tree.body)
        if isinstance(result, float) and not math.isfinite(result):
            raise ToolError("计算结果不是有限数")
        return {"expression": expression, "result": result}

    return Tool(
        name="calculator",
        description="计算纯算术表达式，支持 + - * / // % ** 与括号，不接受变量或函数调用。",
        parameters={
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "算术表达式"}},
            "required": ["expression"],
        },
        func=calculator,
    )


def build_default_tools(retriever: HybridRetriever, db_path: str | Path) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(_kb_search_tool(retriever))
    registry.register(_sql_tool(db_path))
    registry.register(_calculator_tool())
    return registry
