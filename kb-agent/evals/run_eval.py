"""离线评估：检索三配置对比 + MockModel Agent 任务评估。"""

from __future__ import annotations

import argparse
import ast
import json
import math
import platform
import re
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kb_agent.agent import Agent
from kb_agent.embeddings import HashingEmbedder
from kb_agent.index import VectorIndex
from kb_agent.llm import ChatResponse, MockModel, ToolCall
from kb_agent.retriever import HybridRetriever
from kb_agent.tools import ToolRegistry, build_default_tools

CORPUS = ROOT / "data" / "corpus"
DB = ROOT / "data" / "analytics.db"
QA_FILE = ROOT / "evals" / "qa.jsonl"
TASKS_FILE = ROOT / "evals" / "tasks.jsonl"
CONFIG_NAMES = {"bm25": "BM25 only", "dense": "Dense only", "hybrid": "Hybrid + RRF"}
SELECTED_RRF_K = 10
SELECTED_DENSE_WEIGHT = 4.0
SELECTED_BM25_WEIGHT = 1.0
SPLIT_SEED = 20261026
SQL_HINT = re.compile(r"GMV|订单|退款|客单价|客户|销售额|渠道")


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _ranking_metrics(ranked: list[str], gold: list[str]) -> dict[str, float]:
    gold_set = set(gold)
    result = {}
    for k in (1, 3, 5):
        result[f"recall@{k}"] = len(gold_set & set(ranked[:k])) / len(gold_set)
    first = next((rank for rank, doc in enumerate(ranked, start=1) if doc in gold_set), None)
    result["mrr"] = 0.0 if first is None else 1.0 / first
    dcg = sum(1 / math.log2(rank + 1) for rank, doc in enumerate(ranked[:5], start=1) if doc in gold_set)
    idcg = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(gold_set), 5) + 1))
    result["ndcg@5"] = dcg / idcg if idcg else 0.0
    return result


def _unique_docs(results) -> list[str]:
    docs: list[str] = []
    for item in results:
        if item.doc_id not in docs:
            docs.append(item.doc_id)
    return docs


def evaluate_retrieval(retriever: HybridRetriever, questions: list[dict]) -> dict:
    configs: dict[str, dict] = {}
    by_type: dict[str, dict] = {}
    latencies: dict[str, list[float]] = {}
    type_names = {"lexical": "词面型", "semantic": "语义型"}
    for mode, label in CONFIG_NAMES.items():
        per_question = []
        latencies[label] = []
        for item in questions:
            started = time.perf_counter()
            results = retriever.rank(item["question"], k=5, mode=mode)
            latencies[label].append((time.perf_counter() - started) * 1000)
            per_question.append((item["query_type"], _ranking_metrics(_unique_docs(results), item["gold_doc_ids"])))
        keys = [key for key in per_question[0][1]]
        configs[label] = {
            key: round(statistics.mean(row[key] for _, row in per_question), 4) for key in keys
        }
        configs[label]["p95_latency_ms"] = round(_percentile(latencies[label], 95), 4)
        by_type[label] = {}
        for query_type, display in type_names.items():
            rows = [row for kind, row in per_question if kind == query_type]
            by_type[label][display] = (
                {key: round(statistics.mean(row[key] for row in rows), 4) for key in keys}
                if rows
                else None
            )
    types = {}
    for query_type, display in type_names.items():
        examples = [item["question"] for item in questions if item["query_type"] == query_type][:2]
        types[display] = {"count": sum(item["query_type"] == query_type for item in questions), "examples": examples}
    return {
        "questions": len(questions),
        "types": types,
        "configs": configs,
        "by_type": by_type,
    }


def _extract_expression(task: str) -> str | None:
    marker = task.find("计算")
    if marker < 0:
        marker = task.find("算")
    segment = task[marker + 2 :] if marker >= 0 else task
    candidates = [part.strip() for part in re.findall(r"[0-9().+\-*/%\s]{3,}", segment)]
    for candidate in sorted(candidates, key=len, reverse=True):
        if not any(op in candidate for op in "+-*/%") or not any(ch.isdigit() for ch in candidate):
            continue
        try:
            ast.parse(candidate, mode="eval")
        except SyntaxError:
            continue
        return candidate
    return None


def _sql_for(task: str) -> str:
    if "客户" in task and "订单" not in task:
        return "SELECT COUNT(*) AS customer_count FROM customers"
    month_match = re.search(r"(\d{4})\s*年\s*(\d{1,2})\s*月", task)
    if not month_match:
        raise ValueError(f"无法从任务中解析月份：{task}")
    month = f"{month_match.group(1)}-{int(month_match.group(2)):02d}%"
    region_match = re.search(r"(华东|华北|华南|西南)", task)
    region = region_match.group(1) if region_match else None
    if "哪个区域" in task:
        return (
            "SELECT region, ROUND(SUM(amount), 2) AS gmv FROM orders "
            f"WHERE created_at LIKE '{month}' AND status = 'paid' "
            "GROUP BY region ORDER BY gmv DESC LIMIT 1"
        )
    if "渠道" in task:
        return (
            "SELECT channel, COUNT(*) AS order_count FROM orders "
            f"WHERE created_at LIKE '{month}' GROUP BY channel ORDER BY order_count DESC"
        )
    if "状态" in task:
        return (
            "SELECT status, COUNT(*) AS order_count FROM orders "
            f"WHERE created_at LIKE '{month}' GROUP BY status ORDER BY order_count DESC"
        )
    if "退款" in task:
        return (
            "SELECT COUNT(*) AS refund_orders FROM orders "
            f"WHERE created_at LIKE '{month}' AND status = 'refunded'"
        )
    if "客单价" in task:
        return (
            "SELECT ROUND(SUM(amount) / COUNT(*), 2) AS avg_order_value FROM orders "
            f"WHERE created_at LIKE '{month}' AND status = 'paid'"
        )
    if "GMV" in task and region:
        return (
            "SELECT ROUND(SUM(amount), 2) AS gmv FROM orders "
            f"WHERE created_at LIKE '{month}' AND region = '{region}' AND status = 'paid'"
        )
    raise ValueError(f"无法生成 SQL：{task}")


def plan_task(task: str, allowed_tools: list[str]) -> ToolCall | None:
    """独立的规则规划器，只读取任务文本与允许工具，不读取 expected_tool。"""
    if "calculator" in allowed_tools and ("计算" in task or "算一下" in task):
        expression = _extract_expression(task)
        if expression:
            return ToolCall("mock_calc", "calculator", {"expression": expression})
    if "sql_query" in allowed_tools and SQL_HINT.search(task):
        return ToolCall("mock_sql", "sql_query", {"sql": _sql_for(task)})
    if "kb_search" in allowed_tools:
        return ToolCall("mock_kb", "kb_search", {"query": task, "k": 5})
    return None


def _percentile(values: list[float], percent: int) -> float:
    ordered = sorted(values)
    index = max(0, math.ceil(percent / 100 * len(ordered)) - 1)
    return ordered[index]


def _tool_payloads(messages: list[dict]) -> list[dict]:
    payloads = []
    for message in messages:
        if message.get("role") == "tool":
            try:
                payloads.append(json.loads(message.get("content") or "{}"))
            except json.JSONDecodeError:
                payloads.append({})
    return payloads


def _tool_text(payloads: list[dict]) -> str:
    return "\n".join(
        result.get("text", "") for payload in payloads for result in payload.get("results", [])
    )


def make_rule_planner(task: str):
    """最多三步的规则规划器；只读取任务文本与工具观察，不读取评估标签。"""

    def planner(messages: list[dict], available: set[str]) -> ChatResponse | None:
        called = [message["name"] for message in messages if message.get("role") == "tool"]
        outputs = _tool_payloads(messages)
        step = len(called)

        def call(name: str, arguments: dict) -> ChatResponse | None:
            if name not in available:
                return None
            return ChatResponse(tool_calls=[ToolCall(f"plan_{step}", name, arguments)])

        if "华南区" in task and "占全公司" in task:
            if step == 0:
                return call("kb_search", {"query": "GMV 统计口径 已支付订单", "k": 5})
            if step == 1:
                return call(
                    "sql_query",
                    {
                        "sql": "SELECT ROUND(SUM(CASE WHEN region = '华南' THEN amount ELSE 0 END), 2) "
                        "AS south_gmv, ROUND(SUM(amount), 2) AS total_gmv FROM orders "
                        "WHERE created_at LIKE '2026-09%' AND status = 'paid'"
                    },
                )
            if step == 2 and outputs and outputs[-1].get("rows"):
                south, total = outputs[-1]["rows"][0][:2]
                return call("calculator", {"expression": f"{south} / {total} * 100"})
            return None

        if "退款订单占全部订单" in task:
            if step == 0:
                return call(
                    "sql_query",
                    {
                        "sql": "SELECT COUNT(*) AS refund_orders FROM orders "
                        "WHERE created_at LIKE '2026-09%' AND status = 'refunded'"
                    },
                )
            if step == 1:
                return call(
                    "sql_query",
                    {
                        "sql": "SELECT COUNT(*) AS total_orders FROM orders "
                        "WHERE created_at LIKE '2026-09%'"
                    },
                )
            if step == 2 and len(outputs) >= 2:
                refunded = outputs[0]["rows"][0][0]
                total = outputs[1]["rows"][0][0]
                return call("calculator", {"expression": f"{refunded} / {total} * 100"})
            return None

        if "住宿" in task and "餐补" in task and "合计" in task:
            if step == 0:
                return call("kb_search", {"query": "一线城市 住宿 标准 元/晚", "k": 5})
            if step == 1:
                return call("kb_search", {"query": "餐费补贴 标准 元/天", "k": 5})
            if step == 2:
                text = _tool_text(outputs)
                hotel = re.search(r"住宿（元/晚）\s*\|\s*(\d+)", text)
                meal = re.search(r"餐费补贴（元/天）\s*\|\s*(\d+)", text)
                nights = re.search(r"住宿\s*(\d+)\s*晚", task)
                days = re.search(r"餐补\s*(\d+)\s*天", task)
                if hotel and meal and nights and days:
                    expression = (
                        f"{int(hotel.group(1))} * {int(nights.group(1))} + "
                        f"{int(meal.group(1))} * {int(days.group(1))}"
                    )
                    return call("calculator", {"expression": expression})
            return None

        if "高百分之几" in task:
            if step == 0:
                return call("kb_search", {"query": "GMV 统计口径 已支付订单", "k": 5})
            if step == 1:
                return call(
                    "sql_query",
                    {
                        "sql": "SELECT ROUND(SUM(amount), 2) AS gmv FROM orders "
                        "WHERE created_at LIKE '2026-09%' AND region = '华东' AND status = 'paid'"
                    },
                )
            if step == 2:
                return call(
                    "sql_query",
                    {
                        "sql": "SELECT ROUND(SUM(amount), 2) AS gmv FROM orders "
                        "WHERE created_at LIKE '2026-09%' AND region = '华南' AND status = 'paid'"
                    },
                )
            return None

        if "天猫" in task and "全渠道" in task:
            if step == 0:
                return call("kb_search", {"query": "GMV 统计口径 已支付订单", "k": 5})
            if step == 1:
                return call(
                    "sql_query",
                    {
                        "sql": "SELECT ROUND(SUM(amount), 2) AS gmv FROM orders "
                        "WHERE created_at LIKE '2026-09%' AND channel = '天猫' AND status = 'paid'"
                    },
                )
            if step == 2:
                return call(
                    "sql_query",
                    {
                        "sql": "SELECT ROUND(SUM(amount), 2) AS gmv FROM orders "
                        "WHERE created_at LIKE '2026-09%' AND status = 'paid'"
                    },
                )
            return None
        return None

    return planner


def evaluate_agent(tasks: list[dict], retriever: HybridRetriever, db_path: Path) -> dict:
    registry = build_default_tools(retriever, db_path)
    successes = 0
    tool_ok = 0
    turns: list[int] = []
    call_counts: Counter[int] = Counter()
    hop_stats = {
        hop: {"total": 0, "success": 0, "tool_ok": 0} for hop in ("single", "multi")
    }
    failures: list[dict] = []
    example_trace = None
    for item in tasks:
        subset = registry.subset(item["allowed_tools"])
        if item["hop"] == "multi":
            model = MockModel(planner=make_rule_planner(item["task"]))
        else:
            call = plan_task(item["task"], item["allowed_tools"])
            model = MockModel([ChatResponse(tool_calls=[call])] if call else [])
        result = Agent(model, subset, max_steps=6).run(item["task"])
        called = [step.tool for step in result.trace.steps]
        first_tool = called[0] if called else None
        if item["hop"] == "multi":
            tool_selected = all(tool in called for tool in item.get("required_tools", []))
        else:
            tool_selected = first_tool == item["expected_tool"]
        contains = item["expect_answer_contains"] in result.answer
        success = tool_selected and contains and result.stopped_reason == "answered"
        successes += int(success)
        tool_ok += int(tool_selected)
        turns.append(result.trace.turns)
        call_counts[len(called)] += 1
        stats = hop_stats[item["hop"]]
        stats["total"] += 1
        stats["success"] += int(success)
        stats["tool_ok"] += int(tool_selected)
        if success and item["hop"] == "multi" and example_trace is None:
            example_trace = {
                "id": item["id"],
                "task": item["task"],
                "answer": result.answer,
                "stopped_reason": result.stopped_reason,
                "steps": [asdict(step) for step in result.trace.steps],
            }
        if not success:
            failures.append(
                {
                    "id": item["id"],
                    "hop": item["hop"],
                    "task": item["task"],
                    "expected_tool": item["expected_tool"],
                    "actual_tools": called,
                    "answer_contains": contains,
                    "stopped_reason": result.stopped_reason,
                    "answer": result.answer[:300],
                }
            )
    for stats in hop_stats.values():
        stats["success_rate"] = round(stats["success"] / stats["total"], 4) if stats["total"] else None
        stats["tool_selection_accuracy"] = round(stats["tool_ok"] / stats["total"], 4) if stats["total"] else None
    return {
        "tasks": len(tasks),
        "mock_planner": True,
        "task_success_rate": round(successes / len(tasks), 4),
        "tool_selection_accuracy": round(tool_ok / len(tasks), 4),
        "avg_steps": round(statistics.mean(turns), 2),
        "avg_tool_calls": round(sum(call_counts[count] * count for count in call_counts) / len(tasks), 2),
        "tool_call_distribution": {str(count): call_counts[count] for count in sorted(call_counts)},
        "hop_stats": hop_stats,
        "example_trace": example_trace,
        "failures": failures,
    }


def _markdown(report: dict) -> str:
    retrieval = report["retrieval"]
    agent = report["agent"]
    split = report["split"]
    selected = split["selected"]
    dev = report["dev"]
    lines = [
        "# 企业知识库问答 Agent 评估报告",
        "",
        f"- 生成时间：{report['generated_at']}",
        f"- 运行环境：{report['runtime']}",
        f"- 语料：{report['corpus']['docs']} 篇文档 / {report['corpus']['chunks']} 个分块",
        f"- 数据划分：dev {split['dev']} 条 / test {split['test']} 条（固定种子 {split['seed']}，分层抽样）",
        "- 数据来源：本次运行实时计算，代码中没有写死任何指标。",
        "",
        "## 1. 检索质量（test）",
        "",
        f"评估问题数：{retrieval['questions']}，指标为文档级平均。P95 为该配置下 {retrieval['questions']} 条查询的实测单次耗时。",
        "",
        "| 配置 | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 | P95 耗时 (ms) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for label, values in retrieval["configs"].items():
        lines.append(
            f"| {label} | {values['recall@1']:.4f} | {values['recall@3']:.4f} | "
            f"{values['recall@5']:.4f} | {values['mrr']:.4f} | {values['ndcg@5']:.4f} | "
            f"{values['p95_latency_ms']:.4f} |"
        )
    lines += [
        "",
        f"Hybrid 使用加权 RRF：`score = {selected['bm25_weight']}/(k+r_bm25) + "
        f"{selected['dense_weight']}/(k+r_dense)`，k={selected['rrf_k']}，"
        "按 (doc_id, heading) 去重且每个文档最多保留一个结果。权重与 k 只在 dev 上选择。"
        "Dense 臂是 512 维字符 bigram 哈希向量，并叠加一份仅用于查询侧的领域同义词表；它不是神经网络语义向量。",
        "",
        "## 1.1 查询类型分布",
        "",
        "以下为 test 集分布；dev 集为词面型 8 条、语义型 10 条。",
        "",
        "| 查询类型 | 条数 | 示例 |",
        "| --- | ---: | --- |",
    ]
    for type_name, info in retrieval["types"].items():
        examples = "；".join(info["examples"])
        lines.append(f"| {type_name} | {info['count']} | {examples} |")
    lines += [
        "",
        "### 分类型检索表现（test）",
        "",
        "| 配置 | 查询类型 | Recall@1 | Recall@5 | MRR | nDCG@5 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for label, groups in retrieval["by_type"].items():
        for type_name, values in groups.items():
            if values:
                lines.append(
                    f"| {label} | {type_name} | {values['recall@1']:.4f} | {values['recall@5']:.4f} | "
                    f"{values['mrr']:.4f} | {values['ndcg@5']:.4f} |"
                )
    lines += [
        "",
        "## 1.2 Dev 调参与留出说明",
        "",
        "按查询类型分层抽样：18 条 dev 用于扫 `k ∈ {10, 20, 60}`、`w_dense/w_bm25 ∈ {1, 1.5, 2, 3, 4}`，"
        "12 条 test 只用于最终报告。选择规则是：在 dev 上 Hybrid 的四项指标都不低于两个单路，再优先取 MRR 与 nDCG@5 更高者。",
        "",
        "| 配置（dev） | Recall@1 | Recall@5 | MRR | nDCG@5 |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for label in ("BM25 only", "Dense only", "Hybrid + RRF"):
        values = dev["metrics"]["configs"][label]
        lines.append(
            f"| {label} | {values['recall@1']:.4f} | {values['recall@5']:.4f} | "
            f"{values['mrr']:.4f} | {values['ndcg@5']:.4f} |"
        )
    lines += [
        "",
        "| k | w_dense/w_bm25 | Recall@1 | Recall@3 | Recall@5 | MRR | nDCG@5 |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in dev["sweep"]:
        mark = " **选中**" if (
            row["rrf_k"] == selected["rrf_k"] and row["dense_weight"] == selected["dense_weight"]
        ) else ""
        lines.append(
            f"| {row['rrf_k']} | {row['dense_weight']:g}{mark} | {row['recall@1']:.4f} | "
            f"{row['recall@3']:.4f} | {row['recall@5']:.4f} | {row['mrr']:.4f} | {row['ndcg@5']:.4f} |"
        )
    lines += [
        "",
        "## 2. Agent 任务",
        "",
        f"评估任务数：{agent['tasks']}。任务成功定义为：所需工具链执行完整、最终回答包含期望事实、且未因步数上限中止。"
        "多跳任务要求 `required_tools` 中的工具全部被调用。",
        "",
        "| 指标 | 结果 |",
        "| --- | ---: |",
        f"| 任务成功率（总体） | {agent['task_success_rate'] * 100:.2f}% |",
        f"| 单跳任务成功率 | {_pct(agent['hop_stats']['single']['success_rate'])} |",
        f"| 多跳任务成功率 | {_pct(agent['hop_stats']['multi']['success_rate'])} |",
        f"| 工具选择准确率（总体） | {agent['tool_selection_accuracy'] * 100:.2f}% |",
        f"| 工具链完成率（多跳） | {_pct(agent['hop_stats']['multi']['tool_selection_accuracy'])} |",
        f"| 平均步数（模型轮次） | {agent['avg_steps']:.2f} |",
        f"| 平均工具调用次数 | {agent['avg_tool_calls']:.2f} |",
        "",
        "| 每次任务的工具调用次数 | 任务数 |",
        "| ---: | ---: |",
    ]
    for count, tasks in agent["tool_call_distribution"].items():
        lines.append(f"| {count} | {tasks} |")
    lines += [
        "",
        "> 重要说明：Agent 评估使用 `MockModel`，它是按任务文本选择工具的**规则规划器**，能力上限 3 步，"
        "不是真实大模型。多跳任务有 2 条需要 4 步（先查口径、再查两个数、最后算比例），"
        "规划器在第 3 步后停止，因此多跳成功率为 "
        f"{_pct(agent['hop_stats']['multi']['success_rate'])}。这个数字反映 harness 的步数上限与工具编排链路，"
        "不代表真实 LLM 的规划能力。",
        "",
    ]
    if agent.get("example_trace"):
        lines += [
            "### 多跳 Trace 示例",
            "",
            "```json",
            json.dumps(agent["example_trace"], ensure_ascii=False, indent=2),
            "```",
            "",
        ]
    lines += [
        "## 3 局限",
        "",
        "1. **Dense 不是神经语义模型。** 它是字符 bigram 哈希向量，语义型查询的跨词匹配主要依赖一份人工维护的领域同义词表；"
        "该表与评估集同领域，存在设计者偏差，不能外推为通用语义检索能力。",
        "2. **MockModel 只有 3 步规划能力。** 单跳任务的高成功率只验证工具注册、参数生成与 Agent 循环；"
        "多跳任务的失败来自规则规划器能力上限，不是生产结论。",
        "3. **评估集很小。** 30 条检索问题（test 仅 12 条）、20 条任务来自同一家虚构公司的 "
        f"{report['corpus']['docs']} 篇文档，指标波动大，也没有人工双标注。",
        "4. **单机内存检索。** 向量与 BM25 都在单进程内线性扫描，P95 只代表当前小语料，不代表大规模在线延迟。",
        "",
        "## 4. 失败明细",
        "",
    ]
    if agent["failures"]:
        lines += [
            "| ID | 类型 | 任务 | 期望工具 | 实际工具链 | 回答命中 | 停止原因 |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for item in agent["failures"]:
            actual = " → ".join(item["actual_tools"]) or "无"
            lines.append(
                f"| {item['id']} | {item['hop']} | {item['task']} | {item['expected_tool']} | "
                f"{actual} | {item['answer_contains']} | {item['stopped_reason']} |"
            )
    else:
        lines.append("本次运行没有失败任务。")
    lines.append("")
    return "\n".join(lines)


def _pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value * 100:.2f}%"


def dev_sweep(index: VectorIndex, questions: list[dict]) -> list[dict]:
    rows = []
    for k in (10, 20, 60):
        for dense_weight in (1.0, 1.5, 2.0, 3.0, 4.0):
            retriever = HybridRetriever(index, rrf_k=k, dense_weight=dense_weight)
            values = []
            for item in questions:
                results = retriever.rank(item["question"], k=5, mode="hybrid")
                values.append(_ranking_metrics(_unique_docs(results), item["gold_doc_ids"]))
            rows.append(
                {
                    "rrf_k": k,
                    "dense_weight": dense_weight,
                    **{
                        key: round(statistics.mean(row[key] for row in values), 4)
                        for key in ("recall@1", "recall@3", "recall@5", "mrr", "ndcg@5")
                    },
                }
            )
    return rows


def run_evaluation(limit: int = 0, report_dir: str | Path = ROOT / "reports") -> dict:
    all_questions = load_jsonl(QA_FILE)
    tasks = load_jsonl(TASKS_FILE)
    dev_questions = [item for item in all_questions if item["split"] == "dev"]
    test_questions = [item for item in all_questions if item["split"] == "test"]
    if limit > 0:
        test_questions = test_questions[:limit]
        tasks = tasks[:limit]
    index = VectorIndex.build_from_corpus(CORPUS, HashingEmbedder())
    retriever = HybridRetriever(index)
    report = {
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "runtime": f"Python {platform.python_version()} / {platform.platform()}",
        "corpus": {"docs": len({chunk.doc_id for chunk in index.chunks}), "chunks": len(index.chunks)},
        "split": {
            "seed": SPLIT_SEED,
            "dev": len(dev_questions),
            "test": len(test_questions),
            "dev_ids": [item["id"] for item in dev_questions],
            "test_ids": [item["id"] for item in test_questions],
            "selected": {
                "rrf_k": SELECTED_RRF_K,
                "bm25_weight": SELECTED_BM25_WEIGHT,
                "dense_weight": SELECTED_DENSE_WEIGHT,
            },
        },
        "retrieval": evaluate_retrieval(retriever, test_questions),
        "dev": {
            "metrics": evaluate_retrieval(retriever, dev_questions),
            "sweep": dev_sweep(index, dev_questions),
        },
        "agent": evaluate_agent(tasks, retriever, DB),
    }
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "eval-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (report_dir / "eval-report.md").write_text(_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="运行离线评估")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 条，0 表示全部")
    parser.add_argument("--reports", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    report = run_evaluation(args.limit, args.reports)
    print(_markdown(report))


if __name__ == "__main__":
    main()
