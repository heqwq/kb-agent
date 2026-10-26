"""kb_agent 核心行为测试（stdlib unittest）。"""

from __future__ import annotations

import json
import shutil
import unittest
from pathlib import Path

from evals.run_eval import load_jsonl, run_evaluation
from kb_agent.agent import Agent
from kb_agent.bm25 import BM25Index, tokenize
from kb_agent.chunking import chunk_markdown
from kb_agent.embeddings import HashingEmbedder
from kb_agent.index import VectorIndex
from kb_agent.llm import ChatResponse, MockModel, ToolCall
from kb_agent.retriever import HybridRetriever, reciprocal_rank_fusion
from kb_agent.tools import ToolError, build_default_tools

ROOT = Path(__file__).resolve().parents[1]


def make_index() -> VectorIndex:
    return VectorIndex.build_from_corpus(ROOT / "data" / "corpus", HashingEmbedder())


class ChunkingTests(unittest.TestCase):
    def test_heading_metadata_overlap_and_sentence_boundary(self):
        paragraph = "星岚科技规定员工必须遵守信息安全要求，不得泄露客户资料。" * 4
        markdown = "# 安全规范\n\n## 账号管理\n\n" + "\n\n".join([paragraph] * 6)
        chunks = chunk_markdown(markdown, "demo")
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(chunk.heading_path == ("安全规范", "账号管理") for chunk in chunks))
        self.assertTrue(chunks[0].text.endswith("。"))
        self.assertIn(chunks[0].text[-40:], chunks[1].text)
        self.assertTrue(all(len(chunk.text) <= 600 for chunk in chunks))


class BM25Tests(unittest.TestCase):
    def test_cjk_bigrams_and_ascii_words(self):
        tokens = tokenize("企业知识库 RAG")
        self.assertIn("企业", tokens)
        self.assertIn("业知", tokens)
        self.assertIn("知识", tokens)
        self.assertIn("rag", tokens)

    def test_obvious_document_ranks_first(self):
        index = BM25Index()
        index.add("leave", "年假 婚假 产假 请假 审批 考勤")
        index.add("expense", "差旅 报销 发票 住宿 交通")
        hits = index.search("发票抬头怎么填", k=2)
        self.assertEqual(hits[0][0], "expense")


class FusionTests(unittest.TestCase):
    def test_dual_arm_document_beats_single_arm(self):
        scores = reciprocal_rank_fusion([["both", "bm25"], ["both", "dense"]], k=60)
        self.assertGreater(scores["both"], scores["bm25"])

    def test_weighted_rrf(self):
        scores = reciprocal_rank_fusion([["a"], ["b"]], k=10, weights=[1.0, 4.0])
        self.assertGreater(scores["b"], scores["a"])

    def test_hybrid_keeps_shared_top5_and_one_chunk_per_doc(self):
        retriever = HybridRetriever(make_index())
        question = "P0 故障的首次响应 SLA 是多少分钟？"
        bm25_docs = [item.doc_id for item in retriever.rank(question, 5, "bm25")]
        dense_docs = [item.doc_id for item in retriever.rank(question, 5, "dense")]
        hybrid = retriever.rank(question, 5, "hybrid")
        hybrid_docs = [item.doc_id for item in hybrid]
        self.assertTrue(set(bm25_docs[:5]) & set(dense_docs[:5]) <= set(hybrid_docs))
        self.assertEqual(len(hybrid_docs), len(set(hybrid_docs)))


class IndexTests(unittest.TestCase):
    def test_hashing_embedding_is_stable_and_normalized(self):
        first = HashingEmbedder().embed_query("年假与请假制度")
        second = HashingEmbedder().embed_query("年假与请假制度")
        self.assertEqual(first, second)
        self.assertAlmostEqual(sum(value * value for value in first), 1.0, places=6)

    def test_save_load_round_trip(self):
        base = ROOT / "tests" / "_tmp_index"
        shutil.rmtree(base, ignore_errors=True)
        try:
            corpus = base / "corpus"
            corpus.mkdir(parents=True)
            (corpus / "a.md").write_text("# 考勤\n\n年假与补卡规则。", encoding="utf-8")
            index = VectorIndex.build_from_corpus(corpus, HashingEmbedder())
            index.save(base / "index")
            loaded = VectorIndex.load(base / "index")
            self.assertEqual(loaded.chunks, index.chunks)
            query = HashingEmbedder().embed_query("年假")
            self.assertEqual(
                [(chunk.doc_id, round(score, 6)) for chunk, score in loaded.search(query, 2)],
                [(chunk.doc_id, round(score, 6)) for chunk, score in index.search(query, 2)],
            )
        finally:
            shutil.rmtree(base, ignore_errors=True)


class ToolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(make_index())
        cls.tools = build_default_tools(cls.retriever, ROOT / "data" / "analytics.db")

    def test_calculator_valid(self):
        self.assertEqual(self.tools.call("calculator", {"expression": "2 + 3 * 4"})["result"], 14)

    def test_calculator_rejects_import_name_and_call(self):
        for expression in ("__import__('os')", "salary * 2", "sum([1, 2])", "x.__class__"):
            with self.assertRaises(ToolError):
                self.tools.call("calculator", {"expression": expression})

    def test_sql_rejects_delete_and_stacked_statements(self):
        with self.assertRaises(ToolError):
            self.tools.call("sql_query", {"sql": "DELETE FROM orders"})
        with self.assertRaises(ToolError):
            self.tools.call("sql_query", {"sql": "SELECT 1; SELECT 2"})

    def test_sql_valid_select(self):
        result = self.tools.call("sql_query", {"sql": "SELECT COUNT(*) AS n FROM customers"})
        self.assertEqual(result["rows"], [[120]])


class AgentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.retriever = HybridRetriever(make_index())
        cls.tools = build_default_tools(cls.retriever, ROOT / "data" / "analytics.db")

    def test_stops_at_max_steps_and_records_trace(self):
        calls = [
            ChatResponse(tool_calls=[ToolCall(str(i), "calculator", {"expression": "1 + 1"})])
            for i in range(3)
        ]
        result = Agent(MockModel(calls), self.tools, max_steps=2).run("一直算")
        self.assertEqual(result.stopped_reason, "max_steps")
        self.assertEqual(len(result.trace.steps), 2)
        trace = json.loads(result.trace.to_json())
        self.assertEqual(trace["steps"][0]["tool"], "calculator")
        self.assertIn("tool", trace["steps"][0])

    def test_mock_tool_call_executes_and_returns_answer(self):
        script = [ChatResponse(tool_calls=[ToolCall("s1", "kb_search", {"query": "年假", "k": 2})])]
        result = Agent(MockModel(script), self.tools).run("年假有几天？")
        self.assertEqual(result.stopped_reason, "answered")
        self.assertEqual(result.trace.steps[0].tool, "kb_search")
        self.assertIn("01-attendance", result.answer)


class EvalTests(unittest.TestCase):
    def test_dataset_split_counts(self):
        questions = load_jsonl(ROOT / "evals" / "qa.jsonl")
        self.assertEqual(sum(item["split"] == "dev" for item in questions), 18)
        self.assertEqual(sum(item["split"] == "test" for item in questions), 12)
        self.assertEqual(sum(item["query_type"] == "lexical" for item in questions), 14)
        self.assertEqual(sum(item["query_type"] == "semantic" for item in questions), 16)

    def test_smoke_eval_on_three_questions(self):
        base = ROOT / "tests" / "_tmp_eval"
        shutil.rmtree(base, ignore_errors=True)
        try:
            report = run_evaluation(limit=3, report_dir=base)
            self.assertEqual(report["retrieval"]["questions"], 3)
            self.assertIn("Hybrid + RRF", report["retrieval"]["configs"])
            self.assertTrue((base / "eval-report.md").exists())
            self.assertTrue((base / "eval-report.json").exists())
        finally:
            shutil.rmtree(base, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
