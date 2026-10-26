# 架构说明

## 1. 模块职责

| 模块 | 职责 | 关键实现 |
| --- | --- | --- |
| `kb_agent/chunking.py` | Markdown → 带标题路径的块 | 按 `#` 层级维护 `heading_path`；段落/句子打包；600 字上限、80 字 overlap；表格整块保留 |
| `kb_agent/embeddings.py` | 文本 → 向量 | 512 维 blake2b 特征哈希；ASCII 词 + CJK 字符 bigram；L2 归一化；可选 OpenAI 兼容 Embedding |
| `kb_agent/bm25.py` | 关键词召回 | k1=1.5、b=0.75；中文 bigram + ASCII 词；DF/IDF、文档长度归一 |
| `kb_agent/index.py` | 向量存取与检索 | 纯 Python 余弦；`chunks.jsonl` + `meta.json`；`build_from_corpus` |
| `kb_agent/retriever.py` | 混合检索 | BM25 / Dense 排名；加权 RRF（k=10，1:4）；按 `(doc_id, heading)` 去重；每文档最多一个结果 |
| `kb_agent/llm.py` | 模型抽象 | `ChatModel` Protocol；OpenAI 兼容 urllib 客户端；离线 `MockModel`（脚本 + 最多 3 步规则计划） |
| `kb_agent/tools.py` | 工具注册 | JSON Schema 描述；`kb_search`、只读 `sql_query`、AST 白名单 `calculator` |
| `kb_agent/agent.py` | 工具调用循环 | 最多 6 步；每步记录工具、参数、ok、耗时、结果摘要；支持 max_steps 优雅停止 |
| `kb_agent/cli.py` | 命令行 | ingest / ask / eval |
| `evals/run_eval.py` | 评估 | dev/test 分层；三配置检索指标；单跳/多跳 Agent 指标；Markdown/JSON 报告 |
| `scripts/make_analytics_db.py` | 数据生成 | 固定 seed，120 客户、500 订单、SQLite |

## 2. 请求流

### 2.1 ingest

```text
data/corpus/*.md
  → chunk_corpus()
  → Chunk(text, doc_id, heading_path, chunk_index)
  → HashingEmbedder.embed_documents(retrieval_text)
  → VectorIndex(chunks, vectors)
  → data/index/chunks.jsonl + meta.json
```

`retrieval_text()` 由标题路径与正文拼接，让检索能利用“差旅报销管理办法 > 2. 费用标准”这类上下文。BM25 在 `HybridRetriever` 初始化时对同一文本重建，不单独落盘。

### 2.2 ask

```text
问题
  → Agent(messages=[system, user])
  → model.complete(tools=schemas)
      ├─ 无 tool_calls：返回最终答案，stopped_reason=answered
      └─ 有 tool_calls：
           → ToolRegistry.call()
           → 结果 JSON 作为 role=tool 消息
           → 进入下一轮（最多 6 轮）
```

工具失败不会中断循环：`ToolError` 被转成 `{"error": ...}` 回填，模型仍可换工具或给出兜底回答。达到 `max_steps` 时返回部分内容并设置 `stopped_reason=max_steps`。

### 2.3 Agent 消息结构

内部消息保持 OpenAI chat-completions 形状：

- `assistant.tool_calls[].function.arguments` 为 JSON 字符串。
- `tool` 消息带 `tool_call_id`、`name`、`content`。
- `TraceStep.result_excerpt` 保留工具结果前 300 字符，供报告与排障。
- `Trace.to_json()` 输出完整轨迹，不包含 API Key。

## 3. 检索流与融合

```text
query
  ├─ BM25：tokenize → 每文档打分 → top 30
  └─ Dense：领域同义词扩展 → 哈希向量 → 余弦 → top 30
          ↓
   weighted RRF:
   score(d) = 1.0/(10 + rank_bm25(d)) + 4.0/(10 + rank_dense(d))
          ↓
   按 (doc_id, heading) 去重；文档级去重；取 top k
```

- 权重来自 dev 扫参，test 不参与选择；评估报告保留完整扫参表。
- `max_per_doc=1`：保证 top-k 是 k 个不同文档，避免单文档多个块占满上下文。
- Dense 的查询侧同义词表在 `_DENSE_ALIASES`，是人工维护的领域先验，不作为“模型能力”宣传。

## 4. 数据流

### 4.1 文档

- 13 篇中文 Markdown，覆盖考勤、差旅、IT、产品、数据安全、入职、值班、绩效、采购、培训、薪酬、协作、分析口径。
- 每篇有 H1/H2、编号条款、必要的表格；文档 ID 取文件名 stem（如 `01-attendance`）。
- 表格作为整块保留，避免表头和数值被切散。

### 4.2 分析库

`data/analytics.db`：

```sql
customers(customer_id, customer_name, region, tier, joined_at)
orders(order_id, region, channel, amount, created_at, status, customer_id)
```

- seed=20260901；订单时间覆盖 2026-07 至 2026-09。
- `status`：paid / refunded / pending / cancelled。
- 分析口径写在 `13-analytics-glossary.md`：GMV 只算 `status='paid'`。

### 4.3 评估集

- `evals/qa.jsonl`：30 条，14 条词面型、16 条语义型；18 dev / 12 test，固定种子 20261026 分层抽样。
- `evals/tasks.jsonl`：20 条，15 单跳 + 5 多跳；多跳要求 `required_tools` 全被调用。
- 所有指标由 `run_eval.py` 运行时计算，报告只落盘，不写死数字。

## 5. 关键设计选择

### 为什么标题路径单独存

正文不插入标题，避免重复标题污染 BM25 词频；但检索文本拼接标题，让标题词参与召回。返回结果仍展示正文与独立 `heading_path`，便于引用。

### 为什么用 RRF 而不是分数归一化

BM25 分数量级随查询变化，余弦在 [-1,1]；跨查询做 min-max 归一化不稳定。RRF 只用排名，工程上更稳；加权是为了处理两个召回臂质量不对称。

### 为什么 Agent 只做薄循环

项目目标是展示可解释的工具调用与评估，不重造 LangChain。循环、trace、步数上限和工具注册都在 100 行内，便于面试逐行讲。

## 6. 扩展点

1. **替换 Dense**：把 `HashingEmbedder` 换成 `OpenAICompatibleEmbedder` 或本地模型，`VectorIndex` 接口不变；需要重新 ingest。
2. **ANN**：为 `VectorIndex.search` 增加 FAISS/hnswlib 实现，保留 `search` 返回 `(Chunk, score)`。
3. **Reranker**：在 `HybridRetriever.rank` 后加 cross-encoder 或 LLM 重排，top-k 输入。
4. **真实 Planner**：`Agent` 的 `ChatModel` 已与工具 schema 解耦；替换 `MockModel` 即接真实模型。
5. **工具扩展**：实现 `Tool` 并注册；SQL 类工具应使用只读账号/只读副本，不能只依赖字符串校验。
6. **服务化**：`server/app.py` 提供 `/ask`、`/health`；横向扩展时把索引放到共享存储或独立检索服务。
7. **可观测性**：将 `Trace` 接入日志/追踪系统；当前只在响应和报告中返回。

## 7. 失败模式与边界

- 空召回：`kb_search` 返回空列表，MockModel 会明确说无结果；真实模型应被提示不要编造。
- 工具参数错误：`ToolRegistry` 捕获 `TypeError` 转为 `ToolError`，回填给模型。
- SQL 注入/越权：只读 URI + 单条 SELECT/PRAGMA + 分号拦截；生产环境还需只读数据库账号、schema 白名单、审计。
- 大结果：SQL 限制 100 行，KB 片段上限 600 字，避免上下文爆炸。
- 小语料：当前线性扫描；10 倍语料需要索引层扩容。
