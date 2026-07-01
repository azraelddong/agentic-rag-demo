# Agentic RAG Demo — 基于 LangGraph 的智能检索增强生成

## 概述

本模块展示如何使用 [LangGraph](https://github.com/langchain-ai/langgraph) 构建具备**自纠正能力**的 Agentic RAG 图。

与 Phase 1 的线性 RAG 管道（`app/rag/rag_chain.py`）不同，Agentic RAG 引入了**条件路由**和**反馈循环**：当检索结果不足以回答问题时，图会带着评判反馈回到查询改写节点，自动调整检索策略后重试。

## 图结构

```
                    ┌──────────────┐
                    │ rewrite_query│ ←──────────────┐
                    │  查询改写     │                 │
                    └──────┬───────┘                 │
                           │                         │
                    ┌──────▼───────┐                 │
                    │   retrieve   │                 │
                    │   多路检索    │                 │
                    └──────┬───────┘                 │
                           │                         │
                    ┌──────▼───────┐                 │
                    │judge_relevance│    [不相关]     │
                    │   答案评判    ├─────────────────┘
                    └──────┬───────┘
                           │ [相关]
                    ┌──────▼───────┐
                    │generate_answer│
                    │   生成答案    │
                    └──────────────┘
```

## 核心节点

| 节点 | 说明 |
|---|---|
| `rewrite_query` | 将口语化问题改写为适合向量检索的形式，提取 BM25 关键词；自纠正时带入上轮反馈 |
| `retrieve` | 对改写后的多个查询变体分别检索，按文本指纹去重，再经 Reranker 重排序 |
| `judge_relevance` | LLM 评判检索文档是否足以回答问题，输出 JSON（含 `relevant` / `reason` / `feedback`） |
| `generate_answer` | 基于检索上下文生成最终答案，无上下文时返回兜底回复 |

## 自纠正循环

```
第 1 轮：rewrite → retrieve → judge（不相关，给出反馈）
第 2 轮：rewrite（带入反馈）→ retrieve → judge（相关）→ generate
...
第 N 轮：达到 max_rewrite_attempts → 强制 generate（即使不理想）
```

评判 LLM 会输出具体的改写建议（如 "请尝试更通用的关键词"），该建议会拼接到下一轮改写的 prompt 中，引导改写器生成更精准的查询。

## 目录结构

```
app/agent/demo/
├── __init__.py              # 包说明
├── agentic_rag_graph.py     # LangGraph 图定义（节点、路由、状态）
├── demo_runner.py           # 独立运行器（复用项目 DI 组件）
└── README.md                # 本文档
```

## 快速开始

### 前置条件

- Python >= 3.11
- 已安装依赖（`uv sync`）
- Milvus 向量数据库已启动（`docker-compose up -d`）
- `.env` 中已配置 LLM API Key 等参数

### 运行方式

```bash
# 基础用法
python -m app.agent.demo.demo_runner "什么是RAG？"

# 详细模式 —— 打印每个节点的中间状态
python -m app.agent.demo.demo_runner "什么是RAG？" --verbose

# 交互式模式
python -m app.agent.demo.demo_runner --interactive

# JSON 输出（方便程序对接）
python -m app.agent.demo.demo_runner "什么是RAG？" --json

# 调参
python -m app.agent.demo.demo_runner "问题" \
    --retrieval-k 15 \
    --rerank-top-n 8 \
    --max-rewrite-attempts 3 \
    --score-threshold 0.5

# 禁用查询改写（直接使用原始问题检索）
python -m app.agent.demo.demo_runner "问题" --no-rewrite
```

### 交互模式命令

| 输入 | 功能 |
|---|---|
| 任意问题文本 | 执行 Agentic RAG 查询 |
| `verbose` | 切换详细模式 |
| `quit` / `exit` / `q` | 退出 |

## 参数说明

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--retrieval-k` | 10 | 检索返回的文档片段数 |
| `--rerank-top-n` | 5 | 重排序后保留的文档片段数 |
| `--max-rewrite-attempts` | 2 | 最大改写尝试次数（自纠正循环上限） |
| `--score-threshold` | 配置文件中的值 | 检索分数阈值 |
| `--no-rewrite` | false | 禁用查询改写 |
| `--verbose` | false | 打印详细中间状态 |
| `--interactive` | false | 进入交互式问答模式 |
| `--json` | false | 以 JSON 格式输出最终结果 |

## API 使用

如果需要在代码中集成（而非命令行运行），可以直接构建图：

```python
from app.agent.demo.agentic_rag_graph import build_agentic_rag_graph
from app.api.dependencies import get_chat_model, get_retriever, get_reranker, get_query_rewriter

# 构建图
graph = build_agentic_rag_graph(
    chat_model_fn=get_chat_model().generate,
    retriever=get_retriever(),
    reranker=get_reranker(),
    query_rewriter=get_query_rewriter(),
    max_rewrite_attempts=2,
    retrieval_k=10,
    rerank_top_n=5,
)

# invoke 模式 —— 返回最终状态
result = graph.invoke({"query": "什么是RAG？"})
print(result["answer"])

# stream 模式 —— 逐步返回每个节点的状态
for step in graph.stream({"query": "什么是RAG？"}):
    for node_name, node_state in step.items():
        print(f"节点 {node_name} 完成：{list(node_state.keys())}")
```

## 状态字段

`AgenticRAGState` 包含以下字段：

| 字段 | 类型 | 说明 |
|---|---|---|
| `query` | `str` | 用户原始问题 |
| `rewritten_queries` | `list[str]` | 改写后的查询变体列表 |
| `rewrite_keywords` | `list[str]` | BM25 关键词列表 |
| `rewrite_attempts` | `int` | 当前改写轮次 |
| `rewrite_feedback` | `str` | 上轮评判给出的改写建议 |
| `contexts` | `list[SearchResult]` | 检索到的文档片段 |
| `is_relevant` | `bool` | 检索结果是否足以回答问题 |
| `relevance_reason` | `str` | 相关性评判理由 |
| `answer` | `str` | 最终生成的答案 |
| `messages` | `list[str]` | 中间日志消息（调试用） |

## 与 Phase 1 RAG 的对比

| 特性 | Phase 1 `RAGChain` | Agentic RAG 图 |
|---|---|---|
| 架构 | 线性管道 | 有向图 + 条件路由 |
| 查询改写 | 可选单次改写 | 支持多轮带反馈的自纠正改写 |
| 检索 | 单次检索 + 去重 | 同左，但可被评判结果触发重新检索 |
| 答案质量保障 | 无评判环节 | LLM 评判 + 最多 N 轮重试 |
| 观测性 | 日志 | 日志 + 状态中的 `messages` 字段 |
| 框架 | 手写编排 | LangGraph `StateGraph` |

## 扩展方向

当前演示已保留清晰的扩展点，可在基础上增加：

- **多跳检索（Multi-hop）**：检索 → 从结果中提取新实体 → 再检索
- **工具调用（Tool Use）**：让 Agent 自主决定检索 vs 联网搜索 vs 查数据库
- **答案校验（Answer Verify）**：生成后再次评判答案质量，不合格则回退
- **Checkpoint 持久化**：接入 LangGraph Checkpointer 实现状态断点续传
- **Human-in-the-loop**：在 `judge_relevance` 后插入人工审批节点
