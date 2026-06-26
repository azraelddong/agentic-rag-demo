# agentic-rag-demo

企业级 RAG / Agentic RAG 学习项目。基础 RAG 闭环已完成，第一版 Agent（plan → execute → reflect → final）已上线 `/api/agent/ask`。

## 技术栈

- Python 3.11+
- FastAPI
- LangChain / langchain-text-splitters
- LangGraph 预留
- Milvus / Zilliz
- OpenAI-compatible Chat API，支持 Qwen / DeepSeek / OpenAI 等兼容接口
- OpenAI-compatible Embedding，预留 bge-m3
- 预留 bge-reranker
- Docker Compose: Milvus standalone + etcd + minio + Attu
- uv 包管理

## 项目结构

```text
app/
  api/          FastAPI 路由层（chat、agent、document）
  core/         配置、日志、异常
  llm/          Chat 与 Embedding 模型适配
  rag/          文档解析、切分、向量库、检索、Prompt、RAG Chain
  agent/        Agentic RAG：AgentService + LangGraph 骨架预留
  schemas/      请求/响应模型（chat、agent、document）
  services/     应用服务编排
docs/           待索引文档目录
scripts/        本地脚本
tests/          测试
```

## 启动步骤

安装 uv。如果本机已经安装，可跳过：

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv --version
```

创建虚拟环境并同步依赖：

```powershell
cd E:\study\agentic-rag-demo
uv sync
copy .env.example .env
```

编辑 `.env`，填写 `LLM_API_KEY`，并按实际供应商设置 `LLM_BASE_URL`、`LLM_MODEL_NAME`、Embedding 配置和 `EMBEDDING_DIMENSION`。

启动 Milvus 和 Attu：

```powershell
docker compose up -d
```

启动 API：

```powershell
uv run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

健康检查：

```powershell
curl.exe http://localhost:8000/health
```

## .env 配置说明

- `LLM_BASE_URL`: OpenAI-compatible API 地址。
- `LLM_API_KEY`: LLM API Key，不要提交到 Git。
- `LLM_MODEL_NAME`: Chat 模型名，例如 OpenAI、Qwen、DeepSeek 的兼容模型名。
- `EMBEDDING_BASE_URL`: Embedding API 地址；为空时复用 `LLM_BASE_URL`。
- `EMBEDDING_API_KEY`: Embedding API Key；为空时复用 `LLM_API_KEY`。
- `EMBEDDING_MODEL_NAME`: Embedding 模型名。
- `EMBEDDING_DIMENSION`: Embedding 维度，必须和实际模型输出一致，否则 Milvus collection 创建或插入会失败。
- `MILVUS_URI`: 本地默认 `http://localhost:19530`；Zilliz Cloud 可改为云端 URI。
- `MILVUS_COLLECTION_NAME`: Collection 名称，从环境变量读取。
- `RAG_SCORE_THRESHOLD`: 相似度阈值；使用 COSINE/IP 时分数越大越相关。

## Milvus / Attu 地址

- Milvus: `http://localhost:19530`
- Milvus HTTP health: `http://localhost:9091/healthz`
- Attu: `http://localhost:3000`
- MinIO Console: `http://localhost:9001`，默认账号密码 `minioadmin / minioadmin`

## API 示例

上传文档：

```powershell
curl.exe -X POST "http://localhost:8000/api/documents/upload" `
  -F "file=@E:\study\agentic-rag-demo\docs\example.md"
```

索引 `docs/` 目录：

```powershell
curl.exe -X POST "http://localhost:8000/api/documents/index" `
  -H "Content-Type: application/json" `
  -d "{\"rebuild\": true}"
```

提问：

```powershell
curl.exe -X POST "http://localhost:8000/api/chat/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"这个项目第一阶段实现了什么？\",\"top_k\":5}"
```

响应会返回：

```json
{
  "answer": "...",
  "sources": [
    {
      "file_name": "example.md",
      "file_path": "E:\\study\\agentic-rag-demo\\docs\\example.md",
      "chunk_index": 0,
      "source_type": "md",
      "score": 0.82
    }
  ]
}
```

### Agent 提问（带 trace）

```powershell
curl.exe -X POST "http://localhost:8000/api/agent/ask" `
  -H "Content-Type: application/json" `
  -d "{\"question\":\"这个项目第一阶段实现了什么？\",\"top_k\":5}"
```

响应除 `answer` 和 `sources` 外，额外返回 `trace`，包含完整的执行步骤：

```json
{
  "answer": "...",
  "sources": [
    {
      "file_name": "example.md",
      "file_path": "E:\\study\\agentic-rag-demo\\docs\\example.md",
      "chunk_index": 0,
      "source_type": "md",
      "score": 0.82
    }
  ],
  "trace": {
    "plan": "rag_search",
    "reflection": "supported",
    "iterations": 1,
    "steps": [
      {"node": "plan", "decision": "rag_search"},
      {"node": "execute_rag", "decision": "initial", "source_count": 3, "top_k": 5},
      {"node": "reflect", "reflection": "supported"},
      {"node": "final", "reflection": "supported", "source_count": 3}
    ]
  }
}
```

## Agent 架构（第二版 — LangGraph + 智能重试）

第二版 Agent 使用 LangGraph `StateGraph` 编排，支持单次智能重试：

```text
START
  → plan              (确定性规则，固定 rag_search)
  → execute_rag       (调用 RAGChain，复用全部检索能力)
  → reflect           (确定性规则，判断 context 是否充分)
  → prepare_retry     (仅当 insufficient_context 且未达最大轮数)
  → execute_rag       (retry: top_k 翻倍 + 强制 multi-query rewrite + 强制 hybrid 检索)
  → reflect           (再次判断)
  → final             (返回最终结果)
  → END
```

### 重试策略

当第一轮 `reflection = insufficient_context` 时，Agent 自动触发一次重试：

| 策略维度 | 常规执行 | 重试执行 |
|----------|----------|----------|
| `top_k` | 请求指定 或 默认 `rag_top_k` | `min(top_k × 2, 20)` |
| 检索器 | 按 `retrieval_method` 配置 | 强制 `HybridRetriever` |
| Query Rewrite | 按 `query_rewrite_method` 配置 | 强制 `MultiQueryRewriter` |
| 最大轮数 | — | 2（最多重试一次） |

- 重试策略在服务端固定，不暴露给请求参数。
- 不引入 LLM judge，不开放任意工具调用。

### 请求 / 响应契约

**请求** `POST /api/agent/ask`，字段与 `ChatRequest` 保持一致：

| 字段 | 类型 | 必填 | 说明 |
|------|------|------|------|
| `question` | string | 是 | 用户问题，1-2000 字符 |
| `top_k` | int | 否 | 检索召回数量，1-20 |
| `metadata_filter` | dict | 否 | Milvus metadata 过滤条件 |

**响应** `AgentResponse`：

| 字段 | 类型 | 说明 |
|------|------|------|
| `answer` | string | 生成的回答 |
| `sources` | list[Source] | 引用的文档片段（复用 Chat 的 Source 结构） |
| `trace.plan` | string | 规划动作，当前固定 `"rag_search"` |
| `trace.reflection` | string | 反思结果：`"supported"` 或 `"insufficient_context"` |
| `trace.iterations` | int | 执行轮数（1 或 2） |
| `trace.steps` | list[Step] | 图执行步骤，含节点名、决策、source_count、top_k 等 |

### 依赖注入

```text
get_rag_chain()                     # RAGChain（Chat 和 Agent 共享）
get_chat_service()   → get_rag_chain()
get_agent_service()  → get_rag_chain() + retry_retriever (Hybrid) + retry_query_rewriter (MultiQuery)
```

## 本地脚本

```powershell
uv run python scripts/index_docs.py
uv run python scripts/ask.py "这个项目支持哪些 Agentic RAG 扩展？"
```

## 依赖管理

本项目使用 `uv` 管理 Python 环境和依赖，依赖声明集中在 `pyproject.toml`。

常用命令：

```powershell
# 同步生产依赖和 dev 依赖
uv sync

# 只同步生产依赖
uv sync --no-dev

# 添加运行依赖
uv add fastapi

# 添加开发依赖
uv add --dev pytest

# 更新锁文件
uv lock

# 运行测试
uv run pytest -q
```

## 后续 Agentic RAG 扩展路线

- ✅ **第一版 Agent**：确定性 plan/reflect，单轮 rag_search，trace 返回。
- ✅ **第二版 Agent**：LangGraph 图编排，单次智能重试（top_k 翻倍 + 强制 multi-query rewrite + 强制 hybrid 检索）。
- ✅ 将 `app/agent/graph.py` 改为真正的 LangGraph 图编排。
- ✅ 在 trace 中增加执行步骤 `steps`，含节点名、决策、source_count、top_k、retry_reason。
- 将确定性 plan/reflect 替换为可配置 LLM planner/judge。
- 增加更多受控工具，例如文档库路由、SQL 查询或 Web 检索。
- retrieval router: 按问题类型路由到不同知识库、SQL、工具或 Web 检索。
- multi-hop retrieval: 多轮检索和证据组合，处理复杂问题。
- 在 trace 中增加节点耗时、检索 query 详情。

`app/agent/graph.py` 现已包含完整的 `StateGraph` 实现：`plan → execute_rag → reflect → prepare_retry → final`。
