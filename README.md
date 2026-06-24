# agentic-rag-demo

企业级 RAG / Agentic RAG 学习项目。第一阶段只实现基础 RAG 闭环，后续在清晰分层的基础上扩展 LangGraph Agentic RAG。

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
  api/          FastAPI 路由层
  core/         配置、日志、异常
  llm/          Chat 与 Embedding 模型适配
  rag/          文档解析、切分、向量库、检索、Prompt、RAG Chain
  agent/        LangGraph Agentic RAG 骨架预留
  schemas/      请求/响应模型
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

- query rewrite: 根据对话历史或用户意图改写检索问题。
- retrieval router: 按问题类型路由到不同知识库、SQL、工具或 Web 检索。
- multi-hop retrieval: 多轮检索和证据组合，处理复杂问题。
- answer judge: 检查答案是否被上下文支持，降低幻觉。
- self-correction: 当 judge 失败时自动补检索、改写问题或重新生成。

`app/agent/graph.py` 已预留 LangGraph 状态和构建入口。第一阶段 API 不依赖 Agent，保持基础 RAG 链路清晰可调试。
