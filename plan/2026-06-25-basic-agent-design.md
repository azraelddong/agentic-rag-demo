# Basic Agent 设计文档

## 背景

当前项目已经完成 Basic RAG 闭环：

- `/api/chat/ask` 提供稳定的基础问答接口。
- `RAGChain` 已包含 query rewrite、多查询检索、hybrid 检索、rerank、prompt 构建和答案生成。
- `app/agent/graph.py` 已预留 Agentic RAG 扩展点，但尚未接入实际流程。

第一版 Agent 目标是增加“规划、执行、反思”的基础 Agent 能力，同时保持实现简单、风险可控，不影响现有 Basic RAG 链路。

## 目标

第一版新增独立接口 `/api/agent/ask`，提供最小闭环 Agent：

```text
plan -> execute -> reflect -> final
```

核心目标：

- 保留 `/api/chat/ask` 当前行为，不做兼容性破坏。
- Agent 复用现有 RAG 能力，不重复实现检索、rerank、生成逻辑。
- 响应中返回精简 `trace`，让调用方能看到规划和反思结果。
- 第一版不做多轮补检索、不开放任意工具调用、不引入长期记忆。

## 非目标

第一版暂不实现以下能力：

- 多轮循环 retry 或自动补检索。
- 复杂任务拆解和多步骤工具调用。
- 长期记忆、用户画像或会话状态。
- Web 检索、SQL 查询、外部 API 工具调用。
- 用 LLM 做二次 judge 或事实核验。
- 替换或改造现有 `/api/chat/ask`。

## 推荐架构

采用薄封装 Agent 编排层：

```text
FastAPI
  /api/chat/ask    -> ChatService -> RAGChain
  /api/agent/ask   -> AgentService -> RAGChain
```

建议新增文件：

```text
app/
  api/
    agent_api.py
  agent/
    agent_service.py
    schemas.py
  schemas/
    agent_schema.py
```

保留 `app/agent/graph.py` 作为后续 LangGraph 图编排入口。第一版可以不强制使用 LangGraph 真图，避免为了形式增加复杂度。

## 请求和响应契约

### 请求

`POST /api/agent/ask`

```json
{
  "question": "这个项目第一阶段实现了什么？",
  "top_k": 5,
  "metadata_filter": null
}
```

字段建议与现有 `ChatRequest` 保持一致：

- `question`: 用户问题，必填，长度规则沿用现有问答接口。
- `top_k`: 可选，检索召回数量。
- `metadata_filter`: 可选，Milvus metadata filter。

### 响应

```json
{
  "answer": "...",
  "sources": [
    {
      "file_name": "example.md",
      "file_path": "/path/to/example.md",
      "chunk_index": 0,
      "source_type": "md",
      "score": 0.82
    }
  ],
  "trace": {
    "plan": "rag_search",
    "reflection": "supported",
    "iterations": 1
  }
}
```

`sources` 结构复用现有 `Source`。

`trace` 第一版只保留必要字段：

- `plan`: 规划动作。第一版固定或近似固定为 `rag_search`。
- `reflection`: 反思结果，取值为 `supported` 或 `insufficient_context`。
- `iterations`: 执行轮数。第一版固定为 `1`。

## Agent 流程

### 1. Plan

第一版 planner 使用确定性规则，不调用 LLM。

默认计划：

```text
rag_search
```

含义：使用现有 RAG 链路进行检索增强回答。

后续可扩展计划类型：

- `direct_answer`: 不检索，直接回答通用问题。
- `rewrite_and_search`: 强制改写后检索。
- `need_more_context`: 问题不完整时要求用户补充。

第一版只实现 `rag_search`，为后续扩展保留枚举或常量。

### 2. Execute

执行阶段直接调用现有 `RAGChain.ask()` 或通过注入的 `ChatService.ask()` 复用现有能力。

推荐优先复用 `RAGChain.ask()`，因为 AgentService 可以更清晰地拿到：

```text
answer, results
```

然后自行构建 `sources` 和 `trace`。

如果为了减少重复 source 构建逻辑，也可以复用 `ChatService.ask()`。但长期看，Agent 编排层直接依赖 `RAGChain` 更利于后续拿到中间状态。

### 3. Reflect

第一版 reflector 使用确定性规则，不额外调用 LLM。

推荐规则：

```text
如果 sources 非空，且 answer != "知识库中未找到相关信息":
    reflection = "supported"
否则:
    reflection = "insufficient_context"
```

这样可以避免第一版引入额外模型成本和不稳定性。

### 4. Final

返回最终答案、来源和 trace：

```text
answer + sources + trace
```

如果反思结果是 `insufficient_context`，第一版不自动 retry。是否返回原始 RAG 的未找到提示，由现有 `RAGChain` 决定。

## 依赖注入

建议在 `app/api/dependencies.py` 中新增：

```text
get_agent_service()
```

依赖关系：

```text
AgentService
  -> Settings
  -> RAGChain 或 ChatService
```

如果 AgentService 直接依赖 RAGChain，需要考虑把现有 `get_chat_service()` 中创建 RAGChain 的逻辑抽成 `get_rag_chain()`，避免重复构造。

推荐依赖结构：

```text
get_rag_chain()
get_chat_service()  -> get_rag_chain()
get_agent_service() -> get_rag_chain()
```

这样 Basic RAG 和 Agentic RAG 共用同一个核心链路，后续维护成本更低。

## 配置

第一版不新增必需环境变量。

可选预留配置：

```text
agent_enabled = true
agent_max_iterations = 1
```

但为保持简单，第一版可以先不加入配置项，直接固定 `iterations = 1`。

## 错误处理

`/api/agent/ask` 应复用现有异常处理方式：

- 问题为空或超长时，返回与 `/api/chat/ask` 一致的业务错误。
- RAG 执行中出现业务异常时，走现有 `AppError` 处理器。
- 未预期异常仍由全局异常处理器返回 `internal_error`。

不建议第一版在 AgentService 内吞掉所有异常，否则生产排查会变困难。

## 日志与可观测性

AgentService 应记录关键结构化信息：

```text
plan=rag_search
reflection=supported|insufficient_context
iterations=1
source_count=N
```

不要在日志中输出完整用户问题或完整答案，避免生产环境日志泄漏敏感内容。可以按需输出截断后的 question preview。

## 测试建议

第一版测试重点：

- `AgentService` 在有 sources 且 answer 正常时返回 `supported`。
- `AgentService` 在无 sources 或未找到答案时返回 `insufficient_context`。
- `/api/agent/ask` 响应包含 `answer`、`sources`、`trace`。
- `/api/chat/ask` 现有测试不需要改变，确保 Basic RAG 行为未被影响。
- 依赖注入不会重复构造多套 RAG 组件。

## 后续演进

第一版完成后，可以按以下顺序扩展：

1. 将确定性 plan/reflect 替换为可配置 LLM planner/judge。
2. 增加最多一次补检索 retry。
3. 将 `app/agent/graph.py` 改为真正的 LangGraph 图。
4. 在 trace 中增加节点耗时、检索 query、重试原因。
5. 增加更多受控工具，例如文档库路由、SQL 查询或 Web 检索。

## 实施顺序建议

后续进入实现时，建议按以下顺序：

1. 新增 `AgentRequest`、`AgentResponse`、`AgentTrace` schema。
2. 抽出 `get_rag_chain()` 依赖，保持 `ChatService` 行为不变。
3. 新增 `AgentService`，实现 plan/execute/reflect/final。
4. 新增 `agent_api.py` 并在 `app/main.py` 注册路由。
5. 增加服务层和 API 层测试。
6. 运行全量测试，确认 `/api/chat/ask` 未受影响。
