# Simple Agent Demo

基于 LangChain `create_agent` 构建的智能助手示例，集成工具调用防错策略与 Redis 会话记忆。

## 运行方式

```bash
uv run python demo/simple_agent.py
```

## 防错策略（五层）

| 层级 | 说明 | 实现位置 |
|------|------|----------|
| ① 工具描述 | 精确的 docstring：做什么、何时用、何时不用、输入示例 | 每个 `@tool` |
| ② 系统提示词 | 路由规则 + Few-shot 示例 + 边界歧义消解规则 | `SYSTEM_PROMPT` |
| ③ 参数校验 | 使用 `Annotated` + `Field` 让 LLM 得到更精确的参数 schema | 工具参数定义 |
| ④ 中间件 | 拦截每次 tool_call，打日志、做基础校验（可观察 + 可干预） | `ToolMonitorMiddleware` |
| ⑤ 容错返回 | 出错时返回引导性错误信息，帮助 LLM 自我纠正 | 每个工具的 `except` 分支 |

## 架构概览

```
用户输入 → Agent (LLM + Tools + Middleware) → 工具调用 → 返回结果
                ↑                                    ↓
           会话记忆 (Redis) ←── 消息持久化 ──────────┘
                ↓
         Gatekeeper (可选) → 结构化记忆条目
```

## 核心模块

### 1. LLM 配置 (`_build_llm`)

从环境变量构造 `ChatOpenAI` 实例：

| 环境变量 | 说明 | 默认值 |
|----------|------|--------|
| `LLM_BASE_URL` | API 地址 | `https://api.openai.com/v1` |
| `LLM_API_KEY` | API 密钥 | - |
| `LLM_MODEL_NAME` | 模型名称 | `gpt-4o-mini` |

关键设置：`temperature=0.0`（减少随机性，提高工具调用一致性）、`max_tokens=1024`。

### 2. 工具集

#### `calculator` — 数学计算

- **功能**：执行数学表达式求值，支持 `math` 模块函数（`sqrt`、`cos`、`sin` 等）
- **安全**：使用受限命名空间 `eval()`，禁止内置函数
- **容错**：区分 `SyntaxError`（语法错误）和数学错误（除零、负数开方），给出针对性修正建议

#### `get_current_datetime` — 日期时间

- **功能**：返回当前 UTC 时间、本地时间和时区
- **无参数**：不需要任何输入

#### `word_count` — 文本统计

- **功能**：统计文本的行数、词数、字符数（含/不含空格）
- **关键约束**：参数必须是用户提供的原始文本，不可自行生成

#### `json_formatter` — JSON 格式化

- **功能**：将 JSON 字符串美化缩进输出
- **容错**：解析失败时给出常见错误提示（单引号、尾部逗号、key 未加引号）

#### `get_weather` — 天气查询

- **数据源**：[wttr.in](https://wttr.in) 免费天气服务
- **输出**：当前天气（温度、体感温度、湿度、风力、能见度、紫外线）+ 未来 3 天预报
- **校验**：拒绝空城市名、含无关后缀的输入（如 "北京天气"）、超长输入

#### `web_scraper` — 网页抓取

- **功能**：抓取指定 URL 的 HTML 页面，提取标题、描述、正文
- **解析库**：BeautifulSoup + httpx（15 秒超时，自动跟随重定向）
- **处理流程**：
  1. URL 安全校验（协议、域名）
  2. HTTP 请求（带 User-Agent）
  3. Content-Type 检查（仅处理 HTML）
  4. 移除噪音标签（script、style、nav、footer 等）
  5. 提取标题（`<title>` → `<h1>` → `og:title` → URL fallback）
  6. 提取描述（`meta description` → `og:description`）
  7. 正文截断至 3000 字符

### 3. 系统提示词 (`SYSTEM_PROMPT`)

定义工具选择原则与边界歧义消解规则：

- **天气 vs 常识**："北京人口" → 直接回答；"北京天气" → 调用 `get_weather`
- **计算 vs 陈述**："2+3*4" → 调用 `calculator`；"我今年 25 岁" → 不调工具
- **统计 vs 咨询**：给定文本 + "统计字数" → 调用 `word_count`；"怎么统计字数" → 直接解释
- **时间 vs 天气**："现在几点" → `get_current_datetime`；"今天天气" → `get_weather`
- **抓取 vs 追问**：给定 URL → `web_scraper`；只提网站名 → 追问链接
- **感叹 vs 查询**："今天好热啊" → 不调工具，友好回应

### 4. 中间件 (`ToolMonitorMiddleware`)

继承 `AgentMiddleware`，在每次工具调用前后插入逻辑：

- **执行前**：记录工具名称和参数（INFO 日志）
- **预警规则**（只记录日志，不拦截）：
  - `calculator`：空表达式或表达式过长
  - `get_weather`：空城市名或城市名含无关后缀
  - `json_formatter`：空输入或输入不像是 JSON
  - `word_count`：空文本
  - `get_current_datetime`：不应有参数却收到了参数
- **执行后**：记录返回结果摘要和状态

### 5. 会话记忆

#### Redis 会话记忆 (`build_memory`)

- **存储**：`RedisSessionStore`，key 前缀 `chat:session:{session_id}`
- **TTL**：由 `REDIS_SESSION_TTL` 环境变量控制（默认 3600 秒）
- **加密**：可选的消息加密/解密（通过 Fernet 对称加密），依赖 `app.core.memory.crypto`
- **降级**：Redis 不可用时自动降级为无记忆模式

#### Gatekeeper 结构化记忆（可选）

通过 `GATEKEEPER_ENABLED=true` 环境变量启用：

- **功能**：在每轮对话后自动提取结构化记忆条目（事实、偏好、决策等）
- **分类器**：复用 Agent 的 LLM 实例对消息进行分类
- **存储**：独立的 Redis 实例，key 前缀 `mem:entry`，默认 TTL 30 天
- **工具信息**：每轮提取工具调用摘要（名称、参数、结果、成功/失败），用于辅助分类

### 6. 交互式对话循环 (`main`)

#### 特殊命令

| 命令 | 功能 |
|------|------|
| `/clear` | 清除当前会话记忆（含 Gatekeeper 条目） |
| `/memory` | 查看原始消息窗口状态（消息数、轮数、可信度等） |
| `/entries` | 查看 Gatekeeper 结构化记忆条目列表 |
| `/help` | 显示帮助信息 |
| `quit` / `exit` / `q` | 退出程序 |

#### 对话流程

```
1. 用户输入 → 加载历史消息（Redis）→ 追加当前消息
2. Agent.invoke() → LLM 决策 → 工具调用（经 Middleware 监控）
3. 持久化所有消息到 Redis
4. Gatekeeper 提取结构化记忆条目（如启用）
5. 输出最终回答 + 工具调用统计 + 记忆状态
```

## 依赖

- `langchain` / `langchain-openai`：Agent 框架
- `langchain.agents.create_agent`：Agent 构建 API
- `httpx`：HTTP 客户端（天气查询、网页抓取）
- `beautifulsoup4`：HTML 解析
- `redis`：会话记忆存储
- `pydantic`：参数 schema 定义
- `python-dotenv`：环境变量加载
- `cryptography` (Fernet)：消息加密（可选）

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `LLM_BASE_URL` | LLM API 地址 | `https://api.openai.com/v1` |
| `LLM_API_KEY` | LLM API 密钥 | - |
| `LLM_MODEL_NAME` | 模型名称 | `gpt-4o-mini` |
| `REDIS_URL` | Redis 连接地址 | `redis://localhost:6379/0` |
| `REDIS_PASSWORD` | Redis 密码 | - |
| `REDIS_SESSION_TTL` | 会话过期时间（秒） | `3600` |
| `REDIS_ENTRY_TTL` | Gatekeeper 条目过期时间（秒） | `2592000`（30 天） |
| `GATEKEEPER_ENABLED` | 启用结构化记忆 | `false` |
