"""LLM 记忆分类器。

使用 LLM 分析每轮对话，提取值得长期记住的结构化记忆条目。
输出带可信度分数和分类理由的候选列表，供 Gatekeeper 做最终决策。
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable

from app.core.memory.entry_models import (
    ALL_MEMORY_ENTRY_TYPES,
    MemoryCandidate,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt 模板
# ---------------------------------------------------------------------------

MEMORY_EXTRACTION_SYSTEM = """你是一个记忆提取分类系统。分析用户的对话，从中提取值得长期记住的信息。

# 记忆类型定义

你需要判断对话中是否包含以下 6 种值得记忆的信息：

1. **preference** (长期偏好)
   - 用户明确表达的个人偏好、风格选择、习惯要求
   - 例: "我喜欢简洁的回答，不要长篇大论"
   - 例: "以后回复用中文"

2. **work_habit** (工作习惯)
   - 用户反复出现的工作模式、开发习惯、流程偏好
   - 例: "我们每次写 API 都要加 Swagger 注解"
   - 例: "代码提交前必须跑单元测试"

3. **business_config** (业务配置)
   - 稳定的业务系统配置、部署信息、架构决策
   - 例: "生产服务器在阿里云上海区"
   - 例: "数据库用的 PostgreSQL 16"

4. **experience** (有效经验)
   - 任务执行中得到的有效经验、技巧、最佳实践
   - 例: "搜索技术文档时用 hybrid 检索比纯向量检索更准"
   - 例: "用 wttr.in 查天气比直接搜索快"

5. **correction** (用户纠正)
   - 用户明确指出 Agent 的错误并给出正确信息
   - 例: "不对，这个功能是 v2.0 引入的，不是 v1.0"
   - 例: "你刚才算错了，结果是 42 不是 24"

6. **fix_strategy** (修复策略)
   - Agent 工具调用失败后的有效修复方法
   - 例: "calculator 报语法错误时，先检查括号匹配，再用 eval"
   - 例: "天气查询失败时，让用户确认城市名拼写后再重试"

# 不应提取的情况

以下内容**必须**输出 discard：

- 一次性临时问题：问天气、问时间、问"今天几号"
- 明显会过期的信息：当前新闻、实时数据
- 敏感信息：密码、token、API key (sk-*)、身份证号、手机号
- 未确认的推测：Agent 猜测但用户没有确认的内容
- 低可信度结论：confidence < 0.4 的内容
- 对业务系统的错误分析：被用户纠正前 Agent 的错误推理
- 纯闲聊：打招呼、寒暄、无信息量的对话

# 输出格式

严格输出 JSON 数组，每个元素包含：
- entry_type: 记忆类型 (preference | work_habit | business_config | experience | correction | fix_strategy | discard)
- content: 记忆正文 (直接引用或精炼用户原话，≤ 500 字)
- summary: 一句话摘要 (≤ 100 字)
- confidence: 可信度 0.0~1.0 (用户直接陈述=0.9+，强推论=0.7-0.9，弱推论=0.4-0.7)
- reason: 分类理由 (简短说明为什么这样分类)

如果没有值得记住的内容，输出空数组 []。
不确定是否该记住时，宁可输出 discard 也不要漏掉敏感信息。

# Few-shot 示例

示例 1 — 长期偏好:
用户: "以后回答尽量控制在200字以内，我不喜欢太长的解释"
输出: [{"entry_type": "preference", "content": "用户偏好简洁回复，要求回答控制在200字以内", "summary": "用户要求简短回答≤200字", "confidence": 0.95, "reason": "用户明确表达了格式偏好，属于长期偏好"}]

示例 2 — 工作习惯:
用户: "我们团队规范要求所有 API 都要有 rate limiting"
输出: [{"entry_type": "work_habit", "content": "团队规范：所有 API 必须配置 rate limiting", "summary": "团队要求API限流", "confidence": 0.9, "reason": "用户描述了团队开发规范，属于工作习惯"}]

示例 3 — 临时问题，应丢弃:
用户: "今天北京天气怎么样？"
输出: []

示例 4 — 敏感信息，应丢弃:
用户: "我的 API key 是 sk-abc123def456"
输出: [{"entry_type": "discard", "content": "", "summary": "含API key", "confidence": 0.0, "reason": "包含敏感信息 sk-* API key，不应存储"}]

示例 5 — 用户纠正:
用户: "不对，我刚才说的项目名称是 Phoenix，不是 Phoenix2"
输出: [{"entry_type": "correction", "content": "项目名称是 Phoenix（不是 Phoenix2）", "summary": "项目名称为Phoenix", "confidence": 0.95, "reason": "用户明确纠正了Agent的错误，应记住正确信息"}]

示例 6 — 修复策略:
(工具 calculator 返回了语法错误后)
用户: "你应该先去掉表达式里的空格和特殊字符再计算"
输出: [{"entry_type": "fix_strategy", "content": "使用 calculator 前先清理表达式：去掉空格和特殊字符", "summary": "calculator使用前需清理表达式", "confidence": 0.85, "reason": "用户提供了工具调用失败后的修复方法"}]

示例 7 — 有效经验:
用户: "我发现用 site:docs.python.org 搜索 Python 文档比直接搜快很多"
输出: [{"entry_type": "experience", "content": "搜索 Python 文档时用 site:docs.python.org 限定来源效率更高", "summary": "Python文档搜索用site限定来源", "confidence": 0.85, "reason": "用户分享了搜索经验技巧，可复用"}]"""

MEMORY_EXTRACTION_USER = """分析以下对话，提取值得长期记住的信息。

用户消息: {user_message}

助手回复: {assistant_message}

工具调用情况: {tool_info}

请严格按照 JSON 数组格式输出。如果没有值得记住的内容，输出 []。"""


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class MemoryClassifier:
    """使用 LLM 从对话中提取结构化记忆候选。

    典型用法::

        llm = ChatOpenAI(...)
        classifier = MemoryClassifier(lambda prompt: llm.invoke(prompt).content)
        candidates = classifier.extract(user_msg, asst_msg, tool_info)
    """

    def __init__(
        self,
        llm_func: Callable[[str], str],
        *,
        system_prompt: str | None = None,
    ) -> None:
        """初始化分类器。

        Args:
            llm_func: 接受 prompt 字符串、返回响应字符串的可调用对象。
            system_prompt: 自定义系统提示词，为 None 时使用内置模板。
        """
        self._llm = llm_func
        self._system = system_prompt or MEMORY_EXTRACTION_SYSTEM

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(
        self,
        user_message: str,
        assistant_message: str,
        tool_info: str = "",
    ) -> list[MemoryCandidate]:
        """从一轮对话中提取结构化记忆候选。

        Args:
            user_message: 用户本轮输入。
            assistant_message: Agent 本轮最终回复。
            tool_info: 工具调用摘要文本（可选）。

        Returns:
            MemoryCandidate 列表，不含 discard 类型的候选。
        """
        prompt = self._build_prompt(user_message, assistant_message, tool_info)
        try:
            raw = self._llm(prompt)
        except Exception as exc:
            logger.error("CLASSIFIER  LLM call failed: %s", exc)
            return []

        candidates = self._parse_response(raw)
        kept = [c for c in candidates if not c.should_discard]
        if kept:
            logger.info(
                "CLASSIFIER  %d candidates → %d kept (%d discarded)",
                len(candidates),
                len(kept),
                len(candidates) - len(kept),
            )
        else:
            logger.debug("CLASSIFIER  no memorable content found")
        return kept

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        user_message: str,
        assistant_message: str,
        tool_info: str,
    ) -> str:
        """构造完整的 LLM prompt（system + user 拼接为单条消息）。"""
        tool_text = tool_info if tool_info else "（本轮无工具调用）"
        user_prompt = MEMORY_EXTRACTION_USER.format(
            user_message=user_message,
            assistant_message=assistant_message,
            tool_info=tool_text,
        )
        # 对 OpenAI 兼容接口，将 system 和 user 拼接
        return f"{self._system}\n\n{user_prompt}"

    @staticmethod
    def _parse_response(raw: str) -> list[MemoryCandidate]:
        """解析 LLM 返回的 JSON 数组。

        容错策略：
        1. 直接 json.loads
        2. 从文本中提取首个 JSON 数组
        3. 逐对象提取（应对截断）
        4. 全部失败返回空列表
        """
        candidates: list[MemoryCandidate] = []

        # 策略 1: 直接解析
        try:
            data = json.loads(raw.strip())
            if isinstance(data, list):
                return MemoryClassifier._validate_candidates(data)
        except json.JSONDecodeError:
            pass

        # 策略 2: 提取 ```json ... ``` 或首个 [ ... ]
        import re

        # 尝试提取代码块中的 JSON
        m = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(1))
                if isinstance(data, list):
                    return MemoryClassifier._validate_candidates(data)
            except json.JSONDecodeError:
                pass

        # 策略 3: 查找文本中第一个 [ 和最后一个 ]
        start = raw.find("[")
        end = raw.rfind("]")
        if start != -1 and end > start:
            try:
                data = json.loads(raw[start:end + 1])
                if isinstance(data, list):
                    return MemoryClassifier._validate_candidates(data)
            except json.JSONDecodeError:
                pass

        # 策略 4: 逐个匹配 JSON 对象 { ... }
        for obj_match in re.finditer(r"\{[^{}]*\}", raw):
            try:
                obj = json.loads(obj_match.group())
                candidate = MemoryCandidate(**obj)
                candidates.append(candidate)
            except (json.JSONDecodeError, ValueError):
                continue

        if not candidates:
            logger.warning("CLASSIFIER  failed to parse response: %.200s...", raw)

        return candidates

    @staticmethod
    def _validate_candidates(items: list[dict[str, Any]]) -> list[MemoryCandidate]:
        """批量校验并构造 MemoryCandidate 列表。

        对 LLM 返回的异常字段做容错修复，避免因格式微小偏差丢弃有效候选。
        """
        candidates: list[MemoryCandidate] = []
        for item in items:
            try:
                # ── 容错修复 ──────────────────────────────────
                # 1. confidence 负值 → 取绝对值
                if "confidence" in item and isinstance(item["confidence"], (int, float)):
                    if item["confidence"] < 0:
                        item["confidence"] = abs(item["confidence"])
                        logger.debug("CLASSIFIER  fixed negative confidence → %.2f", item["confidence"])
                    elif item["confidence"] > 1.0:
                        item["confidence"] = min(item["confidence"], 1.0)

                # 2. entry_type 非法值 → 尝试修正常见拼写错误
                if "entry_type" in item and isinstance(item["entry_type"], str):
                    raw_type = item["entry_type"].strip().lower()
                    # 常见 LLM 输出偏差
                    type_aliases = {
                        "user_preference": "preference",
                        "habit": "work_habit",
                        "config": "business_config",
                        "knowledge": "experience",
                        "error_correction": "correction",
                        "tool_fix": "fix_strategy",
                    }
                    if raw_type in type_aliases:
                        item["entry_type"] = type_aliases[raw_type]
                        logger.debug("CLASSIFIER  fixed entry_type %s → %s", raw_type, item["entry_type"])

                candidates.append(MemoryCandidate(**item))
            except ValueError as exc:
                logger.warning("CLASSIFIER  invalid candidate skipped: %s", exc)
        return candidates
