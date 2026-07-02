"""基于 LangGraph 的 Agentic RAG 图实现。

图结构：
    rewrite_query ──→ retrieve ──→ judge_relevance ──→ generate_answer
                         ↑                │
                         │   [不相关]      │ [相关]
                         └────────────────┘

核心能力：
- 查询改写：将用户口语化问题改写为更适合向量检索的形式
- 多路检索：结合 dense + BM25 的混合检索，并对结果去重、重排序
- 答案评判：由 LLM 判断检索结果是否足以回答问题
- 自纠正：若评判为不相关，回退到改写节点重新检索（最多 N 轮）
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal

from langgraph.graph import END, StateGraph
from langgraph.graph.state import CompiledStateGraph
from typing_extensions import TypedDict

from app.rag.vector_store import SearchResult  # 运行时需要（LangGraph get_type_hints 解析 TypedDict）

if TYPE_CHECKING:
    from app.llm.chat_model import ChatMessage
    from app.rag.hybrid_retriever import HybridRetriever
    from app.rag.query_rewriter import QueryRewriter, RewriteResult
    from app.rag.reranker import Reranker
    from app.rag.retriever import Retriever

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgenticRAGState(TypedDict, total=False):
    """LangGraph 状态，在节点间传递。

    字段说明：
    - query: 用户原始问题
    - rewritten_queries: 改写后的查询列表（支持多路检索）
    - rewrite_keywords: 改写时提取的关键词（供 BM25 使用）
    - rewrite_attempts: 当前改写尝试次数
    - rewrite_feedback: 上一轮评判给出的改写建议（自纠正时使用）
    - contexts: 检索到的文档片段列表
    - is_relevant: 检索结果是否足以回答问题
    - relevance_reason: 相关性评判的理由
    - answer: 最终生成的答案
    - messages: 节点间传递的中间消息（用于调试）
    """

    query: str
    rewritten_queries: list[str]
    rewrite_keywords: list[str]
    rewrite_attempts: int
    rewrite_feedback: str
    contexts: list[SearchResult]
    is_relevant: bool
    relevance_reason: str
    answer: str
    messages: list[str]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_agentic_rag_graph(
    *,
    chat_model_fn: Any,
    retriever: Retriever | HybridRetriever,
    reranker: Reranker,
    query_rewriter: QueryRewriter | None = None,
    max_rewrite_attempts: int = 2,
    retrieval_k: int = 10,
    rerank_top_n: int = 5,
    score_threshold: float | None = None,
    verbose: bool = False,
) -> CompiledStateGraph:
    """构建 Agentic RAG LangGraph 图。

    Args:
        chat_model_fn: LLM 调用函数，签名为 (messages: list[ChatMessage]) -> str。
        retriever: 检索器实例（支持 Retriever 或 HybridRetriever）。
        reranker: 重排序器实例。
        query_rewriter: 查询改写器实例（可选，为 None 时跳过改写步骤）。
        max_rewrite_attempts: 最大改写尝试次数（自纠正循环上限）。
        retrieval_k: 检索时返回的文档片段数。
        rerank_top_n: 重排序后保留的文档片段数。
        score_threshold: 检索分数阈值（可选）。
        verbose: 是否打印详细的状态流转日志（默认 False）。

    Returns:
        编译后的 LangGraph 图，可通过 .invoke({"query": "..."}) 执行。
    """

    # ---- 状态流转日志工具 ---------------------------------------------------

    _ROUND = 0  # 用闭包变量追踪轮次

    def _state_snapshot(state: AgenticRAGState) -> list[str]:
        """提取状态关键字段的快照，返回格式化行列表。"""
        lines = []
        fields = [
            ("query", "原始问题"),
            ("rewritten_queries", "改写查询"),
            ("rewrite_keywords", "关键词"),
            ("rewrite_attempts", "改写轮次"),
            ("rewrite_feedback", "改写反馈"),
            ("contexts", "检索结果"),
            ("is_relevant", "是否相关"),
            ("relevance_reason", "评判理由"),
            ("answer", "最终答案"),
        ]
        for key, label in fields:
            val = state.get(key)
            if val is None or val == [] or val == "" or val == 0:
                continue
            if key == "contexts":
                scores = ", ".join(f"{c.score:.4f}" for c in val[:5])
                lines.append(f"  {label:12s}: {len(val)} docs [scores: {scores}]")
            elif key == "answer":
                lines.append(f"  {label:12s}: {str(val)[:120]}")
            elif key == "rewritten_queries":
                lines.append(f"  {label:12s}: {val}")
            else:
                lines.append(f"  {label:12s}: {val}")
        return lines

    def _log_enter(node_name: str, state: AgenticRAGState) -> None:
        """打印进入节点时的状态。"""
        if not verbose:
            return
        nonlocal _ROUND
        _ROUND = state.get("rewrite_attempts", 0)
        round_tag = f" (Round {_ROUND})" if node_name == "rewrite_query" else ""
        print(f"\n{'='*60}")
        print(f">>> Enter [{node_name}]{round_tag}")
        print(f"{'='*60}")
        for line in _state_snapshot(state):
            print(line)

    def _log_exit(node_name: str, updates: dict[str, Any]) -> None:
        """打印离开节点时的状态变更。"""
        if not verbose:
            return
        field_labels = {
            "rewritten_queries": "改写查询",
            "rewrite_keywords": "关键词",
            "rewrite_attempts": "改写轮次",
            "rewrite_feedback": "改写反馈",
            "contexts": "检索结果",
            "is_relevant": "是否相关",
            "relevance_reason": "评判理由",
            "answer": "最终答案",
        }
        print(f"{'-'*60}")
        print(f"<<< Exit [{node_name}]")
        for key, val in updates.items():
            if key in ("messages",):
                continue
            label = field_labels.get(key, key)
            if key == "contexts":
                scores = ", ".join(f"{c.score:.4f}" for c in val[:5]) if val else "empty"
                print(f"  -> {label}: {len(val)} docs [scores: {scores}]")
            elif key == "answer":
                print(f"  -> {label}: {str(val)[:120]}")
            elif key == "rewritten_queries":
                print(f"  -> {label}: {val}")
            else:
                print(f"  -> {label}: {val}")
        print()

    def node_rewrite_query(state: AgenticRAGState) -> dict[str, Any]:
        """查询改写节点。"""
        _log_enter("rewrite_query", state)

        query = state["query"]
        attempts = state.get("rewrite_attempts", 0) + 1
        feedback = state.get("rewrite_feedback", "")
        messages: list[str] = list(state.get("messages", []))

        if query_rewriter is None:
            logger.info("[rewrite_query] 未配置 QueryRewriter，使用原始查询")
            messages.append(f"[第 {attempts} 轮] 未配置改写器，使用原始查询：{query}")
            result = {
                "rewritten_queries": [query],
                "rewrite_keywords": [],
                "rewrite_attempts": attempts,
                "rewrite_feedback": "",
                "messages": messages,
            }
            _log_exit("rewrite_query", result)
            return result

        # 若有上一轮的反馈，将其拼接到查询中引导改写
        effective_query = query
        if feedback:
            effective_query = (
                f"原始问题：{query}\n"
                f"上一轮检索效果不佳，请改进查询以获取更相关的文档。反馈：{feedback}"
            )
            logger.info("[rewrite_query] 自纠正模式，反馈：%s", feedback)

        try:
            rewrite_result: RewriteResult = query_rewriter.rewrite(effective_query)
            rewritten = rewrite_result.queries if rewrite_result.queries else [query]
            keywords = rewrite_result.keywords
            logger.info(
                "[rewrite_query] 第 %d 轮改写：queries=%s  keywords=%s",
                attempts, rewritten, keywords,
            )
            messages.append(
                f"[第 {attempts} 轮] 改写查询：{rewritten}，关键词：{keywords}"
            )
        except Exception:
            logger.exception("[rewrite_query] 改写失败，回退到原始查询")
            rewritten = [query]
            keywords = []
            messages.append(f"[第 {attempts} 轮] 改写失败，回退到原始查询")

        result = {
            "rewritten_queries": rewritten,
            "rewrite_keywords": keywords,
            "rewrite_attempts": attempts,
            "rewrite_feedback": "",
            "messages": messages,
        }
        _log_exit("rewrite_query", result)
        return result

    def node_retrieve(state: AgenticRAGState) -> dict[str, Any]:
        """检索节点。"""
        _log_enter("retrieve", state)

        queries = state.get("rewritten_queries", [state["query"]])
        keywords = state.get("rewrite_keywords", [])
        messages: list[str] = list(state.get("messages", []))

        # 多查询去重检索
        all_results: list[SearchResult] = []
        per_query_k = max(3, retrieval_k // max(len(queries), 1))

        for q in queries:
            try:
                batch = retriever.retrieve(
                    q,
                    top_k=per_query_k,
                    keywords=keywords if keywords else None,
                    score_threshold=score_threshold,
                )
                all_results.extend(batch)
                logger.info("[retrieve] 查询 '%s' 返回 %d 条结果", q[:50], len(batch))
            except Exception:
                logger.exception("[retrieve] 查询 '%s' 检索失败", q[:50])

        # 去重（保留最高分）
        seen: dict[str, SearchResult] = {}
        for r in all_results:
            fp = r.text[:200]
            if fp not in seen or r.score > seen[fp].score:
                seen[fp] = r
        deduped = sorted(seen.values(), key=lambda r: r.score, reverse=True)[:retrieval_k]

        # 重排序
        if deduped:
            deduped = reranker.rerank(
                state["query"], deduped, top_n=rerank_top_n
            )

        messages.append(
            f"[检索] 去重后 {len(seen)} 条 → rerank 后保留 {len(deduped)} 条"
        )
        logger.info("[retrieve] 最终返回 %d 条文档", len(deduped))

        result = {
            "contexts": deduped,
            "messages": messages,
        }
        _log_exit("retrieve", result)
        return result

    def node_judge_relevance(state: AgenticRAGState) -> dict[str, Any]:
        """答案评判节点。"""
        _log_enter("judge_relevance", state)

        query = state["query"]
        contexts = state.get("contexts", [])
        messages: list[str] = list(state.get("messages", []))

        if not contexts:
            logger.info("[judge_relevance] 无检索结果，判定为不相关")
            messages.append("[评判] 无检索结果，需要重新检索")
            result = {
                "is_relevant": False,
                "relevance_reason": "检索结果为空",
                "rewrite_feedback": "检索结果为空，请尝试更通用的关键词或同义词改写",
                "messages": messages,
            }
            _log_exit("judge_relevance", result)
            return result

        # 构建评判 prompt
        context_text = _format_contexts_simple(contexts)
        judge_prompt: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "你是一个检索质量评判助手。判断给定的检索文档是否包含足够的信息"
                    "来回答用户问题。\n\n"
                    "输出格式（严格 JSON）：\n"
                    '{"relevant": true/false, "reason": "评判理由", '
                    '"feedback": "若不相关，给出改写建议；若相关，写空字符串"}'
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{query}\n\n检索文档：\n{context_text}\n\n"
                    "请判断这些文档是否足以回答问题。"
                ),
            },
        ]

        try:
            raw = chat_model_fn(judge_prompt)
            logger.info("[judge_relevance] LLM 评判结果：%s", raw[:200])

            # 解析 JSON 输出
            import json
            # 提取可能的 JSON 块
            json_match = _extract_json(raw)
            if json_match:
                parsed = json.loads(json_match)
                is_relevant = bool(parsed.get("relevant", False))
                reason = parsed.get("reason", "")
                feedback = parsed.get("feedback", "")
            else:
                # 回退：简单判断
                is_relevant = True
                reason = "无法解析评判结果，默认认为相关"
                feedback = ""

            messages.append(
                f"[评判] 相关={is_relevant}，理由：{reason[:100]}"
            )
            result = {
                "is_relevant": is_relevant,
                "relevance_reason": reason,
                "rewrite_feedback": feedback,
                "messages": messages,
            }
            _log_exit("judge_relevance", result)
            return result
        except Exception:
            logger.exception("[judge_relevance] 评判失败，默认认为相关")
            messages.append("[评判] 评判调用失败，默认认为相关")
            result = {
                "is_relevant": True,
                "relevance_reason": "评判调用异常，默认通过",
                "rewrite_feedback": "",
                "messages": messages,
            }
            _log_exit("judge_relevance", result)
            return result

    def node_generate_answer(state: AgenticRAGState) -> dict[str, Any]:
        """答案生成节点。"""
        _log_enter("generate_answer", state)

        query = state["query"]
        contexts = state.get("contexts", [])
        messages: list[str] = list(state.get("messages", []))

        if not contexts:
            messages.append("[生成] 无上下文，返回兜底回复")
            result = {
                "answer": "知识库中未找到相关信息，请尝试换个方式提问。",
                "messages": messages,
            }
            _log_exit("generate_answer", result)
            return result

        context_text = _format_contexts_simple(contexts)
        gen_prompt: list[ChatMessage] = [
            {
                "role": "system",
                "content": (
                    "你是企业知识库问答助手。必须只基于提供的检索上下文回答问题。"
                    "如果上下文中没有相关信息，请明确说明。不要编造上下文以外的事实。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"用户问题：{query}\n\n"
                    f"检索上下文：\n{context_text}\n\n"
                    "请给出简洁、准确的答案，并在答案中尽量体现来源依据。"
                ),
            },
        ]

        try:
            answer = chat_model_fn(gen_prompt)
            logger.info("[generate_answer] 生成答案长度：%d 字", len(answer))
            messages.append(f"[生成] 答案长度 {len(answer)} 字")
            result = {"answer": answer, "messages": messages}
            _log_exit("generate_answer", result)
            return result
        except Exception:
            logger.exception("[generate_answer] 生成失败")
            messages.append("[生成] 生成调用失败")
            result = {
                "answer": "抱歉，答案生成失败，请稍后重试。",
                "messages": messages,
            }
            _log_exit("generate_answer", result)
            return result

    # ---- 条件路由 -----------------------------------------------------------

    def route_after_judge(state: AgenticRAGState) -> Literal["rewrite", "generate"]:
        """评判后的路由决策: 相关 -> 生成, 不相关且未超限 -> 改写, 超限 -> 强制生成."""
        is_relevant = state.get("is_relevant", True)
        attempts = state.get("rewrite_attempts", 0)

        if is_relevant:
            logger.info("[route] 检索结果相关 → 生成答案")
            return "generate"

        if attempts < max_rewrite_attempts:
            logger.info(
                "[route] 检索结果不相关，第 %d/%d 轮 → 重新改写",
                attempts, max_rewrite_attempts,
            )
            return "rewrite"

        logger.info(
            "[route] 已达到最大改写次数 %d，强制生成答案", max_rewrite_attempts
        )
        return "generate"

    # ---- 组装图 -------------------------------------------------------------

    graph = StateGraph(AgenticRAGState)

    # 添加节点
    graph.add_node("rewrite_query", node_rewrite_query)
    graph.add_node("retrieve", node_retrieve)
    graph.add_node("judge_relevance", node_judge_relevance)
    graph.add_node("generate_answer", node_generate_answer)

    # 添加边
    graph.set_entry_point("rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "judge_relevance")

    # 条件边：评判后路由
    graph.add_conditional_edges(
        "judge_relevance",
        route_after_judge,
        {
            "rewrite": "rewrite_query",
            "generate": "generate_answer",
        },
    )

    # 终止边
    graph.add_edge("generate_answer", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_contexts_simple(contexts: list[SearchResult]) -> str:
    """将检索结果格式化为简洁的文本块（供评判和生成 prompt 使用）。"""
    blocks: list[str] = []
    for i, ctx in enumerate(contexts, 1):
        meta = ctx.metadata
        blocks.append(
            f"[文档{i}] "
            f"来源={meta.get('file_name', '?')} "
            f"分数={ctx.score:.4f}\n"
            f"{ctx.text[:500]}"  # 截断避免 prompt 过长
        )
    return "\n\n".join(blocks)


def _extract_json(text: str) -> str | None:
    """从 LLM 原始输出中提取 JSON 块。"""
    import re

    # 尝试匹配 ```json ... ``` 代码块
    m = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if m:
        return m.group(1).strip()

    # 尝试匹配 { ... } 块
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        return m.group(0).strip()

    return None
