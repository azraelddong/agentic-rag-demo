"""Agentic RAG 演示运行器。

可作为独立脚本运行，展示 LangGraph Agentic RAG 图的完整工作流程。

用法：
    # 基础用法
    python -m app.agent.demo.demo_runner "什么是RAG？"

    # 指定检索参数
    python -m app.agent.demo.demo_runner "什么是RAG？" --retrieval-k 15 --rerank-top-n 8

    # 详细模式（打印完整状态流转）
    python -m app.agent.demo.demo_runner "什么是RAG？" --verbose

    # 禁用查询改写
    python -m app.agent.demo.demo_runner "什么是RAG？" --no-rewrite

    # 交互式模式
    python -m app.agent.demo.demo_runner --interactive
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.demo.agentic_rag_graph import AgenticRAGState, build_agentic_rag_graph
from app.api.dependencies import (
    get_chat_model,
    get_query_rewriter,
    get_reranker,
    get_retriever,
    get_settings,
)

logger = logging.getLogger(__name__)

"""构建图"""
def build_graph(
    *,
    retrieval_k: int = 10,
    rerank_top_n: int = 5,
    max_rewrite_attempts: int = 2,
    score_threshold: float | None = None,
    enable_rewrite: bool = True,
    verbose: bool = False,
) -> Any:
    """组装 Agentic RAG 图。

    复用项目已有的依赖注入组件，将各模块注入到 LangGraph 图中。
    """
    settings = get_settings()
    chat_model = get_chat_model()
    retriever = get_retriever()
    reranker = get_reranker()
    query_rewriter = get_query_rewriter() if enable_rewrite else None

    return build_agentic_rag_graph(
        chat_model_fn=chat_model.generate,
        retriever=retriever,
        reranker=reranker,
        query_rewriter=query_rewriter,
        max_rewrite_attempts=max_rewrite_attempts,
        retrieval_k=retrieval_k,
        rerank_top_n=rerank_top_n,
        score_threshold=score_threshold or settings.rag_score_threshold,
        verbose=verbose,
    )


def run_query(
    graph: Any,
    question: str,
    *,
    verbose: bool = False,
) -> AgenticRAGState:
    """执行单次 Agentic RAG 查询。

    Args:
        graph: 编译后的 LangGraph 图。
        question: 用户问题。
        verbose: 是否打印详细状态流转。

    Returns:
        最终的 AgenticRAGState（包含 answer、contexts、messages 等字段）。
    """
    initial_state: AgenticRAGState = {
        "query": question,
    }

    if verbose:
        print("=" * 60)
        print(f"[Query] {question}")
        print("=" * 60)
        print("[LangGraph] State flow log:\n")

    # invoke 获取最终状态，verbose 时节点内部 _log_enter/_log_exit 已打印状态流转
    final_state: AgenticRAGState = graph.invoke(initial_state)

    return final_state


def _print_step(step_output: dict[str, Any]) -> None:
    """格式化打印一个图步骤的输出。"""
    for node_name, node_state in step_output.items():
        print(f"--- [Node] {node_name} ---")

        # 打印关键字段
        if "rewritten_queries" in node_state:
            print(f"   改写查询：{node_state['rewritten_queries']}")
        if "rewrite_attempts" in node_state:
            print(f"   改写轮次：{node_state['rewrite_attempts']}")
        if "contexts" in node_state:
            ctx_count = len(node_state["contexts"])
            print(f"   检索文档数：{ctx_count}")
            if ctx_count > 0:
                scores = [f"{c.score:.4f}" for c in node_state["contexts"][:5]]
                print(f"   Top-5 分数：{scores}")
        if "is_relevant" in node_state:
            status = "[OK] relevant" if node_state["is_relevant"] else "[X] not relevant"
            print(f"   relevance judge: {status}")
            reason = node_state.get("relevance_reason", "")
            if reason:
                print(f"   评判理由：{reason[:120]}")
        if "rewrite_feedback" in node_state:
            fb = node_state["rewrite_feedback"]
            if fb:
                print(f"   改写反馈：{fb[:120]}")
        if "answer" in node_state:
            ans = node_state["answer"]
            print(f"   最终答案：{ans[:200]}{'...' if len(ans) > 200 else ''}")
        print()


def print_result(state: AgenticRAGState) -> None:
    """格式化打印最终结果。"""
    print("\n" + "=" * 60)
    print("Final Result")
    print("=" * 60)

    # 答案
    answer = state.get("answer", "")
    print(f"\n[Answer]\n{answer}")

    # 检索来源
    contexts = state.get("contexts", [])
    if contexts:
        print(f"\n[Sources] ({len(contexts)} docs):")
        for i, ctx in enumerate(contexts, 1):
            meta = ctx.metadata
            print(
                f"  [{i}] {meta.get('file_name', '?')} "
                f"chunk={meta.get('chunk_index', '?')} "
                f"score={ctx.score:.4f}"
            )
            print(f"      {ctx.text[:100]}...")

    # 统计信息
    messages = state.get("messages", [])
    attempts = state.get("rewrite_attempts", 0)
    rel_status = "[OK] relevant" if state.get("is_relevant") else "[X] not relevant"
    print(f"\n[Stats] rewrite rounds: {attempts}, node messages: {len(messages)}")
    print(f"        relevance: {rel_status}")


def interactive_mode(
    graph: Any,
    *,
    verbose: bool = False,
) -> None:
    """交互式问答模式。"""
    print("\n" + "=" * 60)
    print("Agentic RAG Interactive Mode")
    print("Type your question and press Enter. 'quit'/'exit' to leave, 'verbose' to toggle detail mode.")
    print("=" * 60 + "\n")

    while True:
        try:
            question = input("Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break
        if question.lower() == "verbose":
            verbose = not verbose
            print(f"Verbose mode: {'ON' if verbose else 'OFF'}")
            continue

        print()
        state = run_query(graph, question, verbose=verbose)
        print_result(state)
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agentic RAG Demo — 基于 LangGraph 的智能检索增强生成演示",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  python -m app.agent.demo.demo_runner "什么是RAG？"
  python -m app.agent.demo.demo_runner "向量数据库有哪些？" --verbose
  python -m app.agent.demo.demo_runner --interactive
        """,
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="要提问的问题（交互模式下可省略）",
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=10,
        help="检索返回的文档片段数（默认：10）",
    )
    parser.add_argument(
        "--rerank-top-n",
        type=int,
        default=5,
        help="重排序后保留的文档片段数（默认：5）",
    )
    parser.add_argument(
        "--max-rewrite-attempts",
        type=int,
        default=2,
        help="最大改写尝试次数（默认：2）",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=None,
        help="检索分数阈值（默认：使用配置文件中的值）",
    )
    parser.add_argument(
        "--no-rewrite",
        action="store_true",
        help="禁用查询改写",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细的中间状态流转",
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="进入交互式问答模式",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="以 JSON 格式输出最终状态",
    )

    args = parser.parse_args()

    # 参数校验
    if not args.interactive and not args.question:
        parser.error("请提供问题，或使用 --interactive 进入交互模式")

    # 构建图
    print("[INFO] Building Agentic RAG graph...")
    graph = build_graph(
        retrieval_k=args.retrieval_k,
        rerank_top_n=args.rerank_top_n,
        max_rewrite_attempts=args.max_rewrite_attempts,
        score_threshold=args.score_threshold,
        enable_rewrite=not args.no_rewrite,
        verbose=args.verbose,
    )
    print("[INFO] Graph built successfully.\n")

    # 执行
    if args.interactive:
        interactive_mode(graph, verbose=args.verbose)
    else:
        state = run_query(graph, args.question, verbose=args.verbose)
        if args.json:
            # JSON 输出（SearchResult 需要手动序列化）
            output = {
                "query": state.get("query", ""),
                "answer": state.get("answer", ""),
                "is_relevant": state.get("is_relevant"),
                "relevance_reason": state.get("relevance_reason", ""),
                "rewrite_attempts": state.get("rewrite_attempts", 0),
                "contexts": [
                    {
                        "text": c.text[:200],
                        "score": c.score,
                        "file_name": c.metadata.get("file_name", ""),
                        "chunk_index": c.metadata.get("chunk_index", ""),
                    }
                    for c in state.get("contexts", [])
                ],
                "messages": state.get("messages", []),
            }
            print(json.dumps(output, ensure_ascii=False, indent=2))
        else:
            print_result(state)


if __name__ == "__main__":
    main()
