# LangGraph Agent V2 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the first-version deterministic Agent into a LangGraph-backed Agent with one controlled retry using expanded `top_k`, forced query rewrite, and forced hybrid retrieval.

**Architecture:** Keep `/api/agent/ask` as the public interface and keep `/api/chat/ask` unchanged. Move Agent orchestration into `app/agent/graph.py` with a `StateGraph` containing `plan -> execute_rag -> reflect -> prepare_retry -> final`, while `AgentService` becomes a thin graph runner. Add small, backward-compatible `RAGChain.ask()` override parameters so the retry pass can reuse the existing RAG internals without copying retrieval logic into the Agent layer.

**Tech Stack:** Python 3.11, FastAPI, Pydantic, LangGraph `StateGraph`, existing RAGChain, existing HybridRetriever, existing QueryRewriter, pytest.

---

## File Map

- Modify `app/schemas/agent_schema.py`
  - Add `AgentTraceStep`.
  - Extend `AgentTrace` with `steps`.
  - Preserve existing `plan`, `reflection`, and `iterations` fields.

- Modify `app/rag/rag_chain.py`
  - Add optional per-call overrides for retriever and query rewriter.
  - Keep existing callers fully compatible.
  - Route retrieval through the selected retriever for each call.

- Modify `app/api/dependencies.py`
  - Split retriever factories into explicit dense and hybrid retriever providers.
  - Add a multi-query rewriter provider for Agent retry.
  - Inject retry dependencies into `AgentService`.

- Replace `app/agent/graph.py`
  - Implement `AgentState`.
  - Implement LangGraph node functions.
  - Build and compile the graph with conditional retry routing.

- Add `app/agent/constants.py`
  - Hold shared Agent plan and reflection constants.
  - Avoid circular imports between `agent_service.py` and `graph.py`.

- Modify `app/agent/agent_service.py`
  - Use the compiled graph instead of hand-written plan/execute/reflect/final flow.
  - Keep public `ask()` signature unchanged.
  - Import shared plan and reflection constants from `app/agent/constants.py`.

- Modify `tests/test_agent_service.py`
  - Update service tests for LangGraph execution, retry, and trace steps.

- Add `tests/test_agent_graph.py`
  - Unit-test graph routing and retry behavior without FastAPI.

- Modify `tests/test_agent_api.py`
  - Verify the response still contains existing trace fields and now includes `steps`.
  - Verify `/api/chat/ask` remains unaffected.

- Modify `README.md`
  - Document Agent V2 graph, retry policy, and trace shape.

---

## Design Decisions

- The retry policy is fixed in backend code for now.
- Max iterations is `2`.
- Retry is attempted only when `reflection == "insufficient_context"` after the first execution.
- Retry uses:
  - `top_k = min((requested_top_k or settings.rag_top_k) * 2, 20)`
  - forced multi-query rewrite
  - forced hybrid retriever
- No open-ended tools.
- No LLM judge in this version.
- No public request parameter for retry strategy.

---

### Task 1: Extend Agent Trace Schema

**Files:**
- Modify: `app/schemas/agent_schema.py`
- Modify: `tests/test_agent_service.py`

- [ ] **Step 1: Add schema expectations in service tests**

Add this test near the existing final-response tests in `tests/test_agent_service.py`:

```python
def test_trace_can_include_steps(self) -> None:
    from app.schemas.agent_schema import AgentTrace, AgentTraceStep

    trace = AgentTrace(
        plan="rag_search",
        reflection="supported",
        iterations=1,
        steps=[
            AgentTraceStep(
                node="plan",
                decision="rag_search",
            ),
            AgentTraceStep(
                node="execute_rag",
                source_count=1,
                top_k=5,
            ),
        ],
    )

    assert trace.steps[0].node == "plan"
    assert trace.steps[0].decision == "rag_search"
    assert trace.steps[1].source_count == 1
    assert trace.steps[1].top_k == 5
```

- [ ] **Step 2: Run the focused test and verify it fails**

Run:

```bash
uv run pytest tests/test_agent_service.py::TestAgentServiceFinal::test_trace_can_include_steps -q
```

Expected: fail with an import or validation error because `AgentTraceStep` does not exist yet.

- [ ] **Step 3: Implement trace step schema**

Update `app/schemas/agent_schema.py` to include this model and field:

```python
class AgentTraceStep(BaseModel):
    """One visible step in the Agent graph execution trace."""

    node: str = Field(..., description="执行节点名称")
    decision: str | None = Field(default=None, description="节点决策")
    reflection: str | None = Field(default=None, description="反思结果")
    source_count: int | None = Field(default=None, description="来源数量")
    top_k: int | None = Field(default=None, description="本轮检索 top_k")
    retry_reason: str | None = Field(default=None, description="重试原因")
```

Then change `AgentTrace` to:

```python
class AgentTrace(BaseModel):
    """Lightweight trace for Agent graph execution."""

    plan: str = Field(default="rag_search", description="规划动作")
    reflection: str = Field(
        default="supported",
        description="反思结果: supported 或 insufficient_context",
    )
    iterations: int = Field(default=1, description="执行轮数")
    steps: list[AgentTraceStep] = Field(
        default_factory=list,
        description="Agent graph execution steps",
    )
```

- [ ] **Step 4: Run the focused test and verify it passes**

Run:

```bash
uv run pytest tests/test_agent_service.py::TestAgentServiceFinal::test_trace_can_include_steps -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add app/schemas/agent_schema.py tests/test_agent_service.py
git commit -m "test: extend agent trace schema"
```

---

### Task 2: Add RAGChain Per-Call Overrides

**Files:**
- Modify: `app/rag/rag_chain.py`
- Add tests in: `tests/test_agent_graph.py`

- [ ] **Step 1: Add tests for retriever and rewriter override behavior**

Create `tests/test_agent_graph.py` with these initial tests:

```python
from unittest.mock import MagicMock

from app.core.config import Settings
from app.rag.rag_chain import RAGChain
from app.rag.reranker import NoopReranker
from app.rag.vector_store import SearchResult


class DummyPromptBuilder:
    def build_messages(self, question, results):
        return [{"role": "user", "content": question}]


class DummyChatModel:
    def generate(self, messages):
        return "generated answer"


def _result(score: float = 0.9) -> SearchResult:
    return SearchResult(
        text="chunk text",
        score=score,
        metadata={
            "file_name": "doc.md",
            "file_path": "/docs/doc.md",
            "chunk_index": 0,
            "source_type": "md",
        },
    )


def test_rag_chain_uses_retriever_override_when_provided() -> None:
    default_retriever = MagicMock()
    default_retriever.retrieve.return_value = []

    override_retriever = MagicMock()
    override_retriever.retrieve.return_value = [_result()]

    chain = RAGChain(
        settings=Settings(query_rewrite_method="none"),
        retriever=default_retriever,
        prompt_builder=DummyPromptBuilder(),
        chat_model=DummyChatModel(),
        reranker=NoopReranker(),
        query_rewriter=None,
    )

    answer, results = chain.ask(
        "question",
        top_k=3,
        retriever_override=override_retriever,
    )

    assert answer == "generated answer"
    assert len(results) == 1
    default_retriever.retrieve.assert_not_called()
    override_retriever.retrieve.assert_called_once()


def test_rag_chain_can_force_query_rewriter_override() -> None:
    retriever = MagicMock()
    retriever.retrieve.return_value = [_result()]

    rewriter = MagicMock()
    rewriter.rewrite.return_value.queries = ["rewritten question"]
    rewriter.rewrite.return_value.keywords = ["keyword"]

    chain = RAGChain(
        settings=Settings(query_rewrite_method="none"),
        retriever=retriever,
        prompt_builder=DummyPromptBuilder(),
        chat_model=DummyChatModel(),
        reranker=NoopReranker(),
        query_rewriter=None,
    )

    chain.ask(
        "original question",
        top_k=3,
        query_rewriter_override=rewriter,
        force_query_rewrite=True,
    )

    rewriter.rewrite.assert_called_once_with("original question")
    retriever.retrieve.assert_called_once()
    assert retriever.retrieve.call_args.args[0] == "rewritten question"
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
uv run pytest tests/test_agent_graph.py -q
```

Expected: fail because `RAGChain.ask()` does not accept `retriever_override`, `query_rewriter_override`, or `force_query_rewrite`.

- [ ] **Step 3: Update `RAGChain.ask()` signature**

Modify `app/rag/rag_chain.py`:

```python
def ask(
    self,
    question: str,
    *,
    top_k: int | None = None,
    metadata_filter: dict[str, Any] | None = None,
    retriever_override: Retriever | HybridRetriever | None = None,
    query_rewriter_override: QueryRewriter | None = None,
    force_query_rewrite: bool | None = None,
) -> tuple[str, list[SearchResult]]:
```

- [ ] **Step 4: Select active retriever and rewriter inside `ask()`**

Inside `ask()`, after `normalized_question = question.strip()`, add:

```python
active_retriever = retriever_override or self.retriever
active_query_rewriter = query_rewriter_override or self.query_rewriter
rewrite_enabled = (
    force_query_rewrite
    if force_query_rewrite is not None
    else self.settings.query_rewrite_enabled
)
```

Then change the rewrite guard to:

```python
if rewrite_enabled and active_query_rewriter is not None:
    logger.info("Query rewrite 前: \"%s\"", normalized_question)
    rewrite_result = active_query_rewriter.rewrite(normalized_question)
    logger.info(
        "Query rewrite 后: queries=%s  keywords=%s",
        rewrite_result.queries, rewrite_result.keywords,
    )
```

- [ ] **Step 5: Pass active retriever into `_multi_retrieve()`**

Change the call:

```python
results = self._multi_retrieve(
    active_retriever,
    rewrite_result.queries,
    retrieval_k,
    metadata_filter,
    keywords=rewrite_result.keywords,
    score_threshold=self.settings.rag_score_threshold,
)
```

Change `_multi_retrieve()` signature:

```python
def _multi_retrieve(
    self,
    retriever: Retriever | HybridRetriever,
    queries: list[str],
    retrieval_k: int,
    metadata_filter: dict[str, Any] | None,
    *,
    keywords: list[str] | None = None,
    score_threshold: float | None = None,
) -> list[SearchResult]:
```

Replace each `self.retriever.retrieve(...)` inside `_multi_retrieve()` with `retriever.retrieve(...)`.

- [ ] **Step 6: Run tests and verify they pass**

Run:

```bash
uv run pytest tests/test_agent_graph.py tests/test_query_rewriter.py -q
```

Expected: pass.

- [ ] **Step 7: Commit**

```bash
git add app/rag/rag_chain.py tests/test_agent_graph.py
git commit -m "feat: add rag chain call overrides"
```

---

### Task 3: Add Explicit Retry Dependencies

**Files:**
- Modify: `app/api/dependencies.py`
- Modify: `tests/test_agent_service.py`

- [ ] **Step 1: Add dependency test for AgentService construction**

Add this method to the existing `TestAgentServiceAsk` class in `tests/test_agent_service.py`:

```python
def test_agent_service_accepts_retry_dependencies(self) -> None:
    rag_chain = MagicMock()
    retry_retriever = MagicMock()
    retry_query_rewriter = MagicMock()

    svc = AgentService(
        rag_chain=rag_chain,
        retry_retriever=retry_retriever,
        retry_query_rewriter=retry_query_rewriter,
    )

    assert svc.rag_chain is rag_chain
    assert svc.retry_retriever is retry_retriever
    assert svc.retry_query_rewriter is retry_query_rewriter
```

- [ ] **Step 2: Run focused test and verify it fails**

Run:

```bash
uv run pytest tests/test_agent_service.py::TestAgentServiceAsk::test_agent_service_accepts_retry_dependencies -q
```

Expected: fail because `AgentService.__init__()` does not accept retry dependencies.

- [ ] **Step 3: Split retriever providers**

Modify `app/api/dependencies.py` around retriever creation:

```python
@lru_cache
def get_dense_retriever() -> Retriever:
    return Retriever(get_embedding_model(), get_vector_store())


@lru_cache
def get_hybrid_retriever() -> HybridRetriever:
    return HybridRetriever(
        dense_retriever=get_dense_retriever(),
        bm25_retriever=get_bm25_retriever(),
    )


@lru_cache
def get_retriever() -> Retriever | HybridRetriever:
    settings = get_settings()
    if settings.retrieval_method == "hybrid":
        return get_hybrid_retriever()
    return get_dense_retriever()
```

- [ ] **Step 4: Add retry query rewriter provider**

Add this function to `app/api/dependencies.py`:

```python
@lru_cache
def get_agent_retry_query_rewriter() -> QueryRewriter:
    return MultiQueryRewriter(
        get_chat_model().generate,
        num_queries=get_settings().query_rewrite_multi_count,
    )
```

- [ ] **Step 5: Update `get_agent_service()`**

Change `get_agent_service()`:

```python
@lru_cache
def get_agent_service() -> AgentService:
    return AgentService(
        rag_chain=get_rag_chain(),
        retry_retriever=get_hybrid_retriever(),
        retry_query_rewriter=get_agent_retry_query_rewriter(),
    )
```

- [ ] **Step 6: Move shared constants to a dedicated module**

Create `app/agent/constants.py`:

```python
"""Shared constants for Agent planning and reflection."""

PLAN_RAG_SEARCH = "rag_search"

REFLECTION_SUPPORTED = "supported"
REFLECTION_INSUFFICIENT_CONTEXT = "insufficient_context"
```

Update `app/agent/agent_service.py` imports:

```python
from app.agent.constants import (
    PLAN_RAG_SEARCH,
    REFLECTION_INSUFFICIENT_CONTEXT,
    REFLECTION_SUPPORTED,
)
```

Remove the local definitions of those three constants from `agent_service.py`.

- [ ] **Step 7: Update `AgentService.__init__()` signature**

Modify `app/agent/agent_service.py`:

```python
def __init__(
    self,
    rag_chain: RAGChain,
    *,
    retry_retriever: Retriever | HybridRetriever | None = None,
    retry_query_rewriter: QueryRewriter | None = None,
) -> None:
    self.rag_chain = rag_chain
    self.retry_retriever = retry_retriever
    self.retry_query_rewriter = retry_query_rewriter
```

Add imports:

```python
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import QueryRewriter
from app.rag.retriever import Retriever
```

- [ ] **Step 8: Run dependency-related tests**

Run:

```bash
uv run pytest tests/test_agent_service.py tests/test_agent_api.py -q
```

Expected: pass.

- [ ] **Step 9: Commit**

```bash
git add app/api/dependencies.py app/agent/constants.py app/agent/agent_service.py tests/test_agent_service.py
git commit -m "feat: inject agent retry dependencies"
```

---

### Task 4: Implement LangGraph Agent Graph

**Files:**
- Replace: `app/agent/graph.py`
- Modify: `tests/test_agent_graph.py`

- [ ] **Step 1: Add graph behavior tests**

Append these tests to `tests/test_agent_graph.py`:

```python
from app.agent.constants import (
    PLAN_RAG_SEARCH,
    REFLECTION_INSUFFICIENT_CONTEXT,
    REFLECTION_SUPPORTED,
)
from app.agent.graph import build_agentic_rag_graph
from app.rag.rag_chain import NOT_FOUND_ANSWER


def test_graph_finishes_after_supported_first_pass() -> None:
    rag_chain = MagicMock()
    rag_chain.ask.return_value = ("answer", [_result()])

    graph = build_agentic_rag_graph(
        rag_chain=rag_chain,
        retry_retriever=MagicMock(),
        retry_query_rewriter=MagicMock(),
    )

    state = graph.invoke(
        {
            "question": "q",
            "top_k": 5,
            "metadata_filter": None,
            "max_iterations": 2,
            "max_top_k": 20,
            "trace_steps": [],
        }
    )

    assert state["answer"] == "answer"
    assert state["reflection"] == REFLECTION_SUPPORTED
    assert state["iterations"] == 1
    assert rag_chain.ask.call_count == 1
    assert [step["node"] for step in state["trace_steps"]] == [
        "plan",
        "execute_rag",
        "reflect",
        "final",
    ]


def test_graph_retries_once_with_enhanced_strategy() -> None:
    rag_chain = MagicMock()
    rag_chain.ask.side_effect = [
        (NOT_FOUND_ANSWER, []),
        ("retry answer", [_result()]),
    ]

    retry_retriever = MagicMock()
    retry_query_rewriter = MagicMock()

    graph = build_agentic_rag_graph(
        rag_chain=rag_chain,
        retry_retriever=retry_retriever,
        retry_query_rewriter=retry_query_rewriter,
    )

    state = graph.invoke(
        {
            "question": "q",
            "top_k": 5,
            "metadata_filter": None,
            "max_iterations": 2,
            "max_top_k": 20,
            "trace_steps": [],
        }
    )

    assert state["answer"] == "retry answer"
    assert state["reflection"] == REFLECTION_SUPPORTED
    assert state["iterations"] == 2
    assert rag_chain.ask.call_count == 2

    retry_call = rag_chain.ask.call_args_list[1]
    assert retry_call.kwargs["top_k"] == 10
    assert retry_call.kwargs["retriever_override"] is retry_retriever
    assert retry_call.kwargs["query_rewriter_override"] is retry_query_rewriter
    assert retry_call.kwargs["force_query_rewrite"] is True
    assert "prepare_retry" in [step["node"] for step in state["trace_steps"]]


def test_graph_stops_after_retry_failure() -> None:
    rag_chain = MagicMock()
    rag_chain.ask.return_value = (NOT_FOUND_ANSWER, [])

    graph = build_agentic_rag_graph(
        rag_chain=rag_chain,
        retry_retriever=MagicMock(),
        retry_query_rewriter=MagicMock(),
    )

    state = graph.invoke(
        {
            "question": "q",
            "top_k": 5,
            "metadata_filter": None,
            "max_iterations": 2,
            "max_top_k": 20,
            "trace_steps": [],
        }
    )

    assert state["reflection"] == REFLECTION_INSUFFICIENT_CONTEXT
    assert state["iterations"] == 2
    assert rag_chain.ask.call_count == 2
```

- [ ] **Step 2: Run graph tests and verify they fail**

Run:

```bash
uv run pytest tests/test_agent_graph.py -q
```

Expected: fail because `build_agentic_rag_graph()` is still a placeholder.

- [ ] **Step 3: Replace `app/agent/graph.py`**

Use this implementation structure:

```python
from __future__ import annotations

from typing import Any, Literal, TypedDict

from langgraph.graph import END, START, StateGraph

from app.agent.constants import (
    PLAN_RAG_SEARCH,
    REFLECTION_INSUFFICIENT_CONTEXT,
    REFLECTION_SUPPORTED,
)
from app.rag.hybrid_retriever import HybridRetriever
from app.rag.query_rewriter import QueryRewriter
from app.rag.rag_chain import NOT_FOUND_ANSWER, RAGChain
from app.rag.retriever import Retriever
from app.rag.vector_store import SearchResult


class AgentState(TypedDict, total=False):
    question: str
    top_k: int | None
    metadata_filter: dict[str, Any] | None
    max_iterations: int
    max_top_k: int
    plan: str
    answer: str
    results: list[SearchResult]
    reflection: str
    iterations: int
    current_top_k: int | None
    retry_reason: str | None
    use_retry_strategy: bool
    trace_steps: list[dict[str, Any]]


def _append_step(state: AgentState, step: dict[str, Any]) -> list[dict[str, Any]]:
    return [*state.get("trace_steps", []), step]


def build_agentic_rag_graph(
    *,
    rag_chain: RAGChain,
    retry_retriever: Retriever | HybridRetriever | None = None,
    retry_query_rewriter: QueryRewriter | None = None,
):
    def plan_node(state: AgentState) -> AgentState:
        return {
            "plan": PLAN_RAG_SEARCH,
            "iterations": state.get("iterations", 0),
            "current_top_k": state.get("top_k"),
            "use_retry_strategy": False,
            "trace_steps": _append_step(
                state,
                {"node": "plan", "decision": PLAN_RAG_SEARCH},
            ),
        }

    def execute_rag_node(state: AgentState) -> AgentState:
        use_retry = state.get("use_retry_strategy", False)
        ask_kwargs: dict[str, Any] = {
            "top_k": state.get("current_top_k"),
            "metadata_filter": state.get("metadata_filter"),
        }
        if use_retry:
            ask_kwargs.update(
                {
                    "retriever_override": retry_retriever,
                    "query_rewriter_override": retry_query_rewriter,
                    "force_query_rewrite": True,
                }
            )

        answer, results = rag_chain.ask(state["question"], **ask_kwargs)
        iterations = state.get("iterations", 0) + 1

        return {
            "answer": answer,
            "results": results,
            "iterations": iterations,
            "trace_steps": _append_step(
                state,
                {
                    "node": "execute_rag",
                    "source_count": len(results),
                    "top_k": state.get("current_top_k"),
                    "decision": "retry" if use_retry else "initial",
                },
            ),
        }

    def reflect_node(state: AgentState) -> AgentState:
        results = state.get("results", [])
        answer = state.get("answer", "")
        if results and answer != NOT_FOUND_ANSWER:
            reflection = REFLECTION_SUPPORTED
        else:
            reflection = REFLECTION_INSUFFICIENT_CONTEXT

        return {
            "reflection": reflection,
            "trace_steps": _append_step(
                state,
                {"node": "reflect", "reflection": reflection},
            ),
        }

    def route_after_reflect(state: AgentState) -> Literal["prepare_retry", "final"]:
        if (
            state.get("reflection") == REFLECTION_INSUFFICIENT_CONTEXT
            and state.get("iterations", 0) < state.get("max_iterations", 2)
        ):
            return "prepare_retry"
        return "final"

    def prepare_retry_node(state: AgentState) -> AgentState:
        base_top_k = state.get("top_k") or rag_chain.settings.rag_top_k
        max_top_k = state.get("max_top_k", 20)
        retry_top_k = min(base_top_k * 2, max_top_k)
        retry_reason = "insufficient_context"

        return {
            "current_top_k": retry_top_k,
            "retry_reason": retry_reason,
            "use_retry_strategy": True,
            "trace_steps": _append_step(
                state,
                {
                    "node": "prepare_retry",
                    "retry_reason": retry_reason,
                    "top_k": retry_top_k,
                    "decision": "expanded_rewrite_hybrid_retry",
                },
            ),
        }

    def final_node(state: AgentState) -> AgentState:
        return {
            "trace_steps": _append_step(
                state,
                {
                    "node": "final",
                    "reflection": state.get("reflection"),
                    "source_count": len(state.get("results", [])),
                },
            )
        }

    builder = StateGraph(AgentState)
    builder.add_node("plan", plan_node)
    builder.add_node("execute_rag", execute_rag_node)
    builder.add_node("reflect", reflect_node)
    builder.add_node("prepare_retry", prepare_retry_node)
    builder.add_node("final", final_node)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "execute_rag")
    builder.add_edge("execute_rag", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflect,
        {
            "prepare_retry": "prepare_retry",
            "final": "final",
        },
    )
    builder.add_edge("prepare_retry", "execute_rag")
    builder.add_edge("final", END)

    return builder.compile()
```

- [ ] **Step 4: Run graph tests**

Run:

```bash
uv run pytest tests/test_agent_graph.py -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add app/agent/graph.py tests/test_agent_graph.py
git commit -m "feat: build langgraph agent workflow"
```

---

### Task 5: Wire AgentService to LangGraph

**Files:**
- Modify: `app/agent/agent_service.py`
- Modify: `tests/test_agent_service.py`

- [ ] **Step 1: Update service tests for retry response**

Add this test to `tests/test_agent_service.py`:

```python
def test_ask_retries_once_when_context_is_insufficient(self) -> None:
    mock_chain = MagicMock()
    mock_chain.ask.side_effect = [
        (NOT_FOUND_ANSWER, []),
        ("第二轮回答", [_make_result()]),
    ]

    retry_retriever = MagicMock()
    retry_query_rewriter = MagicMock()

    svc = AgentService(
        rag_chain=mock_chain,
        retry_retriever=retry_retriever,
        retry_query_rewriter=retry_query_rewriter,
    )

    resp = svc.ask(question="测试问题", top_k=5)

    assert resp.answer == "第二轮回答"
    assert resp.trace.iterations == 2
    assert resp.trace.reflection == REFLECTION_SUPPORTED
    assert [step.node for step in resp.trace.steps] == [
        "plan",
        "execute_rag",
        "reflect",
        "prepare_retry",
        "execute_rag",
        "reflect",
        "final",
    ]

    retry_call = mock_chain.ask.call_args_list[1]
    assert retry_call.kwargs["top_k"] == 10
    assert retry_call.kwargs["retriever_override"] is retry_retriever
    assert retry_call.kwargs["query_rewriter_override"] is retry_query_rewriter
    assert retry_call.kwargs["force_query_rewrite"] is True
```

- [ ] **Step 2: Run focused service tests and verify failure**

Run:

```bash
uv run pytest tests/test_agent_service.py -q
```

Expected: fail because `AgentService.ask()` still uses hand-written single-pass flow.

- [ ] **Step 3: Update `AgentService.ask()` to invoke graph**

In `app/agent/agent_service.py`, import:

```python
from app.agent.graph import build_agentic_rag_graph
from app.agent.constants import (
    PLAN_RAG_SEARCH,
    REFLECTION_INSUFFICIENT_CONTEXT,
)
from app.schemas.agent_schema import AgentResponse, AgentTrace, AgentTraceStep
```

In `__init__()`, compile the graph:

```python
self.graph = build_agentic_rag_graph(
    rag_chain=rag_chain,
    retry_retriever=retry_retriever,
    retry_query_rewriter=retry_query_rewriter,
)
```

Replace `ask()` body with:

```python
state = self.graph.invoke(
    {
        "question": question,
        "top_k": top_k,
        "metadata_filter": metadata_filter,
        "max_iterations": 2,
        "max_top_k": 20,
        "trace_steps": [],
    }
)

return AgentResponse(
    answer=state.get("answer", NOT_FOUND_ANSWER),
    sources=self._build_sources(state.get("results", [])),
    trace=AgentTrace(
        plan=state.get("plan", PLAN_RAG_SEARCH),
        reflection=state.get("reflection", REFLECTION_INSUFFICIENT_CONTEXT),
        iterations=state.get("iterations", 0),
        steps=[AgentTraceStep(**step) for step in state.get("trace_steps", [])],
    ),
)
```

Keep `_build_sources()` unchanged.

- [ ] **Step 4: Remove private flow methods and replace private-method tests**

After `ask()` uses graph, remove these methods from `app/agent/agent_service.py`:

```python
_plan
_execute
_reflect
_final
```

Keep `_build_sources()` because `AgentService.ask()` still uses it to convert `SearchResult` objects into response `Source` objects.

In `tests/test_agent_service.py`, remove these private-method test classes:

```python
class TestAgentServicePlan
class TestAgentServiceReflect
class TestAgentServiceFinal
```

Move the trace schema test from Task 1 into a new class:

```python
class TestAgentTraceSchema:
    def test_trace_can_include_steps(self) -> None:
        from app.schemas.agent_schema import AgentTrace, AgentTraceStep

        trace = AgentTrace(
            plan="rag_search",
            reflection="supported",
            iterations=1,
            steps=[
                AgentTraceStep(node="plan", decision="rag_search"),
                AgentTraceStep(node="execute_rag", source_count=1, top_k=5),
            ],
        )

        assert trace.steps[0].node == "plan"
        assert trace.steps[0].decision == "rag_search"
        assert trace.steps[1].source_count == 1
        assert trace.steps[1].top_k == 5
```

Keep `TestAgentServiceAsk` and update it to assert public `ask()` behavior only.

- [ ] **Step 5: Run service tests**

Run:

```bash
uv run pytest tests/test_agent_service.py tests/test_agent_graph.py -q
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add app/agent/agent_service.py tests/test_agent_service.py
git commit -m "feat: run agent service through langgraph"
```

---

### Task 6: Update Agent API Tests

**Files:**
- Modify: `tests/test_agent_api.py`

- [ ] **Step 1: Update API response shape test**

Change `test_response_has_answer_sources_trace()` assertions:

```python
assert "answer" in body
assert "sources" in body
assert "trace" in body
assert body["trace"]["plan"] == "rag_search"
assert body["trace"]["iterations"] >= 1
assert "steps" in body["trace"]
assert body["trace"]["steps"][0]["node"] == "plan"
```

- [ ] **Step 2: Add API retry shape test**

Add:

```python
def test_response_trace_shows_retry_when_first_pass_has_no_sources(self) -> None:
    with patch(
        "app.rag.rag_chain.RAGChain.ask",
        side_effect=[
            ("知识库中未找到相关信息", []),
            ("第二轮回答", []),
        ],
    ):
        client = TestClient(app)
        resp = client.post("/api/agent/ask", json={"question": "测试", "top_k": 5})

    assert resp.status_code == 200
    body = resp.json()
    assert body["trace"]["iterations"] == 2
    assert "prepare_retry" in [step["node"] for step in body["trace"]["steps"]]
```

- [ ] **Step 3: Run API tests**

Run:

```bash
uv run pytest tests/test_agent_api.py -q
```

Expected: pass.

- [ ] **Step 4: Run chat unaffected test explicitly**

Run:

```bash
uv run pytest tests/test_agent_api.py::TestChatApiUnaffected::test_chat_ask_still_accepts_requests -q
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_agent_api.py
git commit -m "test: cover langgraph agent api trace"
```

---

### Task 7: Update README Documentation

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update Agent section**

In the Agent section, describe V2 as:

````markdown
第二版 Agent 使用 LangGraph 编排：

```text
START
  -> plan
  -> execute_rag
  -> reflect
  -> prepare_retry 或 final
  -> END
```

当第一轮 `reflection = insufficient_context` 时，Agent 最多重试一次：

- `top_k` 翻倍，最大不超过 20。
- 强制使用 multi-query rewrite。
- 强制使用 hybrid 检索。
- 不开放任意工具调用。
```
````

- [ ] **Step 2: Update response example**

Use this trace shape:

```json
{
  "answer": "...",
  "sources": [],
  "trace": {
    "plan": "rag_search",
    "reflection": "supported",
    "iterations": 2,
    "steps": [
      {
        "node": "plan",
        "decision": "rag_search"
      },
      {
        "node": "execute_rag",
        "decision": "initial",
        "source_count": 0,
        "top_k": 5
      },
      {
        "node": "reflect",
        "reflection": "insufficient_context"
      },
      {
        "node": "prepare_retry",
        "decision": "expanded_rewrite_hybrid_retry",
        "retry_reason": "insufficient_context",
        "top_k": 10
      }
    ]
  }
}
```

- [ ] **Step 3: Run README-related smoke tests**

Run:

```bash
uv run pytest tests/test_agent_api.py tests/test_agent_service.py tests/test_agent_graph.py -q
```

Expected: pass.

- [ ] **Step 4: Commit**

```bash
git add README.md
git commit -m "docs: describe langgraph agent retry"
```

---

### Task 8: Full Verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run full test suite**

Run:

```bash
uv run pytest -q
```

Expected: all tests pass.

- [ ] **Step 2: Run formatting check if project has one**

Inspect `pyproject.toml`. This project currently does not define ruff, black, mypy, or coverage commands. No formatting command is required unless such tooling is added before execution.

- [ ] **Step 3: Check git diff**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

- [ ] **Step 4: Inspect final changed files**

Run:

```bash
git status --short
```

Expected changed files:

```text
M README.md
M app/agent/agent_service.py
A app/agent/constants.py
M app/agent/graph.py
M app/api/dependencies.py
M app/rag/rag_chain.py
M app/schemas/agent_schema.py
M tests/test_agent_api.py
M tests/test_agent_service.py
A tests/test_agent_graph.py
```

- [ ] **Step 5: Final commit**

If the previous task commits were made, no extra commit is needed. If implementation was done without intermediate commits, run:

```bash
git add README.md app/agent/agent_service.py app/agent/constants.py app/agent/graph.py app/api/dependencies.py app/rag/rag_chain.py app/schemas/agent_schema.py tests/test_agent_api.py tests/test_agent_service.py tests/test_agent_graph.py
git commit -m "feat: add langgraph agent retry workflow"
```

---

## Self-Review Notes

- The plan covers the confirmed scope: LangGraph graph orchestration, fixed backend retry strategy, combined retry with expanded `top_k`, forced rewrite, forced hybrid retrieval, and a small compatible `RAGChain.ask()` change.
- The plan does not add public retry strategy parameters.
- The plan keeps `/api/chat/ask` unchanged.
- The plan keeps retry bounded to two iterations.
- The plan does not add open-ended tools, external web search, SQL tools, memory, or LLM judge.
