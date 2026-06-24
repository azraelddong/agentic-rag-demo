import pytest
import httpx

from app.core.config import Settings, get_settings
from app.core.exceptions import ConfigurationError, LLMProviderError
from app.rag.reranker import (
    NoopReranker,
    SiliconFlowReranker,
    BgeReranker,
    Reranker,
)
from app.rag.vector_store import SearchResult
from app.api.dependencies import get_reranker


def _clear_settings_caches() -> None:
    """Clear lru_cache on both get_settings and get_reranker."""
    get_settings.cache_clear()
    get_reranker.cache_clear()


def test_settings_parses_rerank_integers_with_inline_comments(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_TOP_N", "5  # Rerank 后保留数量")
    monkeypatch.setenv("RERANK_RETRIEVAL_K", "20  # Rerank 前召回候选数")

    settings = Settings()

    assert settings.rerank_top_n == 5
    assert settings.rerank_retrieval_k == 20


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def make_result(
    text: str = "content",
    score: float = 0.85,
    *,
    file_name: str = "demo.md",
) -> SearchResult:
    return SearchResult(
        text=text,
        score=score,
        metadata={"file_name": file_name, "chunk_index": 0},
    )


def make_results(*texts: str) -> list[SearchResult]:
    return [make_result(text=t) for t in texts]


class FakeHttpxClient:
    """Records the last POST call and returns a canned response."""

    def __init__(self, response_json: dict, status_code: int = 200) -> None:
        self.response_json = response_json
        self.status_code = status_code
        self.last_url: str = ""
        self.last_json: dict | None = None
        self.last_headers: dict = {}

    def post(self, url: str, *, json: dict, headers: dict) -> "FakeResponse":
        self.last_url = url
        self.last_json = json
        self.last_headers = headers
        return FakeResponse(self.status_code, self.response_json)

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


class FakeResponse:
    def __init__(self, status_code: int, json_body: dict) -> None:
        self.status_code = status_code
        self._json = json_body

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                "error",
                request=httpx.Request("POST", ""),
                response=httpx.Response(self.status_code),
            )

    def json(self) -> dict:
        return self._json


class FailingHttpxClient:
    """Always raises a network error on POST."""

    def post(self, url: str, *, json: dict, headers: dict) -> None:
        raise httpx.ConnectError("connection refused")

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass


# ---------------------------------------------------------------------------
# NoopReranker
# ---------------------------------------------------------------------------

def test_noop_reranker_returns_results_unchanged() -> None:
    reranker = NoopReranker()
    results = make_results("a", "b", "c")

    output = reranker.rerank("query", results)
    assert output == results


def test_noop_reranker_ignores_top_n() -> None:
    reranker = NoopReranker()
    results = make_results("a", "b", "c")

    output = reranker.rerank("query", results, top_n=1)
    assert output == results
    assert len(output) == 3


# ---------------------------------------------------------------------------
# SiliconFlowReranker – request construction
# ---------------------------------------------------------------------------

def test_siliconflow_reranker_builds_correct_payload() -> None:
    fake = FakeHttpxClient({"results": []})
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    reranker._client = fake
    results = make_results("doc a", "doc b")

    reranker.rerank("test query", results, top_n=5)

    assert fake.last_url == "https://api.siliconflow.cn/v1/rerank"
    assert fake.last_json == {
        "model": "BAAI/bge-reranker-v2-m3",
        "query": "test query",
        "documents": ["doc a", "doc b"],
        "top_n": 5,
    }
    assert fake.last_headers["Authorization"] == "Bearer sk-test"
    assert fake.last_headers["Content-Type"] == "application/json"


def test_siliconflow_reranker_strips_trailing_slash_from_base_url() -> None:
    fake = FakeHttpxClient({"results": []})
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1/",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    reranker._client = fake

    reranker.rerank("q", make_results("x"), top_n=1)
    assert fake.last_url == "https://api.siliconflow.cn/v1/rerank"


# ---------------------------------------------------------------------------
# SiliconFlowReranker – response handling
# ---------------------------------------------------------------------------

def test_siliconflow_reranker_reorders_by_relevance_score() -> None:
    fake = FakeHttpxClient(
        {
            "results": [
                {"index": 2, "relevance_score": 0.95},
                {"index": 0, "relevance_score": 0.72},
                {"index": 1, "relevance_score": 0.31},
            ]
        }
    )
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    reranker._client = fake
    results = make_results("zero", "one", "two")

    output = reranker.rerank("q", results)

    assert len(output) == 3
    assert output[0].text == "two"  # index 2 → highest score
    assert output[0].score == 0.95
    assert output[1].text == "zero"  # index 0
    assert output[1].score == 0.72
    assert output[2].text == "one"  # index 1
    assert output[2].score == 0.31


def test_siliconflow_reranker_applies_top_n_trimming() -> None:
    fake = FakeHttpxClient(
        {
            "results": [
                {"index": 2, "relevance_score": 0.9},
                {"index": 0, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.7},
            ]
        }
    )
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    reranker._client = fake

    output = reranker.rerank("q", make_results("a", "b", "c"), top_n=2)
    assert len(output) == 2
    assert output[0].text == "c"  # index 2
    assert output[1].text == "a"  # index 0


def test_siliconflow_reranker_handles_empty_results() -> None:
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    # Should not make any HTTP call
    output = reranker.rerank("q", [])
    assert output == []


def test_siliconflow_reranker_falls_back_when_api_returns_no_items() -> None:
    fake = FakeHttpxClient({"results": []})
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    reranker._client = fake
    results = make_results("a", "b")

    output = reranker.rerank("q", results, top_n=1)
    # Falls back to original order, trimmed by top_n
    assert len(output) == 1
    assert output[0].text == "a"


def test_siliconflow_reranker_ignores_out_of_range_indices() -> None:
    fake = FakeHttpxClient(
        {
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 99, "relevance_score": 0.8},  # out of range
                {"index": 1, "relevance_score": 0.7},
            ]
        }
    )
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    reranker._client = fake
    results = make_results("a", "b")

    output = reranker.rerank("q", results)
    assert len(output) == 2
    assert output[0].text == "a"
    assert output[1].text == "b"


# ---------------------------------------------------------------------------
# SiliconFlowReranker – error handling
# ---------------------------------------------------------------------------

def test_siliconflow_reranker_raises_on_missing_api_key() -> None:
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    with pytest.raises(ConfigurationError, match="Rerank API Key"):
        reranker.rerank("q", make_results("a"))


def test_siliconflow_reranker_raises_on_placeholder_api_key() -> None:
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="your_api_key_here",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    with pytest.raises(ConfigurationError, match="Rerank API Key"):
        reranker.rerank("q", make_results("a"))


def test_siliconflow_reranker_raises_on_http_error() -> None:
    fake = FakeHttpxClient({"error": "unauthorized"}, status_code=401)
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    reranker._client = fake

    with pytest.raises(LLMProviderError, match="Rerank API"):
        reranker.rerank("q", make_results("a"))


def test_siliconflow_reranker_raises_on_network_error() -> None:
    reranker = SiliconFlowReranker(
        base_url="https://api.siliconflow.cn/v1",
        api_key="sk-test",
        model_name="BAAI/bge-reranker-v2-m3",
    )
    reranker._client = FailingHttpxClient()

    with pytest.raises(LLMProviderError, match="Rerank API"):
        reranker.rerank("q", make_results("a"))


# ---------------------------------------------------------------------------
# BgeReranker placeholder
# ---------------------------------------------------------------------------

def test_bge_reranker_raises_not_implemented() -> None:
    reranker = BgeReranker()
    with pytest.raises(NotImplementedError, match="bge-reranker"):
        reranker.rerank("q", make_results("a"))


# ---------------------------------------------------------------------------
# get_reranker factory function
# ---------------------------------------------------------------------------

def test_get_reranker_returns_noop_by_default(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_PROVIDER", "none")
    _clear_settings_caches()
    assert isinstance(get_reranker(), NoopReranker)


def test_get_reranker_returns_siliconflow(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_PROVIDER", "siliconflow")
    monkeypatch.setenv("LLM_API_KEY", "sk-fake")
    _clear_settings_caches()
    assert isinstance(get_reranker(), SiliconFlowReranker)


def test_get_reranker_returns_bge(monkeypatch) -> None:
    monkeypatch.setenv("RERANK_PROVIDER", "bge")
    _clear_settings_caches()
    assert isinstance(get_reranker(), BgeReranker)


# ---------------------------------------------------------------------------
# Config – effective_rerank_* properties
# ---------------------------------------------------------------------------

def test_effective_rerank_base_url_defaults_by_provider() -> None:
    settings = Settings(rerank_provider="siliconflow")
    assert settings.effective_rerank_base_url == "https://api.siliconflow.cn/v1"

    settings = Settings(rerank_provider="jina")
    assert settings.effective_rerank_base_url == "https://api.jina.ai/v1"


def test_effective_rerank_base_url_uses_explicit_value() -> None:
    settings = Settings(
        rerank_provider="siliconflow",
        rerank_base_url="https://custom.rerank.com/v1",
    )
    assert settings.effective_rerank_base_url == "https://custom.rerank.com/v1"


def test_effective_rerank_api_key_falls_back_to_llm_key() -> None:
    settings = Settings(
        llm_api_key="sk-llm",
        rerank_api_key=None,
    )
    assert settings.effective_rerank_api_key == "sk-llm"


def test_effective_rerank_api_key_prefers_own_key() -> None:
    settings = Settings(
        llm_api_key="sk-llm",
        rerank_api_key="sk-rerank",
    )
    assert settings.effective_rerank_api_key == "sk-rerank"


def test_rerank_enabled() -> None:
    assert Settings(rerank_provider="none").rerank_enabled is False
    assert Settings(rerank_provider="siliconflow").rerank_enabled is True
    assert Settings(rerank_provider="jina").rerank_enabled is True
    assert Settings(rerank_provider="bge").rerank_enabled is True
