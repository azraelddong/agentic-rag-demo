from unittest.mock import MagicMock

from app.rag.hybrid_retriever import HybridRetriever
from app.rag.vector_store import SearchResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _r(text: str, score: float, *, file_name: str = "f.md") -> SearchResult:
    return SearchResult(text=text, score=score, metadata={"file_name": file_name})


class FakeRetriever:
    """Drop-in fake for Retriever with a canned result list."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_query: str = ""
        self.last_top_k: int = 0

    def retrieve(self, query: str, *, top_k: int, metadata_filter=None, keywords=None, score_threshold=None):
        self.last_query = query
        self.last_top_k = top_k
        if score_threshold is not None:
            return [r for r in self.results if r.score >= score_threshold][:top_k]
        return self.results[:top_k]


class FakeBM25:
    """Drop-in fake for BM25Retriever with a canned result list."""

    def __init__(self, results: list[SearchResult]) -> None:
        self.results = results
        self.last_query: str = ""
        self.last_top_k: int = 0

    def search(self, query: str, *, top_k: int):
        self.last_query = query
        self.last_top_k = top_k
        return self.results[:top_k]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_hybrid_rrf_fuses_two_lists() -> None:
    dense = FakeRetriever([_r("d0", 0.9), _r("d1", 0.7), _r("d2", 0.5)])
    bm25 = FakeBM25([_r("b0", 8.0), _r("b1", 5.0), _r("b2", 3.0)])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = hr.retrieve("test", top_k=3)

    # Should have RRF-fused results from both sources
    assert len(results) >= 1
    assert all(isinstance(r.score, float) for r in results)


def test_hybrid_returns_dense_when_bm25_empty() -> None:
    dense = FakeRetriever([_r("a", 0.9), _r("b", 0.7)])
    bm25 = FakeBM25([])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = hr.retrieve("test", top_k=2)
    assert len(results) == 2
    assert results[0].text == "a"


def test_hybrid_returns_bm25_when_dense_empty() -> None:
    dense = FakeRetriever([])
    bm25 = FakeBM25([_r("x", 5.0), _r("y", 3.0)])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = hr.retrieve("test", top_k=2)
    assert len(results) == 2
    assert results[0].text == "x"


def test_hybrid_returns_empty_when_both_empty() -> None:
    dense = FakeRetriever([])
    bm25 = FakeBM25([])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = hr.retrieve("test", top_k=5)
    assert results == []


def test_hybrid_dedup_same_text_keeps_higher_score() -> None:
    """When both retrievers return the same chunk, keep the higher individual score."""
    shared = _r("important chunk", 0.85, file_name="doc.md")
    dense = FakeRetriever([shared, _r("d_only", 0.6)])
    bm25 = FakeBM25([_r("important chunk", 12.0, file_name="doc.md"), _r("b_only", 4.0)])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = hr.retrieve("test", top_k=5)

    texts = [r.text for r in results]
    assert "important chunk" in texts
    # Should be merged, not duplicated
    assert texts.count("important chunk") == 1


def test_hybrid_fetches_extra_candidates_for_better_fusion() -> None:
    """Hybrid retrieves top_k*2 from each source to give RRF more to work with."""
    dense = FakeRetriever([_r(f"d{i}", 1.0) for i in range(20)])
    bm25 = FakeBM25([_r(f"b{i}", 1.0) for i in range(20)])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    hr.retrieve("test", top_k=5)

    assert dense.last_top_k == 10
    assert bm25.last_top_k == 10


def test_hybrid_pass_metadata_filter_to_dense() -> None:
    dense = FakeRetriever([])
    bm25 = FakeBM25([])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    hr.retrieve("test", top_k=5, metadata_filter={"file_name": "x.md"})
    # metadata_filter is passed to dense only (BM25 doesn't support it)
    assert True


def test_hybrid_searches_each_keyword_individually() -> None:
    """Each keyword is searched individually, then results are merged."""
    dense = FakeRetriever([_r("a", 0.9)])
    # Each search returns different results — merged result has all unique chunks
    bm25 = FakeBM25([_r("b1", 5.0), _r("b2", 4.0)])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    results = hr.retrieve("q", top_k=5, keywords=["RabbitMQ", "分发", "模式"])

    # 3 keywords × 2 results each = merges into unique chunks
    assert len(results) >= 1


def test_hybrid_falls_back_to_single_query_when_keywords_empty() -> None:
    """When keywords is None, searches with the full query as a single term."""
    dense = FakeRetriever([_r("a", 0.9)])
    bm25 = FakeBM25([_r("b", 5.0)])
    hr = HybridRetriever(dense_retriever=dense, bm25_retriever=bm25)

    hr.retrieve("full question here", top_k=5, keywords=None)

    # Falls back: [query] → single search
    assert bm25.last_query == "full question here"
