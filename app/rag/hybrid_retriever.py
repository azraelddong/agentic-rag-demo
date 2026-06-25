from __future__ import annotations

import logging
from typing import Any

from app.rag.bm25_retriever import BM25Retriever
from app.rag.retriever import Retriever
from app.rag.vector_store import SearchResult

logger = logging.getLogger(__name__)

# Reciprocal Rank Fusion constant (industry-standard default)
RRF_K = 60


class HybridRetriever:
    """Hybrid retriever: dense vector + BM25 keyword, fused via RRF.

    Has the same public interface as ``Retriever`` so it can be dropped into
    ``RAGChain`` without any changes to the chain itself.
    """

    def __init__(
        self,
        *,
        dense_retriever: Retriever,
        bm25_retriever: BM25Retriever,
    ) -> None:
        self._dense = dense_retriever
        self._bm25 = bm25_retriever

    def retrieve(
        self,
        query: str,
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
        keywords: list[str] | None = None,
        score_threshold: float | None = None,
    ) -> list[SearchResult]:
        """Dense + BM25, filtered individually before RRF fusion.

        Dense results are filtered by *COSINE* ≥ score_threshold.
        BM25 results are filtered by score > 0.
        Only then are the two lists fused via RRF — so the threshold
        compares against each retriever's native score scale.
        """
        fetch_k = max(top_k * 2, 10)

        # --- Dense (COSINE similarity) ---
        dense_raw = self._dense.retrieve(
            query, top_k=fetch_k, metadata_filter=metadata_filter,
        )
        dense_filtered = dense_raw
        if score_threshold is not None:
            dense_filtered = [r for r in dense_raw if r.score >= score_threshold]
            if len(dense_filtered) < len(dense_raw):
                logger.info(
                    "Dense 过滤: %d → %d (threshold=%.2f)",
                    len(dense_raw), len(dense_filtered), score_threshold,
                )

        # --- BM25 — search each keyword individually, then merge ---
        bm25_terms = keywords if keywords else [query]
        logger.info("BM25 关键词检索 (%d 个): %s", len(bm25_terms), bm25_terms)
        per_kw_k = max(3, fetch_k // len(bm25_terms))
        bm25_merged: dict[str, tuple[SearchResult, float]] = {}
        for term in bm25_terms:
            batch = self._bm25.search(term, top_k=per_kw_k)
            for r in batch:
                if r.score <= 0:
                    continue
                fp = self._fingerprint(r)
                if fp not in bm25_merged or r.score > bm25_merged[fp][1]:
                    bm25_merged[fp] = (r, r.score)
        bm25_filtered = [
            item[0] for item in
            sorted(bm25_merged.values(), key=lambda x: x[1], reverse=True)
        ][:fetch_k]

        # --- Log ---
        self._log_source_results("Dense 检索", dense_filtered)
        self._log_source_results("BM25 检索", bm25_filtered)

        if not dense_filtered and not bm25_filtered:
            return []

        if not dense_filtered:
            return bm25_filtered[:top_k]
        if not bm25_filtered:
            return dense_filtered[:top_k]

        return self._rrf_fuse(dense_filtered, bm25_filtered, top_k)

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_source_results(label: str, results: list[SearchResult]) -> None:
        if not results:
            logger.info("%s：0 条", label)
            return
        logger.info("%s（共 %d 条）：", label, len(results))
        for i, r in enumerate(results):
            meta = r.metadata
            preview = r.text[:100].replace("\n", " ")
            logger.info(
                "  [%2d] score=%.4f  file=%s  chunk=%s  text=\"%s...\"",
                i, r.score,
                meta.get("file_name", "?"),
                meta.get("chunk_index", "?"),
                preview,
            )

    # ------------------------------------------------------------------
    # RRF fusion
    # ------------------------------------------------------------------

    @staticmethod
    def _rrf_fuse(
        dense: list[SearchResult],
        bm25: list[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        """Reciprocal Rank Fusion over two ranked lists."""
        # Build dedup map (fingerprint → result + rrf_score)
        fused: dict[str, tuple[SearchResult, float]] = {}

        for rank, result in enumerate(dense):
            fp = HybridRetriever._fingerprint(result)
            rrf = 1.0 / (RRF_K + rank + 1)
            fused[fp] = (result, rrf)

        for rank, result in enumerate(bm25):
            fp = HybridRetriever._fingerprint(result)
            rrf = 1.0 / (RRF_K + rank + 1)
            if fp in fused:
                prev_result, prev_rrf = fused[fp]
                # Keep the result with the higher individual score
                best = prev_result if prev_result.score >= result.score else result
                fused[fp] = (best, prev_rrf + rrf)
            else:
                fused[fp] = (result, rrf)

        # Sort by RRF score descending, return top_k
        sorted_items = sorted(fused.values(), key=lambda x: x[1], reverse=True)
        results = []
        for result, rrf_score in sorted_items[:top_k]:
            result.score = rrf_score  # replace with fused score
            results.append(result)

        logger.info(
            "RRF fusion: dense=%d + bm25=%d → %d unique → top %d",
            len(dense), len(bm25), len(fused), len(results),
        )
        return results

    @staticmethod
    def _fingerprint(result: SearchResult) -> str:
        """Stable identity for dedup (first 200 chars)."""
        from hashlib import sha256
        return sha256(result.text[:200].encode()).hexdigest()
