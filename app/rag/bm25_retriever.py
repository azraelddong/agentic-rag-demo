from __future__ import annotations

import logging
import re
import jieba
from typing import Any

from rank_bm25 import BM25Okapi

from app.rag.vector_store import MilvusVectorStore, SearchResult

logger = logging.getLogger(__name__)


class BM25Retriever:
    """In-memory BM25 keyword retriever backed by chunks stored in Milvus."""

    def __init__(self, vector_store: MilvusVectorStore) -> None:
        self._vector_store = vector_store
        self._corpus: list[str] = []
        self._metadata: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = True

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def search(self, query: str, *, top_k: int) -> list[SearchResult]:
        """BM25 keyword search. Returns results with BM25 scores in [0, 1]."""
        if self._dirty:
            self._rebuild()
        if self._bm25 is None or not self._corpus:
            return []

        tokenized_query = self._tokenize(query)
        scores = self._bm25.get_scores(tokenized_query)
        if not scores.any():
            return []

        # Get top-k indices sorted by score descending
        indices = scores.argsort()[::-1][:top_k]
        return [
            SearchResult(
                text=self._corpus[i],
                score=float(scores[i]),
                metadata=self._metadata[i],
            )
            for i in indices
            if scores[i] > 0
        ]

    def rebuild(self) -> None:
        """Force rebuild the BM25 index from Milvus (call after document changes)."""
        self._dirty = True

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _rebuild(self) -> None:
        """Load all chunks from Milvus and build the BM25 index."""
        logger.info("Building BM25 index from Milvus chunks ...")
        try:
            raw = self._vector_store.fetch_chunks_batch()
        except Exception:
            logger.exception("Failed to load chunks for BM25, index will be empty")
            self._corpus = []
            self._metadata = []
            self._bm25 = None
            self._dirty = False
            return

        self._corpus = [item.get("text", "") for item in raw]
        self._metadata = [
            {
                "file_name": item.get("file_name"),
                "file_path": item.get("file_path"),
                "chunk_index": item.get("chunk_index"),
                "source_type": item.get("source_type"),
                "created_at": item.get("created_at"),
            }
            for item in raw
        ]
        tokenized = [self._tokenize(text) for text in self._corpus]
        self._bm25 = BM25Okapi(tokenized)
        self._dirty = False
        logger.info("BM25 index ready (%d documents)", len(self._corpus))

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Tokenize Chinese+English text using jieba for word segmentation.

        English words are space-separated natively; Chinese characters need
        jieba to split into meaningful words.  Without segmentation a phrase
        like "分发模型" stays as one token and cannot match documents where
        the same characters appear inside a larger unsegmented chunk.
        """
        # Split on whitespace / punctuation first to isolate CJK runs
        raw_tokens: list[str] = []
        for chunk in re.split(r"[\s，。！？、；：""''（）…+\\|/—-]+", text):
            chunk = chunk.strip()
            if not chunk:
                continue
            # If the chunk contains CJK characters, segment it with jieba
            if re.search(r"[一-鿿]", chunk):
                raw_tokens.extend(jieba.lcut(chunk))
            else:
                raw_tokens.append(chunk)

        return [t.lower() for t in raw_tokens if t]
