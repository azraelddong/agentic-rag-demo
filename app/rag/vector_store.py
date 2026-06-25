from dataclasses import dataclass
from hashlib import sha256
import logging
import os
from typing import Any

from pymilvus import DataType, MilvusClient

from app.core.config import Settings
from app.core.exceptions import VectorStoreError
from app.rag.text_splitter import DocumentChunk

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class SearchResult:
    """Vector search result normalized for the RAG pipeline."""

    text: str
    score: float
    metadata: dict

"""Milvus向量存储，支持集合自动创建和元数据过滤。"""
class MilvusVectorStore:
    """Milvus vector store with collection auto-creation and metadata filtering."""

    filterable_fields = {"file_name", "file_path", "source_type", "chunk_index"}

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.collection_name = settings.milvus_collection_name
        self.dimension = settings.embedding_dimension
        self.metric_type = settings.milvus_metric_type
        self.index_type = settings.milvus_index_type
        self._client: MilvusClient | None = None

    @property
    def client(self) -> MilvusClient:
        if self._client is None:
            try:
                self._configure_grpc_proxy_bypass()
                kwargs: dict[str, str] = {"uri": self.settings.milvus_uri}
                if self.settings.milvus_user:
                    kwargs["user"] = self.settings.milvus_user
                if self.settings.milvus_password:
                    kwargs["password"] = self.settings.milvus_password
                self._client = MilvusClient(**kwargs)
            except Exception as exc:
                logger.exception("Failed to connect Milvus")
                raise VectorStoreError(
                    "连接 Milvus 失败",
                    detail={"uri": self.settings.milvus_uri, "error": str(exc)},
                ) from exc
        return self._client

    def _configure_grpc_proxy_bypass(self) -> None:
        configured_hosts = [
            host.strip()
            for host in self.settings.milvus_no_grpc_proxy.split(",")
            if host.strip()
        ]
        if not configured_hosts:
            return

        existing_hosts = [
            host.strip()
            for host in os.environ.get("no_grpc_proxy", "").split(",")
            if host.strip()
        ]
        known_hosts = {host.lower() for host in existing_hosts}
        for host in configured_hosts:
            if host.lower() not in known_hosts:
                existing_hosts.append(host)
                known_hosts.add(host.lower())
        os.environ["no_grpc_proxy"] = ",".join(existing_hosts)

    def ensure_collection(self) -> None:
        try:
            if self.client.has_collection(self.collection_name):
                return

            schema = MilvusClient.create_schema(auto_id=False, enable_dynamic_field=True)
            schema.add_field(
                field_name="id",
                datatype=DataType.VARCHAR,
                is_primary=True,
                max_length=64,
            )
            schema.add_field(
                field_name="vector",
                datatype=DataType.FLOAT_VECTOR,
                dim=self.dimension,
            )
            schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=8192)
            schema.add_field(
                field_name="file_name",
                datatype=DataType.VARCHAR,
                max_length=512,
            )
            schema.add_field(
                field_name="file_path",
                datatype=DataType.VARCHAR,
                max_length=2048,
            )
            schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
            schema.add_field(
                field_name="source_type",
                datatype=DataType.VARCHAR,
                max_length=64,
            )
            schema.add_field(
                field_name="created_at",
                datatype=DataType.VARCHAR,
                max_length=64,
            )

            index_params = self.client.prepare_index_params()
            index_params.add_index(
                field_name="vector",
                metric_type=self.metric_type,
                index_type=self.index_type,
            )

            self.client.create_collection(
                collection_name=self.collection_name,
                schema=schema,
                index_params=index_params,
                consistency_level=self.settings.milvus_consistency_level,
            )
            logger.info("Created Milvus collection: %s", self.collection_name)
        except VectorStoreError:
            raise
        except Exception as exc:
            logger.exception("Failed to ensure Milvus collection")
            raise VectorStoreError(
                "创建或检查 Milvus Collection 失败",
                detail={"collection": self.collection_name, "error": str(exc)},
            ) from exc

    def fetch_chunks_batch(self, limit: int = 1000) -> list[dict[str, Any]]:
        """Return all chunk texts with metadata (for BM25 / keyword index building)."""
        self.ensure_collection()
        self.client.load_collection(self.collection_name)
        output_fields = [
            "id", "text", "file_name", "file_path",
            "chunk_index", "source_type", "created_at",
        ]
        chunks: list[dict[str, Any]] = []
        offset = 0
        try:
            while True:
                batch = self.client.query(
                    collection_name=self.collection_name,
                    filter="",
                    output_fields=output_fields,
                    limit=limit,
                    offset=offset,
                )
                if not batch:
                    break
                chunks.extend(batch)
                offset += limit
        except Exception as exc:
            logger.exception("Failed to fetch chunks from Milvus")
            raise VectorStoreError(
                "读取 Milvus 全量 chunks 失败",
                detail={"collection": self.collection_name, "error": str(exc)},
            ) from exc
        return chunks

    def drop_collection(self) -> None:
        try:
            if self.client.has_collection(self.collection_name):
                self.client.drop_collection(self.collection_name)
                logger.info("Dropped Milvus collection: %s", self.collection_name)
        except Exception as exc:
            logger.exception("Failed to drop Milvus collection")
            raise VectorStoreError(
                "删除 Milvus Collection 失败",
                detail={"collection": self.collection_name, "error": str(exc)},
            ) from exc

    def delete_chunks_for_files(self, file_paths: list[str]) -> None:
        unique_paths = sorted(set(file_paths))
        if not unique_paths:
            return

        self.ensure_collection()
        try:
            for file_path in unique_paths:
                escaped_path = self._escape_string(file_path)
                self.client.delete(
                    collection_name=self.collection_name,
                    filter=f'file_path == "{escaped_path}"',
                )
        except Exception as exc:
            logger.exception("Failed to delete existing document chunks")
            raise VectorStoreError(
                "删除文档旧向量失败",
                detail={"collection": self.collection_name, "error": str(exc)},
            ) from exc

    def add_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> int:
        if len(chunks) != len(vectors):
            raise VectorStoreError(
                "Chunk 数量与向量数量不一致",
                detail={"chunks": len(chunks), "vectors": len(vectors)},
            )
        if not chunks:
            return 0

        self.ensure_collection()
        records = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            records.append(
                {
                    "id": self._chunk_id(chunk),
                    "vector": vector,
                    "text": chunk.content,
                    "file_name": chunk.metadata["file_name"],
                    "file_path": chunk.metadata["file_path"],
                    "chunk_index": int(chunk.metadata["chunk_index"]),
                    "source_type": chunk.metadata["source_type"],
                    "created_at": chunk.metadata["created_at"],
                }
            )

        try:
            self.client.upsert(collection_name=self.collection_name, data=records)
            return len(records)
        except Exception as exc:
            logger.exception("Failed to upsert vectors into Milvus")
            raise VectorStoreError(
                "写入 Milvus 向量失败",
                detail={"collection": self.collection_name, "error": str(exc)},
            ) from exc

    """相似度搜索，支持可选的元数据过滤和结果数量限制"""
    def similarity_search(
        self,
        query_vector: list[float],
        *,
        top_k: int,
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """根据查询向量进行相似度搜索，返回相似的文档块列表。支持可选的元数据过滤和结果数量限制。"""
        self.ensure_collection()
        filter_expression = self._build_filter_expression(metadata_filter)
        output_fields = [
            "text",
            "file_name",
            "file_path",
            "chunk_index",
            "source_type",
            "created_at",
        ]

        try:
            self.client.load_collection(self.collection_name)
            raw_results = self.client.search(
                collection_name=self.collection_name,
                data=[query_vector],
                anns_field="vector",
                search_params={"metric_type": self.metric_type, "params": {}},
                limit=top_k,
                filter=filter_expression,
                output_fields=output_fields,
            )
        except Exception as exc:
            logger.exception("Milvus similarity search failed")
            raise VectorStoreError(
                "Milvus 相似度检索失败",
                detail={"collection": self.collection_name, "error": str(exc)},
            ) from exc

        results: list[SearchResult] = []
        for item in raw_results[0] if raw_results else []:
            entity = item.get("entity", {})
            metadata = {field: entity.get(field) for field in output_fields if field != "text"}
            results.append(
                SearchResult(
                    text=entity.get("text", ""),
                    score=float(item.get("distance", 0.0)),
                    metadata=metadata,
                )
            )
        return results

    def _build_filter_expression(self, metadata_filter: dict[str, Any] | None) -> str | None:
        if not metadata_filter:
            return None

        expressions: list[str] = []
        for key, value in metadata_filter.items():
            if key not in self.filterable_fields or value is None:
                continue
            if isinstance(value, str):
                escaped = self._escape_string(value)
                expressions.append(f'{key} == "{escaped}"')
            elif isinstance(value, bool):
                expressions.append(f"{key} == {str(value).lower()}")
            elif isinstance(value, int | float):
                expressions.append(f"{key} == {value}")

        return " and ".join(expressions) if expressions else None

    def _chunk_id(self, chunk: DocumentChunk) -> str:
        file_path = os.path.normcase(os.path.normpath(str(chunk.metadata["file_path"])))
        chunk_index = int(chunk.metadata["chunk_index"])
        identity = f"{file_path}\0{chunk_index}".encode("utf-8")
        return sha256(identity).hexdigest()

    def _escape_string(self, value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"')
