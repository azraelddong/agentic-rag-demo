from datetime import UTC, datetime
import logging
from pathlib import Path

from fastapi import UploadFile

from app.core.config import Settings
from app.core.exceptions import DocumentProcessingError
from app.llm.embedding_model import EmbeddingModel
from app.rag.document_loader import DocumentLoader
from app.rag.text_splitter import ChunkSplitter, DocumentChunk
from app.rag.vector_store import MilvusVectorStore
from app.schemas.document_schema import DocumentIndexResponse, DocumentUploadResponse

logger = logging.getLogger(__name__)

"""文档service"""
class DocumentService:
    """Document upload and indexing application service."""

    """初始化构造函数"""
    def __init__(
        self,
        *,
        settings: Settings,
        loader: DocumentLoader,
        splitter: ChunkSplitter,
        embedding_model: EmbeddingModel,
        vector_store: MilvusVectorStore,
    ) -> None:
        self.settings = settings
        self.loader = loader
        self.splitter = splitter
        self.embedding_model = embedding_model
        self.vector_store = vector_store

    """上传文档"""
    async def upload_document(self, file: UploadFile) -> DocumentUploadResponse:
        filename = self._safe_filename(file.filename)
        suffix = Path(filename).suffix.lower().lstrip(".")
        if suffix not in self.settings.allowed_document_extensions:
            raise DocumentProcessingError(
                f"不支持的文档类型: .{suffix}",
                detail={"allowed": self.settings.allowed_document_extensions},
            )

        content = await file.read()
        if not content:
            raise DocumentProcessingError("上传文件为空")
        if len(content) > self.settings.max_upload_bytes:
            raise DocumentProcessingError(
                "上传文件超过大小限制",
                detail={"max_mb": self.settings.upload_max_mb},
            )

        docs_path = self.settings.docs_path
        docs_path.mkdir(parents=True, exist_ok=True)
        target_path = self._deduplicate_path(docs_path / filename)
        target_path.write_bytes(content)
        logger.info("Uploaded document saved: %s", target_path)

        return DocumentUploadResponse(
            file_name=target_path.name,
            file_path=str(target_path.resolve()),
            size_bytes=len(content),
        )

    """分隔文档建立索引"""
    def index_documents(self, *, docs_dir: str | None = None, rebuild: bool = False) -> DocumentIndexResponse:
        source_dir = Path(docs_dir) if docs_dir else self.settings.docs_path
        if not source_dir.is_absolute():
            source_dir = self.settings.docs_path.parent / source_dir

        if rebuild:
            self.vector_store.drop_collection()

        documents = self.loader.load_directory(source_dir)
        chunks = self.splitter.split_documents(documents)
        if not chunks:
            return DocumentIndexResponse(
                indexed_files=len(documents),
                chunk_count=0,
                inserted_count=0,
                collection_name=self.settings.milvus_collection_name,
                message="没有可索引的文档内容",
            )

        self.vector_store.delete_chunks_for_files(
            [str(document.metadata["file_path"]) for document in documents]
        )

        inserted_count = 0
        batch_size = self.settings.embedding_batch_size
        for batch in self._batch_chunks(chunks, batch_size):
            texts = [chunk.content for chunk in batch]
            vectors = self.embedding_model.embed_texts(texts)
            inserted_count += self.vector_store.add_chunks(batch, vectors)

        logger.info(
            "Indexed documents: files=%s chunks=%s inserted=%s",
            len(documents),
            len(chunks),
            inserted_count,
        )
        return DocumentIndexResponse(
            indexed_files=len(documents),
            chunk_count=len(chunks),
            inserted_count=inserted_count,
            collection_name=self.settings.milvus_collection_name,
            message="索引建立完成",
        )

    def _batch_chunks(self, chunks: list[DocumentChunk], batch_size: int) -> list[list[DocumentChunk]]:
        """将文档块分批处理"""
        return [chunks[index : index + batch_size] for index in range(0, len(chunks), batch_size)]

    def _safe_filename(self, filename: str | None) -> str:
        """生成安全的文件名"""
        if not filename:
            raise DocumentProcessingError("缺少上传文件名")
        safe_name = Path(filename).name
        if safe_name in {"", ".", ".."}:
            raise DocumentProcessingError("非法上传文件名")
        return safe_name

    """避免文件名冲突，生成唯一的文件路径"""
    def _deduplicate_path(self, path: Path) -> Path:
        if not path.exists():
            return path
        timestamp = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
        return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")
