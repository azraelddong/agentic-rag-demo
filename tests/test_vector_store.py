from app.core.config import Settings
from app.rag.text_splitter import DocumentChunk
from app.rag.vector_store import MilvusVectorStore


class FakeMilvusClient:
    def __init__(self) -> None:
        self.deleted_filters: list[str] = []
        self.upserted_batches: list[list[dict]] = []

    def has_collection(self, collection_name: str) -> bool:
        return True

    def delete(self, *, collection_name: str, filter: str) -> None:
        self.deleted_filters.append(filter)

    def upsert(self, *, collection_name: str, data: list[dict]) -> None:
        self.upserted_batches.append(data)


def make_chunk(*, file_path: str, chunk_index: int, content: str = "content") -> DocumentChunk:
    return DocumentChunk(
        content=content,
        metadata={
            "file_name": "demo.md",
            "file_path": file_path,
            "chunk_index": chunk_index,
            "source_type": "md",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )


def test_add_chunks_uses_stable_ids_and_upsert() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings())
    store._client = client
    chunk = make_chunk(file_path=r"E:\docs\demo.md", chunk_index=2)

    store.add_chunks([chunk], [[0.1, 0.2]])
    store.add_chunks([chunk], [[0.3, 0.4]])

    first_record = client.upserted_batches[0][0]
    second_record = client.upserted_batches[1][0]
    assert first_record["id"] == second_record["id"]
    assert len(first_record["id"]) == 64
    assert second_record["vector"] == [0.3, 0.4]


def test_chunk_id_changes_for_a_different_chunk_index() -> None:
    store = MilvusVectorStore(Settings())
    first = make_chunk(file_path=r"E:\docs\demo.md", chunk_index=1)
    second = make_chunk(file_path=r"E:\docs\demo.md", chunk_index=2)

    assert store._chunk_id(first) != store._chunk_id(second)


def test_delete_chunks_for_files_deduplicates_and_escapes_paths() -> None:
    client = FakeMilvusClient()
    store = MilvusVectorStore(Settings())
    store._client = client
    file_path = 'E:\\docs\\the "demo".md'

    store.delete_chunks_for_files([file_path, file_path])

    assert client.deleted_filters == [
        'file_path == "E:\\\\docs\\\\the \\"demo\\".md"'
    ]
