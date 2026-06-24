from app.rag.document_loader import ParsedDocument
from app.rag.text_splitter import ChunkSplitter


def test_chunk_metadata_is_preserved() -> None:
    splitter = ChunkSplitter(chunk_size=20, chunk_overlap=5)
    document = ParsedDocument(
        text="第一段内容。第二段内容。第三段内容。",
        metadata={
            "file_name": "demo.md",
            "file_path": "docs/demo.md",
            "source_type": "md",
            "created_at": "2026-01-01T00:00:00+00:00",
        },
    )

    chunks = splitter.split_documents([document])

    assert chunks
    assert chunks[0].metadata["file_name"] == "demo.md"
    assert chunks[0].metadata["chunk_index"] == 0
