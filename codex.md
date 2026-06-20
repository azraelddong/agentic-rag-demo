# Codex Notes

## Current scope

This project is initialized as a phase-1 Basic RAG application. Agentic RAG is intentionally reserved but not wired into the runtime path.

## Runtime flow

1. Upload or place documents in `docs/`.
2. Parse `txt`, `md`, and `pdf`.
3. Split text into chunks with `chunk_size=800` and `chunk_overlap=100`.
4. Embed chunks with an OpenAI-compatible embedding model.
5. Replace existing chunks for each file and upsert vectors with stable chunk IDs.
6. Retrieve relevant chunks for a user question.
7. Build a grounded RAG prompt.
8. Generate an answer through an OpenAI-compatible chat model.
9. Return answer and sources.

## Extension points

- `app/llm/embedding_model.py`: add local bge-m3 implementation.
- `app/rag/reranker.py`: add bge-reranker implementation.
- `app/rag/vector_store.py`: extend metadata filters and document lifecycle operations.
- `app/agent/graph.py`: build LangGraph Agentic RAG nodes.
- `app/services/`: add async indexing jobs, auth, tenant isolation, and observability.
