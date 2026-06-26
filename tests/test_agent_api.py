"""Integration tests for the Agent API endpoint."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app


def _mock_rag_chain_ask(return_value=None):
    """Patch RAGChain.ask so the Agent endpoint doesn't need real infra."""
    if return_value is None:
        return_value = ("测试回答", [])
    return patch(
        "app.rag.rag_chain.RAGChain.ask",
        return_value=return_value,
    )


class TestAgentApiResponseShape:
    """Verify the /api/agent/ask response contract."""

    def test_response_has_answer_sources_trace(self) -> None:
        with _mock_rag_chain_ask():
            client = TestClient(app)
            resp = client.post("/api/agent/ask", json={"question": "测试"})

        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
        assert "trace" in body
        assert body["trace"]["plan"] == "rag_search"
        assert body["trace"]["iterations"] >= 1
        assert "steps" in body["trace"]
        assert body["trace"]["steps"][0]["node"] == "plan"

    def test_question_empty_is_rejected(self) -> None:
        client = TestClient(app)
        resp = client.post("/api/agent/ask", json={"question": ""})
        assert resp.status_code == 422

    def test_question_too_long_is_rejected(self) -> None:
        client = TestClient(app)
        resp = client.post("/api/agent/ask", json={"question": "x" * 2001})
        assert resp.status_code == 422

    def test_top_k_out_of_range_is_rejected(self) -> None:
        client = TestClient(app)
        resp = client.post(
            "/api/agent/ask",
            json={"question": "ok", "top_k": 0},
        )
        assert resp.status_code == 422

    def test_metadata_filter_accepted(self) -> None:
        with _mock_rag_chain_ask():
            client = TestClient(app)
            resp = client.post(
                "/api/agent/ask",
                json={
                    "question": "测试",
                    "metadata_filter": {"source_type": "md"},
                },
            )
        assert resp.status_code == 200

    def test_response_trace_shows_retry_when_first_pass_has_no_sources(
        self,
    ) -> None:
        with patch(
            "app.rag.rag_chain.RAGChain.ask",
            side_effect=[
                ("知识库中未找到相关信息", []),
                ("第二轮回答", []),
            ],
        ):
            client = TestClient(app)
            resp = client.post(
                "/api/agent/ask", json={"question": "测试", "top_k": 5}
            )

        assert resp.status_code == 200
        body = resp.json()
        assert body["trace"]["iterations"] == 2
        assert "prepare_retry" in [
            step["node"] for step in body["trace"]["steps"]
        ]


class TestChatApiUnaffected:
    """Confirm /api/chat/ask still works after Agent changes."""

    def test_chat_ask_still_accepts_requests(self) -> None:
        with _mock_rag_chain_ask():
            client = TestClient(app)
            resp = client.post("/api/chat/ask", json={"question": "测试"})

        assert resp.status_code == 200
        body = resp.json()
        assert "answer" in body
        assert "sources" in body
