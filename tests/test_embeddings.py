import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from openai import RateLimitError

from engimind.core.config import Settings
from engimind.retrieval.embeddings import embed_texts


class EmbeddingTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings(
            openai_api_key="test-key",
            embedding_model="all-minilm",
            chat_model="test-chat",
            chunk_size=800,
            chunk_overlap=100,
            top_k=4,
            chroma_path=".chroma",
            max_distance=0.5,
            embedding_provider="ollama",
            ollama_base_url="http://localhost:11434",
        )

    def test_ollama_provider_uses_local_embed_endpoint(self):
        payload = {"embeddings": [[0.1, 0.2], [0.3, 0.4]]}
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=None)
        response.read.return_value = json.dumps(payload).encode("utf-8")
        client = Mock()
        request_status = Mock()

        with patch("engimind.retrieval.embeddings.urlopen", return_value=response) as urlopen:
            vectors = embed_texts(
                client,
                self.settings,
                ["first row", "second row"],
                on_retry=request_status,
            )

        self.assertEqual(vectors, payload["embeddings"])
        request = urlopen.call_args.args[0]
        self.assertEqual(request.full_url, "http://localhost:11434/api/embed")
        self.assertEqual(
            json.loads(request.data),
            {"model": "all-minilm", "input": ["first row", "second row"]},
        )
        client.embeddings.create.assert_not_called()
        request_status.assert_not_called()

    def test_retries_rate_limit_using_server_delay_and_reports_status(self):
        error = RateLimitError.__new__(RateLimitError)
        Exception.__init__(error, "rate limited")
        object.__setattr__(
            error,
            "response",
            SimpleNamespace(headers={"retry-after-ms": "217"}),
        )
        error.body = {"error": {"code": "rate_limit_exceeded"}}
        response = SimpleNamespace(
            data=[SimpleNamespace(index=0, embedding=[0.1])]
        )
        client = Mock()
        client.embeddings.create.side_effect = [error, response]
        settings = Settings(
            openai_api_key="test-key",
            embedding_model="test-embedding",
            chat_model="test-chat",
            chunk_size=800,
            chunk_overlap=100,
            top_k=4,
            chroma_path=".chroma",
            max_distance=0.5,
        )
        status = Mock()

        with patch("engimind.retrieval.embeddings.time.sleep") as sleep:
            vectors = embed_texts(
                client,
                settings,
                ["one row"],
                on_retry=status,
            )

        self.assertEqual(vectors, [[0.1]])
        sleep.assert_called_once_with(0.217)
        status.assert_called_once()
        self.assertIn("retry 1/5", status.call_args.args[0])
        self.assertEqual(client.embeddings.create.call_count, 2)


if __name__ == "__main__":
    unittest.main()
