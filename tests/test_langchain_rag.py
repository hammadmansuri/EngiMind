import unittest
from types import SimpleNamespace
from unittest.mock import patch

from engimind.application.langchain_rag import answer_question_langchain


class FakePrompt:
    def format_messages(self, **values):
        return [SimpleNamespace(type="human", content=values["context"])]


class FakeChain:
    first = FakePrompt()

    def __init__(self):
        self.invocations = []

    def invoke(self, values):
        self.invocations.append(values)
        return "Grounded answer [p. 2]"


class LangChainRagTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(
            openai_api_key="test-key",
            chat_model="test-chat-model",
            chroma_path=".chroma",
            top_k=4,
            max_distance=0.55,
            embedding_provider="openai",
            embedding_model="test-embedding-model",
            ollama_base_url="http://localhost:11434",
        )

    def test_refuses_without_retrieved_evidence(self):
        chain = FakeChain()
        store = SimpleNamespace(similarity_search_with_score=lambda query, k: [])

        with patch(
            "engimind.application.langchain_rag.chroma_client"
        ) as chroma_client, patch(
            "engimind.application.langchain_rag.build_langchain_rag_chain",
            return_value=(chain, store, FakePrompt()),
        ):
            chroma_client.return_value.get_collection.return_value.metadata = {
                "embedding_provider": "openai",
                "embedding_model": "test-embedding-model",
            }
            result = answer_question_langchain(
                "What is the policy?", "pdf_test", self.settings
            )

        self.assertTrue(result.refused)
        self.assertEqual(result.chunks, [])
        self.assertEqual(chain.invocations, [])

    def test_uses_existing_chunk_metadata_and_generates_answer(self):
        chain = FakeChain()
        document = SimpleNamespace(
            page_content="Employees receive 20 days of annual leave.",
            metadata={"page": 2, "source": "policy.pdf", "section": "Annual Leave"},
        )
        store = SimpleNamespace(
            similarity_search_with_score=lambda query, k: [(document, 0.1)]
        )

        with patch(
            "engimind.application.langchain_rag.chroma_client"
        ) as chroma_client, patch(
            "engimind.application.langchain_rag.build_langchain_rag_chain",
            return_value=(chain, store, FakePrompt()),
        ):
            chroma_client.return_value.get_collection.return_value.metadata = {
                "embedding_provider": "openai",
                "embedding_model": "test-embedding-model",
            }
            result = answer_question_langchain(
                "How many days of annual leave?", "pdf_test", self.settings
            )

        self.assertFalse(result.refused)
        self.assertEqual(result.answer, "Grounded answer [p. 2]")
        self.assertEqual(result.chunks[0].page, 2)
        self.assertEqual(result.chunks[0].source, "policy.pdf")
        self.assertIn("page=2", result.prompt)
        self.assertIn("retrieval", result.timings_ms)
        self.assertIn("generation", result.timings_ms)
        self.assertEqual(len(chain.invocations), 1)


if __name__ == "__main__":
    unittest.main()
