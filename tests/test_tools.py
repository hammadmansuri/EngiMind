import asyncio
import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

from engimind.application.rag import RagResult
from engimind.core.config import Settings
from engimind.mcp_server.server import _report_expected_errors, create_server
from engimind.retrieval.store import RetrievedChunk, chroma_client, get_or_create_collection
from engimind.tools import DocumentTools
from mcp.server.mcpserver.exceptions import ToolError


def make_settings(path: str, **overrides) -> Settings:
    values = dict(
        openai_api_key="test-key",
        embedding_model="text-embedding-3-small",
        chat_model="test-chat",
        chunk_size=800,
        chunk_overlap=150,
        top_k=4,
        chroma_path=path,
        max_distance=0.55,
    )
    values.update(overrides)
    return Settings(**values)


def add_document(path: str, name: str, source: str, *, provider: str, model: str) -> None:
    collection = get_or_create_collection(
        chroma_client(make_settings(path)),
        name,
        embedding_provider=provider,
        embedding_model=model,
    )
    collection.add(
        ids=[f"{name}-1"],
        embeddings=[[0.1, 0.2, 0.3]],
        documents=["sample text"],
        metadatas=[{"source": source, "page": 1, "end_page": 1}],
    )


class DocumentToolsTests(unittest.TestCase):
    def setUp(self):
        self._dir = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self._dir.cleanup)
        self.path = self._dir.name
        add_document(
            self.path, "pdf_openai", "report.pdf", provider="openai", model="text-embedding-3-small"
        )
        add_document(
            self.path, "pdf_ollama", "report.pdf", provider="ollama", model="all-minilm"
        )
        self.tools = DocumentTools(make_settings(self.path))

    def test_list_documents_flags_collections_incompatible_with_current_embeddings(self):
        documents = {doc["collection"]: doc for doc in self.tools.list_documents()}

        self.assertTrue(documents["pdf_openai"]["searchable"])
        self.assertFalse(documents["pdf_ollama"]["searchable"])
        self.assertEqual(documents["pdf_ollama"]["chunks"], 1)
        self.assertEqual(documents["pdf_ollama"]["embedding_model"], "all-minilm")

    def test_search_only_queries_compatible_collections(self):
        chunk = RetrievedChunk(text="found", page=2, source="report.pdf", distance=0.12345)
        with patch(
            "engimind.tools.document_tools.query_collection", return_value=[chunk]
        ) as query:
            result = self.tools.search_documents("what is it?", k=3, source="REPORT.pdf")

        self.assertEqual(query.call_args.kwargs["name"], ["pdf_openai"])
        self.assertEqual(query.call_args.kwargs["k"], 3)
        self.assertEqual(
            result["results"],
            [
                {
                    "source": "report.pdf",
                    "location": "page=2",
                    "section": "",
                    "distance": 0.1235,
                    "text": "found",
                }
            ],
        )

    def test_search_clamps_k(self):
        with patch("engimind.tools.document_tools.query_collection", return_value=[]) as query:
            self.tools.search_documents("question", k=500)
            self.tools.search_documents("question", k=-4)

        self.assertEqual([call.kwargs["k"] for call in query.call_args_list], [20, 1])

    def test_rejects_empty_and_oversized_text(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            self.tools.search_documents("   ")
        with self.assertRaisesRegex(ValueError, "at most"):
            self.tools.ask_document("x" * 2001)

    def test_unknown_source_lists_available_documents(self):
        with self.assertRaisesRegex(ValueError, "Unknown source 'nope.pdf'.*report.pdf"):
            self.tools.search_documents("question", source="nope.pdf")

    def test_source_with_only_incompatible_collections_asks_for_reindex(self):
        add_document(
            self.path, "pdf_only_ollama", "local.csv", provider="ollama", model="all-minilm"
        )

        with self.assertRaisesRegex(ValueError, "Re-index"):
            self.tools.search_documents("question", source="local.csv")

    def test_no_documents_indexed(self):
        with TemporaryDirectory(ignore_cleanup_errors=True) as empty:
            tools = DocumentTools(make_settings(empty))
            with self.assertRaisesRegex(ValueError, "No documents are indexed"):
                tools.search_documents("question")

    def test_ask_document_applies_k_and_omits_prompt(self):
        chunk = RetrievedChunk(text="evidence", page=1, source="report.pdf", distance=0.2)
        result = RagResult(
            answer="Yes [p. 1]",
            chunks=[chunk],
            prompt="secret prompt",
            refused=False,
            timings_ms={"retrieval": 1.0, "generation": 2.0},
        )
        tools = DocumentTools(make_settings(self.path), openai_client=object())
        with patch("engimind.tools.document_tools.answer_question", return_value=result) as ask:
            payload = tools.ask_document("Is it?", source="report.pdf", k=7)

        self.assertEqual(ask.call_args.args[1], ["pdf_openai"])
        self.assertEqual(ask.call_args.args[2].top_k, 7)
        self.assertEqual(payload["answer"], "Yes [p. 1]")
        self.assertFalse(payload["refused"])
        self.assertEqual(payload["sources"][0]["location"], "page=1")
        self.assertEqual(payload["timings_ms"], {"retrieval": 1.0, "generation": 2.0})
        self.assertNotIn("prompt", payload)


class McpServerTests(unittest.TestCase):
    def test_registers_three_read_only_tools_with_clean_schemas(self):
        server = create_server(DocumentTools(make_settings(".unused")))

        tools = {tool.name: tool for tool in asyncio.run(server.list_tools())}

        self.assertEqual(
            set(tools), {"list_documents", "search_documents", "ask_document"}
        )
        for tool in tools.values():
            self.assertTrue(tool.annotations.read_only_hint)
            self.assertNotIn("self", tool.input_schema.get("properties", {}))
        self.assertEqual(tools["search_documents"].input_schema["required"], ["query"])
        self.assertEqual(tools["ask_document"].input_schema["required"], ["question"])

    def test_expected_errors_become_tool_errors(self):
        def failing():
            raise ValueError("source is unknown")

        with self.assertRaisesRegex(ToolError, "source is unknown"):
            _report_expected_errors(failing)()

    def test_unexpected_errors_are_not_converted(self):
        def crashing():
            raise KeyError("internal")

        with self.assertRaises(KeyError):
            _report_expected_errors(crashing)()


if __name__ == "__main__":
    unittest.main()
