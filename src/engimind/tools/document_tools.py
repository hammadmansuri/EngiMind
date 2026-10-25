from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from openai import OpenAI

from engimind.application.rag import _chunk_location, answer_question
from engimind.core.config import Settings, load_settings
from engimind.retrieval.store import (
    RetrievedChunk,
    chroma_client,
    query_collection,
    validate_collection_embedding,
)

MAX_QUERY_CHARS = 2000
MAX_K = 20


class DocumentTools:
    """Read-only operations over documents that are already indexed.

    The methods return plain, JSON-serializable data and know nothing about MCP
    or any other protocol, so the same tools can back an MCP server, an HTTP
    endpoint, or an LLM function-calling loop.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        settings_factory: Callable[[], Settings] = load_settings,
        openai_client: OpenAI | None = None,
    ) -> None:
        self._settings = settings
        self._settings_factory = settings_factory
        self._openai_client = openai_client

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = self._settings_factory()
        return self._settings

    def _openai(self) -> OpenAI:
        if self._openai_client is None:
            self._openai_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._openai_client

    def _embedding_client(self) -> OpenAI | None:
        # Ollama embeddings are local HTTP calls and need no OpenAI client.
        if self.settings.embedding_provider == "ollama":
            return None
        return self._openai()

    def list_documents(self) -> list[dict[str, Any]]:
        """List indexed documents.

        Each entry shows the source name, chunk count, embedding model, and whether
        it is searchable with the server's current embedding settings. Documents
        built with a different embedding provider/model must be re-indexed first.
        """
        client = chroma_client(self.settings)
        documents: list[dict[str, Any]] = []
        for collection in client.list_collections():
            sample = collection.get(limit=1, include=["metadatas"])
            metadatas = sample.get("metadatas") or []
            if not metadatas:
                continue
            metadata = collection.metadata or {}
            documents.append(
                {
                    "source": str(metadatas[0].get("source", "")),
                    "collection": collection.name,
                    "chunks": collection.count(),
                    "embedding_provider": metadata.get("embedding_provider", "openai"),
                    "embedding_model": metadata.get("embedding_model", ""),
                    "searchable": self._is_searchable(collection),
                }
            )
        return sorted(documents, key=lambda item: (item["source"].casefold(), item["collection"]))

    def _is_searchable(self, collection: Any) -> bool:
        try:
            validate_collection_embedding(collection, self.settings)
        except ValueError:
            return False
        return True

    def search_documents(
        self,
        query: str,
        k: int = 4,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Find the passages most relevant to a query, without generating an answer.

        Args:
            query: Natural-language search text.
            k: Number of passages to return (1-20).
            source: Optional document name (see list_documents) to restrict the search.
        """
        query = _validated_text(query, "query")
        collections = self._collections_for(source)
        chunks = query_collection(
            chroma_client(self.settings),
            self._embedding_client(),
            self.settings,
            name=collections,
            question=query,
            k=_clamp_k(k),
        )
        return {
            "query": query,
            "results": [_chunk_payload(chunk) for chunk in chunks],
        }

    def ask_document(
        self,
        question: str,
        source: str | None = None,
        k: int | None = None,
    ) -> dict[str, Any]:
        """Answer a question using only the indexed documents, with source citations.

        Args:
            question: The question to answer.
            source: Optional document name (see list_documents) to restrict the answer.
            k: Optional number of passages to retrieve (1-20); defaults to the app setting.
        """
        question = _validated_text(question, "question")
        collections = self._collections_for(source)
        settings = self.settings
        if k is not None:
            settings = replace(settings, top_k=_clamp_k(k))
        result = answer_question(question, collections, settings, self._openai())
        return {
            "answer": result.answer,
            "refused": result.refused,
            "sources": [_chunk_payload(chunk) for chunk in result.chunks],
            "timings_ms": result.timings_ms or {},
        }

    def _collections_for(self, source: str | None) -> list[str]:
        documents = self.list_documents()
        if not documents:
            raise ValueError("No documents are indexed. Index a PDF or CSV first.")
        if source is not None and source.strip():
            wanted = source.strip().casefold()
            documents = [item for item in documents if item["source"].casefold() == wanted]
            if not documents:
                available = ", ".join(sorted({item["source"] for item in self.list_documents()}))
                raise ValueError(f"Unknown source '{source}'. Indexed documents: {available}")
        searchable = [item["collection"] for item in documents if item["searchable"]]
        if not searchable:
            raise ValueError(
                "No matching document was indexed with the current embedding setting "
                f"({self.settings.embedding_provider} / {self.settings.embedding_model}). "
                "Re-index it with that setting, or change EMBEDDING_PROVIDER."
            )
        return searchable


def _validated_text(value: str, label: str) -> str:
    text = (value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty.")
    if len(text) > MAX_QUERY_CHARS:
        raise ValueError(f"{label} must be at most {MAX_QUERY_CHARS} characters.")
    return text


def _clamp_k(k: int) -> int:
    return max(1, min(int(k), MAX_K))


def _chunk_payload(chunk: RetrievedChunk) -> dict[str, Any]:
    return {
        "source": chunk.source,
        "location": _chunk_location(chunk),
        "section": chunk.section,
        "distance": round(chunk.distance, 4),
        "text": chunk.text,
    }
