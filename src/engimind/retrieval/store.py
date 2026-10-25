import re
from dataclasses import dataclass
from typing import Callable

import chromadb
from openai import OpenAI

from engimind.ingestion.chunking import Chunk
from engimind.core.config import Settings
from engimind.observability import traced_tool
from engimind.retrieval.embeddings import embed_texts

_UPSERT_BATCH_SIZE = 64


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    page: int
    source: str
    distance: float
    section: str = ""
    end_page: int | None = None


def chroma_client(settings: Settings) -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=settings.chroma_path)


def collection_name(file_hash: str) -> str:
    # Chroma names: 3–63 chars, alphanumerics and [._-]
    return f"pdf_{file_hash[:32]}"


def get_or_create_collection(
    client: chromadb.PersistentClient,
    name: str,
    *,
    embedding_provider: str = "openai",
    embedding_model: str = "",
):
    return client.get_or_create_collection(
        name=name,
        metadata={
            "hnsw:space": "cosine",
            "embedding_provider": embedding_provider,
            "embedding_model": embedding_model,
        },
    )


def validate_collection_embedding(
    collection,
    settings: Settings,
) -> None:
    metadata = collection.metadata or {}
    if not isinstance(metadata, dict):
        metadata = {}
    embedding_provider = getattr(settings, "embedding_provider", "openai")
    embedding_model = getattr(settings, "embedding_model", "")
    stored_provider = metadata.get("embedding_provider")
    stored_model = metadata.get("embedding_model")
    if stored_provider is None and embedding_provider == "openai":
        return
    if (
        stored_provider != embedding_provider
        or stored_model != embedding_model
    ):
        raise ValueError(
            "The indexed collection uses a different embedding provider/model "
            f"({stored_provider or 'legacy OpenAI'} / {stored_model or 'unknown'}) "
            f"than the current selection ({embedding_provider} / "
            f"{embedding_model}). Re-index this document with the "
            "current embedding selection."
        )


def upsert_chunks(
    client: chromadb.PersistentClient,
    openai_client: OpenAI,
    settings: Settings,
    *,
    name: str,
    chunks: list[Chunk],
    progress_callback: Callable[[int, int], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
) -> int:
    try:
        client.delete_collection(name)
    except Exception:
        pass
    collection = get_or_create_collection(
        client,
        name,
        embedding_provider=getattr(settings, "embedding_provider", "openai"),
        embedding_model=getattr(settings, "embedding_model", ""),
    )
    for start in range(0, len(chunks), _UPSERT_BATCH_SIZE):
        batch = chunks[start : start + _UPSERT_BATCH_SIZE]
        embeddings = embed_texts(
            openai_client,
            settings,
            [chunk.text for chunk in batch],
            on_retry=status_callback,
        )
        collection.upsert(
            ids=[chunk.chunk_id for chunk in batch],
            embeddings=embeddings,
            documents=[chunk.text for chunk in batch],
            metadatas=[
                {
                    "page": chunk.page,
                    "end_page": chunk.end_page if chunk.end_page is not None else chunk.page,
                    "source": chunk.source,
                    "chunk_id": chunk.chunk_id,
                    "section": chunk.section,
                }
                for chunk in batch
            ],
        )
        if progress_callback is not None:
            progress_callback(min(start + len(batch), len(chunks)), len(chunks))
    return collection.count()


def query_rewrite(question: str) -> str:
    """Expand common HR policy synonyms before semantic retrieval.

    The goal is to preserve the original intent while surfacing equivalent terms
    like vacation/holiday/leave and common policy phrases.
    """
    cleaned = (question or "").strip()
    if not cleaned:
        return ""

    rewritten = re.sub(
        r"\b(vacation|vacations|holiday|holidays|time off|paid time off|pto)\b",
        "leave",
        cleaned,
        flags=re.IGNORECASE,
    )
    rewritten = re.sub(
        r"\b(annual leave|leave policy|leave)\b",
        "annual leave",
        rewritten,
        flags=re.IGNORECASE,
    )

    if re.search(r"\b(leave|vacation|holiday|pto)\b", rewritten, flags=re.IGNORECASE):
        rewritten = re.sub(
            r"\b(policy)\b",
            "annual leave policy",
            rewritten,
            count=1,
            flags=re.IGNORECASE,
        )

    # Preserve numeric constraints like years and durations.
    rewritten = re.sub(r"\s+", " ", rewritten).strip()
    return rewritten


def _tokenize(text: str) -> set[str]:
    return {token for token in re.findall(r"\b\w+\b", text.lower()) if token}


def _similarity_ratio(left: str, right: str) -> float:
    left_tokens = _tokenize(left)
    right_tokens = _tokenize(right)
    if not left_tokens or not right_tokens:
        return 0.0
    union = left_tokens | right_tokens
    if not union:
        return 0.0
    return len(left_tokens & right_tokens) / len(union)


def _remove_near_duplicates(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    deduped: list[RetrievedChunk] = []
    for chunk in chunks:
        if any(_similarity_ratio(chunk.text, kept.text) >= 0.5 for kept in deduped):
            continue
        deduped.append(chunk)
    return deduped


def _lexical_relevance(question: str, chunk: RetrievedChunk) -> float:
    question_tokens = _tokenize(question)
    chunk_tokens = _tokenize(chunk.text)
    if not question_tokens or not chunk_tokens:
        return 0.0

    overlap = question_tokens & chunk_tokens
    if not overlap:
        return 0.0

    section_tokens = _tokenize(chunk.section or "")
    section_bonus = 0.35 * len(question_tokens & section_tokens)
    phrase_bonus = 0.25 if any(token in chunk.text.lower() for token in question_tokens) else 0.0
    coverage = len(overlap) / len(question_tokens)
    tf_score = min(1.0, len(overlap) / max(len(chunk_tokens), 1))
    return coverage + tf_score + section_bonus + phrase_bonus


def hybrid_rank_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []

    question_tokens = _tokenize(question)
    if not question_tokens:
        return sorted(chunks, key=lambda chunk: chunk.distance)

    def score(chunk: RetrievedChunk) -> tuple[float, int, int, float]:
        lexical = _lexical_relevance(question, chunk)
        vector_score = 1.0 - chunk.distance
        section_overlap = len(question_tokens & _tokenize(chunk.section or ""))
        text_overlap = len(question_tokens & _tokenize(chunk.text))
        hybrid_score = (0.65 * vector_score) + (0.35 * lexical)
        return (hybrid_score, section_overlap, text_overlap, -chunk.distance)

    ranked = sorted(chunks, key=score, reverse=True)
    return _remove_near_duplicates(ranked)


def rerank_chunks(question: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    return hybrid_rank_chunks(question, chunks)


@traced_tool(
    "vector_search",
    describe=lambda *a, **kw: {"collection": kw.get("name"), "k": kw.get("k")},
    summarize=lambda chunks: {"results": len(chunks)},
    when=lambda *a, **kw: not isinstance(kw.get("name"), list),
)
def query_collection(
    client: chromadb.PersistentClient,
    openai_client: OpenAI,
    settings: Settings,
    *,
    name: str | list[str],
    question: str,
    k: int,
    expand_query: bool = True,
) -> list[RetrievedChunk]:
    if isinstance(name, list):
        retrieved: list[RetrievedChunk] = []
        for collection_name in dict.fromkeys(name):
            retrieved.extend(
                query_collection(
                    client,
                    openai_client,
                    settings,
                    name=collection_name,
                    question=question,
                    k=k,
                    expand_query=expand_query,
                )
            )
        return hybrid_rank_chunks(question, retrieved)[:k]

    collection = client.get_collection(name)
    validate_collection_embedding(collection, settings)
    search_question = query_rewrite(question) if expand_query else question
    query_vec = embed_texts(openai_client, settings, [search_question])[0]
    result = collection.query(
        query_embeddings=[query_vec],
        n_results=min(k, max(collection.count(), 1)),
        include=["documents", "metadatas", "distances"],
    )
    docs = (result.get("documents") or [[]])[0]
    metas = (result.get("metadatas") or [[]])[0]
    distances = (result.get("distances") or [[]])[0]
    retrieved: list[RetrievedChunk] = []
    for doc, meta, dist in zip(docs, metas, distances):
        if doc is None:
            continue
        retrieved.append(
            RetrievedChunk(
                text=doc,
                page=int(meta.get("page", 0)),
                source=str(meta.get("source", "")),
                distance=float(dist),
                section=str(meta.get("section", "")),
                end_page=int(meta.get("end_page", meta.get("page", 0))),
            )
        )
    return hybrid_rank_chunks(search_question, retrieved)
