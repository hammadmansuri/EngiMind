from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from pydantic import SecretStr

from engimind.core.config import Settings
from engimind.retrieval.embeddings import embed_texts
from engimind.application.rag import (
    RagResult,
    _chunk_location,
    _should_refuse,
    format_history_context,
    record_answer,
)
from engimind.observability import agent_run, tool_call
from engimind.observability.langchain_callback import UsageCallbackHandler
from engimind.retrieval.store import (
    RetrievedChunk,
    chroma_client,
    hybrid_rank_chunks,
    query_rewrite,
    validate_collection_embedding,
)


SYSTEM_PROMPT = (
    "You are a careful assistant answering from a single indexed document. Use only "
    "the numbered context snippets. Cite PDF page numbers as [p. N] or CSV row "
    "numbers as [row N] after supported claims. If the context does not contain the "
    "answer, say you cannot find it in the document. CSV context labels each "
    "original row explicitly; cite the exact row number supporting a claim, not a "
    "chunk's row range. Do not use outside knowledge or invent citations."
)


@dataclass(frozen=True)
class LangChainConfig:
    api_key: str
    embedding_model: str
    chat_model: str
    persist_directory: str
    collection_name: str = "pdf_langchain"
    embedding_provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"


def build_langchain_config(
    *,
    api_key: str,
    embedding_model: str,
    chat_model: str,
    persist_directory: str,
    collection_name: str = "pdf_langchain",
    embedding_provider: str = "openai",
    ollama_base_url: str = "http://localhost:11434",
) -> LangChainConfig:
    return LangChainConfig(
        api_key=api_key,
        embedding_model=embedding_model,
        chat_model=chat_model,
        persist_directory=persist_directory,
        collection_name=collection_name,
        embedding_provider=embedding_provider,
        ollama_base_url=ollama_base_url,
    )


def make_embeddings(api_key: str, model: str):
    return OpenAIEmbeddings(model=model, api_key=SecretStr(api_key))


class OllamaLangChainEmbeddings(Embeddings):
    def __init__(self, settings: Settings):
        self.settings = settings

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return embed_texts(None, self.settings, texts)

    def embed_query(self, text: str) -> list[float]:
        return embed_texts(None, self.settings, [text])[0]


def documents_from_chunks(chunks: list) -> list[Document]:
    """Convert our stored chunks into LangChain documents with metadata kept intact."""
    documents: list[Document] = []
    for chunk in chunks:
        documents.append(
            Document(
                page_content=chunk.text,
                metadata={
                    "page": getattr(chunk, "page", 0),
                    "end_page": (
                        getattr(chunk, "end_page", None)
                        or getattr(chunk, "page", 0)
                    ),
                    "source": getattr(chunk, "source", ""),
                    "section": getattr(chunk, "section", "general"),
                },
            )
        )
    return documents


def create_vectorstore(config: LangChainConfig, settings: Settings) -> Chroma:
    if config.embedding_provider == "ollama":
        embeddings = OllamaLangChainEmbeddings(settings)
    else:
        embeddings = make_embeddings(config.api_key, config.embedding_model)
    return Chroma(
        collection_name=config.collection_name,
        persist_directory=config.persist_directory,
        embedding_function=embeddings,
    )


def add_chunks_to_store(store: Chroma, chunks: list) -> None:
    store.add_documents(documents_from_chunks(chunks))


def build_langchain_rag_chain(config: LangChainConfig, settings: Settings):
    """Create the prompt and LLM chain used for the LangChain answer path."""
    store = create_vectorstore(config, settings)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Question: {question}\n\nConversation history:\n{history}\n\n"
                "Context snippets:\n{context}",
            ),
        ]
    )
    llm = ChatOpenAI(
        model=config.chat_model,
        temperature=0,
        api_key=SecretStr(config.api_key),
        callbacks=[UsageCallbackHandler(config.chat_model)],
    )
    chain = prompt | llm | StrOutputParser()
    return chain, store, prompt


def answer_question_langchain(
    question: str,
    collection: str | list[str],
    settings,
    chat_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    collections = [collection] if isinstance(collection, str) else collection
    with agent_run(
        "langchain_rag",
        input_text=question,
        collections=len(collections),
        top_k=getattr(settings, "top_k", None),
        chat_model=getattr(settings, "chat_model", None),
    ) as run:
        result = _answer_question_langchain(
            question, collection, settings, chat_history
        )
        record_answer(run, result)
    return result


def _answer_question_langchain(
    question: str,
    collection: str | list[str],
    settings,
    chat_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    collection_names = [collection] if isinstance(collection, str) else collection
    chain = prompt = None
    candidates = []
    retrieval_started = perf_counter()
    for collection_name in collection_names:
        validate_collection_embedding(
            chroma_client(settings).get_collection(collection_name),
            settings,
        )
        config = build_langchain_config(
            api_key=settings.openai_api_key,
            embedding_model=settings.embedding_model,
            chat_model=settings.chat_model,
            persist_directory=settings.chroma_path,
            collection_name=collection_name,
            embedding_provider=settings.embedding_provider,
            ollama_base_url=settings.ollama_base_url,
        )
        current_chain, current_store, current_prompt = build_langchain_rag_chain(
            config, settings
        )
        if chain is None:
            chain, prompt = current_chain, current_prompt
        with tool_call(
            "vector_search", collection=collection_name, k=settings.top_k
        ) as tool_outcome:
            found = current_store.similarity_search_with_score(
                query_rewrite(question), k=settings.top_k
            )
            tool_outcome["results"] = len(found)
        candidates.extend(found)

    # Keep the same retrieval safety as the custom path: rewrite the query and re-rank.
    chunks = [
        RetrievedChunk(
            text=document.page_content,
            page=int(document.metadata.get("page", 0)),
            source=str(document.metadata.get("source", "")),
            distance=float(distance),
            section=str(document.metadata.get("section", "")),
            end_page=int(document.metadata.get("end_page", document.metadata.get("page", 0))),
        )
        for document, distance in candidates
    ]
    chunks = hybrid_rank_chunks(question, chunks)[: settings.top_k]
    retrieval_ms = (perf_counter() - retrieval_started) * 1000
    assert chain is not None and prompt is not None

    context_lines = []
    for index, chunk in enumerate(chunks, start=1):
        context_lines.append(
            f"[{index}] source={chunk.source} {_chunk_location(chunk)} "
            f"section={chunk.section or 'general'} "
            f"cosine_distance={chunk.distance:.4f}\n{chunk.text}"
        )
    context = "\n\n".join(context_lines) or "(none retrieved)"
    history = format_history_context(chat_history)
    prompt_messages = prompt.format_messages(
        question=question,
        history=history,
        context=context,
    )
    prompt_text = "\n\n".join(
        f"{message.type.upper()}: {message.content}" for message in prompt_messages
    )

    if _should_refuse(chunks, settings.max_distance):
        return RagResult(
            answer=(
                "I cannot find this in the indexed document with enough confidence. "
                "Try rephrasing, or increase k / inspect the retrieved chunks in Debug."
            ),
            chunks=chunks,
            prompt=prompt_text,
            refused=True,
            timings_ms={"retrieval": retrieval_ms, "generation": 0.0},
        )

    generation_started = perf_counter()
    answer = chain.invoke({"question": question, "history": history, "context": context}).strip()
    return RagResult(
        answer=answer,
        chunks=chunks,
        prompt=prompt_text,
        refused=False,
        timings_ms={
            "retrieval": retrieval_ms,
            "generation": (perf_counter() - generation_started) * 1000,
        },
    )
