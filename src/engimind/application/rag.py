from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any, Callable

from openai import OpenAI

from engimind.ingestion.chunking import chunk_pages
from engimind.core.config import Settings
from engimind.ingestion.csv_loader import CsvTable, csv_table_to_pages, load_csv_table
from engimind.storage.csv_query import query_csv_tables
from engimind.storage.csv_store import load_csv_tables, save_csv_table
from engimind.ingestion.pdf_loader import load_pdf_pages
from engimind.observability import AgentRun, agent_run, llm_call, tool_call
from engimind.retrieval.store import (
    RetrievedChunk,
    chroma_client,
    collection_name,
    query_collection,
    upsert_chunks,
)

SYSTEM_PROMPT = """You are a careful assistant answering from a single indexed document.

Rules:
- Use ONLY the numbered context snippets below.
- Cite PDF page numbers as [p. N] or CSV row numbers as [row N] immediately after the claim they support.
- CSV context labels each original row explicitly; cite the exact CSV row number that supports the claim, not the chunk's row range.
- If the snippets do not contain the answer, say you cannot find it in the document.
- Do not use outside knowledge. Do not invent page or row numbers or quotes.
"""


@dataclass(frozen=True)
class IngestResult:
    collection: str
    file_hash: str
    pages: int
    chunks: int


@dataclass(frozen=True)
class RagResult:
    answer: str
    chunks: list[RetrievedChunk]
    prompt: str
    refused: bool
    timings_ms: dict[str, float] | None = None


def file_hash(data: bytes) -> str:
    return sha256(data).hexdigest()


def document_collection_name(filename: str, digest: str) -> str:
    identity = sha256(f"{filename.casefold()}\0{digest}".encode("utf-8")).hexdigest()
    return collection_name(identity)


def ingest_pdf(
    data: bytes,
    filename: str,
    settings: Settings,
    openai_client: OpenAI,
    progress_callback: Callable[[str, float], None] | None = None,
) -> IngestResult:
    return ingest_document(
        data, filename, settings, openai_client, progress_callback=progress_callback
    )


def ingest_document(
    data: bytes,
    filename: str,
    settings: Settings,
    openai_client: OpenAI,
    progress_callback: Callable[[str, float], None] | None = None,
) -> IngestResult:
    with agent_run(
        "indexer",
        filename=filename,
        bytes=len(data),
        embedding_provider=getattr(settings, "embedding_provider", None),
        embedding_model=getattr(settings, "embedding_model", None),
        chunk_size=getattr(settings, "chunk_size", None),
        csv_rows_per_chunk=getattr(settings, "csv_rows_per_chunk", None),
    ) as run:
        result = _ingest_document(
            data, filename, settings, openai_client, progress_callback
        )
        run.set_output(
            collection=result.collection, pages=result.pages, chunks=result.chunks
        )
    return result


def _ingest_document(
    data: bytes,
    filename: str,
    settings: Settings,
    openai_client: OpenAI,
    progress_callback: Callable[[str, float], None] | None = None,
) -> IngestResult:
    def report(stage: str, fraction: float) -> None:
        if progress_callback is not None:
            progress_callback(stage, fraction)

    report("Parsing document", 0.05)
    digest = file_hash(data)
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if suffix == "pdf":
        pages = load_pdf_pages(data)
        csv_table = None
    elif suffix == "csv":
        csv_table = load_csv_table(data, source=filename)
        pages = csv_table_to_pages(
            csv_table,
            rows_per_page=settings.csv_rows_per_chunk,
            max_chars=settings.chunk_size,
        )
    else:
        raise ValueError("Only PDF and CSV files are supported.")
    report(f"Parsed {len(pages):,} pages/rows; splitting into chunks", 0.15)
    chunks = chunk_pages(
        pages,
        source=filename,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    report(f"Prepared {len(chunks):,} chunks; creating embeddings and indexing", 0.2)
    name = document_collection_name(filename, digest)
    client = chroma_client(settings)
    completed_chunks = 0

    def report_indexing(completed: int, total: int) -> None:
        nonlocal completed_chunks
        completed_chunks = completed
        fraction = 0.2 + (0.7 * completed / total)
        report(f"Embedding and indexing chunks: {completed:,}/{total:,}", fraction)

    def report_retry(message: str) -> None:
        fraction = 0.2 + 0.7 * completed_chunks / len(chunks)
        report(message, fraction)

    count = upsert_chunks(
        client,
        openai_client,
        settings,
        name=name,
        chunks=chunks,
        progress_callback=report_indexing,
        status_callback=report_retry,
    )
    if csv_table is not None:
        report("Saving structured CSV rows", 0.92)
        save_csv_table(name, csv_table, settings.chroma_path)
    report(f"Index complete: {count:,} chunks", 1.0)
    return IngestResult(
        collection=name,
        file_hash=digest,
        pages=len(pages),
        chunks=count,
    )


def answer_question(
    question: str,
    collection: str | list[str],
    settings: Settings,
    openai_client: OpenAI,
    chat_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    collections = [collection] if isinstance(collection, str) else collection
    with agent_run(
        "basic_rag",
        input_text=question,
        collections=len(collections),
        top_k=getattr(settings, "top_k", None),
        embedding_provider=getattr(settings, "embedding_provider", None),
        chat_model=getattr(settings, "chat_model", None),
    ) as run:
        result = _answer_question(
            question, collection, settings, openai_client, chat_history
        )
        record_answer(run, result)
    return result


def answer_telemetry(result: RagResult) -> dict:
    """Quality and latency signals worth logging for any answer."""
    distances = [chunk.distance for chunk in result.chunks]
    return {
        "refused": result.refused,
        "chunks": len(result.chunks),
        "best_distance": round(min(distances), 4) if distances else None,
        "retrieval_ms": round((result.timings_ms or {}).get("retrieval", 0.0), 2),
        "generation_ms": round((result.timings_ms or {}).get("generation", 0.0), 2),
    }


def record_answer(run: AgentRun, result: RagResult) -> None:
    """Attach output summary and quality telemetry to an agent run."""
    try:
        run.set_output(result.answer, **answer_telemetry(result))
        if result.refused:
            run.set_status("refused")
    except Exception:
        pass


def _answer_question(
    question: str,
    collection: str | list[str],
    settings: Settings,
    openai_client: OpenAI,
    chat_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    collection_names = [collection] if isinstance(collection, str) else collection
    csv_result = answer_csv_question(
        question,
        load_csv_tables(collection_names, settings.chroma_path),
        settings=settings,
        openai_client=openai_client,
    )
    if csv_result is not None:
        return csv_result

    db = chroma_client(settings)
    chunks = []
    retrieval_started = perf_counter()
    for name in collection_names:
        chunks.extend(
            query_collection(
                db,
                openai_client,
                settings,
                name=name,
                question=question,
                k=settings.top_k,
            )
        )
    chunks = sorted(chunks, key=lambda chunk: chunk.distance)[: settings.top_k]
    retrieval_ms = (perf_counter() - retrieval_started) * 1000
    prompt = _user_prompt(question, chunks, chat_history)
    refused = _should_refuse(chunks, settings.max_distance)
    if refused:
        answer = (
            "I cannot find this in the indexed document with enough confidence. "
            "Try rephrasing, or increase k / inspect the retrieved chunks in Debug."
        )
        return RagResult(
            answer=answer,
            chunks=chunks,
            prompt=prompt,
            refused=True,
            timings_ms={"retrieval": retrieval_ms, "generation": 0.0},
        )

    generation_started = perf_counter()
    with llm_call(settings.chat_model, purpose="answer") as call:
        response = openai_client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        call.record(response)
    answer = (response.choices[0].message.content or "").strip()
    return RagResult(
        answer=answer,
        chunks=chunks,
        prompt=prompt,
        refused=False,
        timings_ms={
            "retrieval": retrieval_ms,
            "generation": (perf_counter() - generation_started) * 1000,
        },
    )


def answer_csv_question(
    question: str,
    tables: list[CsvTable],
    *,
    settings: Settings,
    openai_client: OpenAI,
) -> RagResult | None:
    retrieval_started = perf_counter()
    if tables:
        with tool_call("csv_query", tables=len(tables)) as tool_outcome:
            result = query_csv_tables(question, tables)
            tool_outcome["matched"] = result is not None
    else:
        result = query_csv_tables(question, tables)
    if result is None:
        return None
    retrieval_ms = (perf_counter() - retrieval_started) * 1000

    with agent_run(
        "csv_query_agent",
        input_text=question,
        rows=len(getattr(result, "source_rows", ())),
        chat_model=getattr(settings, "chat_model", None),
    ) as run:
        answer = _explain_csv_result(
            question, result, retrieval_ms, settings, openai_client
        )
        record_answer(run, answer)
    return answer


def _explain_csv_result(
    question: str,
    result: Any,
    retrieval_ms: float,
    settings: Settings,
    openai_client: OpenAI,
) -> RagResult:
    chunks = [
        RetrievedChunk(
            text="\n".join(
                f"{column}: {value}" for column, value in row.values.items()
            ),
            page=row.row_number,
            source=table.source,
            distance=0.0,
            section="",
            end_page=row.row_number,
        )
        for table, row in result.source_rows
    ]
    prompt = (
        f"Question: {question}\n\n"
        "Exact result computed from every row in the loaded CSV data:\n"
        f"{result.answer}\n\n"
        "Respond naturally and concisely using only this exact result. Preserve "
        "the count and dates exactly. Do not recalculate, infer, or add facts. "
        "For count questions, do not list matching rows unless the question asks "
        "for the rows."
    )
    generation_started = perf_counter()
    with llm_call(settings.chat_model, purpose="csv_explain") as call:
        response = openai_client.chat.completions.create(
            model=settings.chat_model,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You explain exact CSV query results. The supplied result was "
                        "computed deterministically from the full dataset and is the "
                        "source of truth. Keep count answers concise. Do not list every "
                        "matching row unless the user asks for row details. Do not "
                        "alter any numbers or citations."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )
        call.record(response)
    answer = (response.choices[0].message.content or "").strip()
    if not answer:
        raise RuntimeError("The chat model returned an empty CSV query answer.")
    return RagResult(
        answer=answer,
        chunks=chunks,
        prompt=prompt,
        refused=not bool(chunks),
        timings_ms={
            "retrieval": retrieval_ms,
            "generation": (perf_counter() - generation_started) * 1000,
        },
    )


def _should_refuse(chunks: list[RetrievedChunk], max_distance: float) -> bool:
    if not chunks:
        return True
    return min(c.distance for c in chunks) > max_distance


def format_history_context(
    chat_history: list[tuple[str, str]] | None,
) -> str:
    if not chat_history:
        return "No prior conversation."

    lines: list[str] = []
    for idx, (past_question, past_answer) in enumerate(chat_history, start=1):
        lines.append(f"Previous question {idx}: {past_question}")
        lines.append(f"Previous answer {idx}: {past_answer}")
    return "\n".join(lines)


def _user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    chat_history: list[tuple[str, str]] | None = None,
) -> str:
    history = format_history_context(chat_history)
    if not chunks:
        return (
            f"Question: {question}\n\nConversation history:\n{history}\n\n"
            "Context: (none retrieved)"
        )
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        section = chunk.section or "general"
        location = _chunk_location(chunk)
        lines.append(
            f"[{i}] source={chunk.source} {location} "
            f"section={section} cosine_distance={chunk.distance:.4f}\n{chunk.text}"
        )
    context = "\n\n".join(lines)
    return (
        f"Question: {question}\n\nConversation history:\n{history}\n\n"
        f"Context snippets:\n{context}"
    )


def _chunk_location(chunk: RetrievedChunk) -> str:
    end_page = getattr(chunk, "end_page", None) or chunk.page
    if chunk.source.lower().endswith(".csv"):
        label = "row" if chunk.page == end_page else "rows"
        return f"{label}={chunk.page}" if chunk.page == end_page else f"{label}={chunk.page}-{end_page}"
    label = "page" if chunk.page == end_page else "pages"
    return f"{label}={chunk.page}" if chunk.page == end_page else f"{label}={chunk.page}-{end_page}"


def agentic_route_sketch(question: str, has_index: bool) -> str:
    """Interview aid only — not used as a real agent loop.

    Production agentic RAG typically lets the model choose tools:
    - answer from memory (bad for private PDFs; we never do this)
    - rewrite the query (HyDE / multi-query) then retrieve
    - retrieve, then decide to retrieve again with a narrower query
    - refuse / ask a clarifying question

    This function is a static illustration of that decision table.
    """
    q = question.strip().lower()
    if not has_index:
        return "clarify"
    if any(w in q for w in ("summarize", "overview", "themes")):
        return "retrieve_then_synthesize"
    if len(q.split()) <= 2:
        return "rewrite_then_retrieve"
    return "retrieve_once"
