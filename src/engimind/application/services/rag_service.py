from __future__ import annotations

from time import perf_counter
from typing import Any, Callable

from openai import OpenAI

from engimind.application.agentic_rag import answer_question_agentic
from engimind.core.config import Settings, load_settings
from engimind.evaluation.metrics import EvaluationCase, judge_answer, recall_at_k
from engimind.application.langchain_rag import answer_question_langchain
from engimind.application.rag import (
    answer_csv_question,
    answer_question,
    ingest_document,
    ingest_pdf,
    record_answer,
)
from engimind.observability import agent_run, record_handoff
from engimind.storage.csv_store import load_csv_tables


def load_app_settings(**kwargs: Any) -> Settings:
    return load_settings(**kwargs)


def ingest_pdf_document(pdf_bytes: bytes, filename: str, settings: Settings) -> Any:
    client = OpenAI(api_key=settings.openai_api_key)
    return ingest_pdf(pdf_bytes, filename, settings, client)


def ingest_document_file(
    file_bytes: bytes,
    filename: str,
    settings: Settings,
    progress_callback: Callable[[str, float], None] | None = None,
) -> Any:
    client = OpenAI(api_key=settings.openai_api_key)
    return ingest_document(
        file_bytes,
        filename,
        settings,
        client,
        progress_callback=progress_callback,
    )


def answer_question_for_mode(
    question: str,
    collection: str | list[str],
    settings: Settings,
    mode: str,
    chat_history: list[tuple[str, str]] | None = None,
    openai_client: OpenAI | None = None,
):
    with agent_run("router", input_text=question, mode=mode) as run:
        result = _answer_question_for_mode(
            question, collection, settings, mode, chat_history, openai_client
        )
        record_answer(run, result)
    return result


def _answer_question_for_mode(
    question: str,
    collection: str | list[str],
    settings: Settings,
    mode: str,
    chat_history: list[tuple[str, str]] | None = None,
    openai_client: OpenAI | None = None,
):
    collection_names = [collection] if isinstance(collection, str) else collection
    client = openai_client or OpenAI(api_key=settings.openai_api_key)
    csv_result = answer_csv_question(
        question,
        load_csv_tables(collection_names, settings.chroma_path),
        settings=settings,
        openai_client=client,
    )
    if csv_result is not None:
        record_handoff("csv_query_agent", "question answered from CSV tables")
        return csv_result

    if mode == "LangChain":
        record_handoff("langchain_rag", f"mode={mode}")
        return answer_question_langchain(
            question,
            collection,
            settings,
            chat_history=chat_history,
        )
    if mode == "Agentic RAG":
        record_handoff("agentic_rag", f"mode={mode}")
        return answer_question_agentic(
            question,
            collection,
            settings,
            client,
            chat_history=chat_history,
        )
    record_handoff("basic_rag", f"mode={mode}")
    return answer_question(
        question,
        collection,
        settings,
        client,
        chat_history=chat_history,
    )


def run_evaluation_case(
    case: EvaluationCase,
    collection: str | list[str],
    settings: Settings,
    mode: str,
) -> dict[str, Any]:
    with agent_run("eval_runner", input_text=case.question, mode=mode) as run:
        row = _run_evaluation_case(case, collection, settings, mode)
        try:
            run.set_output(
                row["answer"],
                recall_at_k=row["recall_at_k"],
                relevance=row["relevance"],
                groundedness=row["groundedness"],
                retrieval_ms=round(row["retrieval_ms"], 2),
                generation_ms=round(row["generation_ms"], 2),
                evaluation_ms=round(row["evaluation_ms"], 2),
            )
        except Exception:
            pass
    return row


def _run_evaluation_case(
    case: EvaluationCase,
    collection: str | list[str],
    settings: Settings,
    mode: str,
) -> dict[str, Any]:
    client = OpenAI(api_key=settings.openai_api_key)
    started = perf_counter()
    result = answer_question_for_mode(
        case.question,
        collection,
        settings,
        mode,
        openai_client=client,
    )
    evaluation_started = perf_counter()
    scores = judge_answer(
        case.question,
        case.reference_answer,
        result.answer,
        [chunk.text for chunk in result.chunks],
        openai_client=client,
        model=settings.chat_model,
    )
    evaluation_ms = (perf_counter() - evaluation_started) * 1000
    timings = result.timings_ms or {}
    return {
        "question": case.question,
        "answer": result.answer,
        "recall_at_k": recall_at_k(case.relevant_locations, result),
        "relevance": scores.relevance,
        "relevance_reason": scores.relevance_reason,
        "groundedness": scores.groundedness,
        "groundedness_reason": scores.groundedness_reason,
        "retrieval_ms": timings.get("retrieval", 0.0),
        "generation_ms": timings.get("generation", 0.0),
        "evaluation_ms": evaluation_ms,
        "total_ms": (perf_counter() - started) * 1000,
        "result": result,
    }
