from __future__ import annotations

import functools
from time import perf_counter
from typing import TypedDict

from openai import OpenAI

from engimind.core.config import Settings
from engimind.application.rag import (
    RagResult,
    SYSTEM_PROMPT,
    _should_refuse,
    _user_prompt,
    record_answer,
)
from engimind.observability import agent_run, llm_call, record_handoff
from engimind.retrieval.store import (
    RetrievedChunk,
    chroma_client,
    query_collection,
    query_rewrite,
)

try:  # pragma: no cover - optional dependency path
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = None
    StateGraph = None


class AgentState(TypedDict):
    original_question: str
    question: str
    query: str
    collection: str | list[str]
    settings: Settings
    client: OpenAI
    chat_history: list[tuple[str, str]] | None
    chunks: list[RetrievedChunk]
    answer: str
    retrieval_rewritten: bool
    query_changed: bool
    needs_clarification: bool
    stage_timings_ms: dict[str, float]


def _is_empty_question(question: str) -> bool:
    return not (question or "").strip()


def _clarification_message(question: str = "") -> str:
    if not question.strip():
        return "Please enter a question about the indexed document."
    return (
        f"I couldn't find strong evidence in the document to answer: {question.strip()} "
        "Could you rephrase it or specify a relevant term, section, or topic?"
    )


def _retrieve_candidates(
    db,
    openai_client: OpenAI,
    settings,
    collection: str | list[str],
    question: str,
) -> list[RetrievedChunk]:
    collection_names = [collection] if isinstance(collection, str) else collection
    candidates = []
    for name in collection_names:
        candidates.extend(
            query_collection(
                db,
                openai_client,
                settings,
                name=name,
                question=question,
                k=settings.top_k,
                expand_query=False,
            )
        )
    return sorted(candidates, key=lambda chunk: chunk.distance)[: settings.top_k]


def _generate_answer(
    question: str,
    chunks: list[RetrievedChunk],
    settings,
    openai_client: OpenAI,
    chat_history: list[tuple[str, str]] | None = None,
) -> str:
    prompt = _user_prompt(question, chunks, chat_history)
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
    return (response.choices[0].message.content or "").strip()


_HANDOFF_REASONS = {
    ("retrieve", "answer"): "retrieved evidence is strong enough to answer",
    ("retrieve", "rewrite"): "evidence too weak; try a rewritten query",
    ("retrieve", "clarify"): "evidence still weak after rewriting",
    ("rewrite", "retrieve"): "query was rewritten; retrieve again",
    ("rewrite", "clarify"): "rewrite changed nothing; ask the user",
}


def _observed_node(name: str, function):
    """Run a graph node as its own agent run; state handling is untouched."""

    @functools.wraps(function)
    def wrapper(state):
        with agent_run(f"agentic_rag.{name}") as run:
            result = function(state)
            if name == "clarify":
                run.set_status("refused")
            return result

    return wrapper


def _observed_router(source: str, function):
    """Log the handoff each routing decision makes, then return it unchanged."""

    @functools.wraps(function)
    def wrapper(state):
        target = function(state)
        record_handoff(
            f"agentic_rag.{target}",
            _HANDOFF_REASONS.get((source, target), ""),
            from_agent=f"agentic_rag.{source}",
        )
        return target

    return wrapper


def _build_graph():
    if StateGraph is None or END is None:
        return None

    def retrieve_state(state: AgentState) -> AgentState:
        started = perf_counter()
        db = chroma_client(state["settings"])
        candidates = _retrieve_candidates(
            db,
            state["client"],
            state["settings"],
            state["collection"],
            state["question"],
        )
        state["chunks"] = candidates
        state["query"] = state["question"]
        timings = state["stage_timings_ms"]
        timings["retrieval"] = timings.get("retrieval", 0.0) + (
            (perf_counter() - started) * 1000
        )
        return state

    def decide_after_retrieval(state: AgentState) -> str:
        chunks = state.get("chunks", [])
        if chunks and not _should_refuse(chunks, state["settings"].max_distance):
            return "answer"
        if state.get("retrieval_rewritten"):
            return "clarify"
        return "rewrite"

    def rewrite_state(state: AgentState) -> AgentState:
        rewritten = query_rewrite(state["question"])
        state["retrieval_rewritten"] = True
        state["query_changed"] = bool(rewritten and rewritten != state["question"])
        if state["query_changed"]:
            state["question"] = rewritten
        return state

    def decide_after_rewrite(state: AgentState) -> str:
        return "retrieve" if state.get("query_changed") else "clarify"

    def clarify_state(state: AgentState) -> AgentState:
        state["answer"] = _clarification_message(state.get("original_question", ""))
        state["needs_clarification"] = True
        return state

    def answer_state(state: AgentState) -> AgentState:
        started = perf_counter()
        state["answer"] = _generate_answer(
            state["original_question"],
            state["chunks"],
            state["settings"],
            state["client"],
            state.get("chat_history"),
        )
        timings = state["stage_timings_ms"]
        timings["generation"] = timings.get("generation", 0.0) + (
            (perf_counter() - started) * 1000
        )
        return state

    graph = StateGraph(AgentState)
    graph.add_node("retrieve", _observed_node("retrieve", retrieve_state))
    graph.add_node("rewrite", _observed_node("rewrite", rewrite_state))
    graph.add_node("clarify", _observed_node("clarify", clarify_state))
    graph.add_node("answer", _observed_node("answer", answer_state))
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        _observed_router("retrieve", decide_after_retrieval),
        {
            "clarify": "clarify",
            "rewrite": "rewrite",
            "answer": "answer",
        },
    )
    graph.add_conditional_edges(
        "rewrite",
        _observed_router("rewrite", decide_after_rewrite),
        {"retrieve": "retrieve", "clarify": "clarify"},
    )
    graph.add_edge("clarify", END)
    graph.add_edge("answer", END)
    return graph.compile()


def answer_question_agentic(
    question: str,
    collection: str | list[str],
    settings,
    openai_client: OpenAI,
    chat_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    """Explicit agentic-RAG loop with routing, refinement, and clarification.

    The graph aims to behave like an orchestration layer: route -> retrieve ->
    rewrite if needed -> answer when evidence is strong; otherwise ask for
    clarification rather than hallucinating from weak context.
    """
    collections = [collection] if isinstance(collection, str) else collection
    with agent_run(
        "agentic_rag",
        input_text=question,
        collections=len(collections),
        top_k=getattr(settings, "top_k", None),
        chat_model=getattr(settings, "chat_model", None),
    ) as run:
        result = _answer_question_agentic(
            question, collection, settings, openai_client, chat_history
        )
        record_answer(run, result)
    return result


def _answer_question_agentic(
    question: str,
    collection: str | list[str],
    settings,
    openai_client: OpenAI,
    chat_history: list[tuple[str, str]] | None = None,
) -> RagResult:
    if _is_empty_question(question):
        return RagResult(
            answer=_clarification_message(question),
            chunks=[],
            prompt=_user_prompt(question, [], chat_history),
            refused=True,
        )

    graph = _build_graph()
    if graph is not None:
        state: AgentState = {
            "original_question": question,
            "question": question,
            "query": question,
            "collection": collection,
            "settings": settings,
            "client": openai_client,
            "chat_history": chat_history,
            "chunks": [],
            "answer": "",
            "retrieval_rewritten": False,
            "query_changed": False,
            "needs_clarification": False,
            "stage_timings_ms": {},
        }
        final_state = graph.invoke(state)
        chunks = final_state.get("chunks", [])
        answer = final_state.get("answer", "")
        if final_state.get("needs_clarification"):
            return RagResult(
                answer=answer,
                chunks=chunks,
                prompt=_user_prompt(question, chunks, chat_history),
                refused=True,
                timings_ms=final_state.get("stage_timings_ms", {}),
            )
        refused = _should_refuse(chunks, settings.max_distance)
        prompt = _user_prompt(question, chunks, chat_history)
        return RagResult(
            answer=answer,
            chunks=chunks,
            prompt=prompt,
            refused=refused,
            timings_ms=final_state.get("stage_timings_ms", {}),
        )

    db = chroma_client(settings)
    retrieval_started = perf_counter()
    chunks = _retrieve_candidates(db, openai_client, settings, collection, question)
    retrieval_ms = (perf_counter() - retrieval_started) * 1000
    if _should_refuse(chunks, settings.max_distance):
        rewritten_question = query_rewrite(question)
        if rewritten_question and rewritten_question != question:
            retrieval_started = perf_counter()
            chunks = _retrieve_candidates(
                db, openai_client, settings, collection, rewritten_question
            )
            retrieval_ms += (perf_counter() - retrieval_started) * 1000
    prompt = _user_prompt(question, chunks, chat_history)
    if _should_refuse(chunks, settings.max_distance):
        return RagResult(
            answer=_clarification_message(question),
            chunks=chunks,
            prompt=prompt,
            refused=True,
            timings_ms={"retrieval": retrieval_ms, "generation": 0.0},
        )

    generation_started = perf_counter()
    answer = _generate_answer(question, chunks, settings, openai_client, chat_history)
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
