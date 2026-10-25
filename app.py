import os

import streamlit as st

from engimind.application.rag import agentic_route_sketch
from engimind.application.services.rag_service import (
    answer_question_for_mode,
    ingest_document_file,
    load_app_settings,
    run_evaluation_case,
)
from engimind.evaluation.metrics import EvaluationCase
from engimind.observability import configure_logging
from engimind.ui.session import build_history_pairs, init_session_state

configure_logging()


st.set_page_config(page_title="EngiMind | Document Intelligence", layout="wide")
st.title("EngiMind")
st.caption(
    "Grounded question answering over PDF and CSV documents, with source citations and configurable retrieval."
)

with st.sidebar:
    st.subheader("Index settings")
    chunk_size = st.slider("Chunk size (characters)", 200, 2000, 800, 50)
    chunk_overlap = st.slider("Chunk overlap", 0, 400, 150, 10)
    csv_rows_per_chunk = st.slider(
        "CSV rows per embedding chunk",
        1,
        100,
        1,
        help=(
            "Group this many consecutive CSV data rows into each embedding chunk. "
            "Every row keeps its original row number. Re-index CSVs after changing."
        ),
    )
    top_k = st.slider("k (neighbors to retrieve)", 1, 12, 4)
    max_distance = st.slider(
        "Max cosine distance (refuse if best chunk is worse)",
        0.10,
        1.00,
        0.55,
        0.05,
        help="Chroma cosine distance ≈ 1 − cosine similarity. Lower = closer match.",
    )
    rag_implementation = st.radio(
        "RAG implementation",
        ["Custom", "LangChain", "Agentic RAG"],
        help="Custom and LangChain are direct retrieval flows; Agentic RAG adds a rewrite-and-retry loop before answering.",
    )
    embedding_provider = st.selectbox(
        "Embedding provider",
        ["openai", "ollama"],
        index=0 if os.getenv("EMBEDDING_PROVIDER", "openai").lower() == "openai" else 1,
        help="Switching provider/model requires re-indexing documents.",
    )
    if embedding_provider == "ollama":
        st.caption(
            "Local embeddings use Ollama model "
            f"`{os.getenv('OLLAMA_EMBEDDING_MODEL', 'all-minilm')}`. "
            "Run `ollama pull all-minilm` and keep Ollama running. "
            "OpenAI is still used for answer generation and Eval judging."
        )
    st.markdown(
        "Copy `.env.example` to `.env` and set `OPENAI_API_KEY` for answers. "
        "Re-index after changing chunk size, overlap, or CSV rows per chunk."
    )

try:
    settings = load_app_settings(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        csv_rows_per_chunk=csv_rows_per_chunk,
        embedding_provider=embedding_provider,
        top_k=top_k,
        max_distance=max_distance,
    )
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

init_session_state()
if "indexed_documents" not in st.session_state:
    st.session_state["indexed_documents"] = []
if st.session_state.get("index_feedback"):
    feedback = st.session_state["index_feedback"]
    if feedback["failures"]:
        st.error("Some files could not be indexed:\n\n" + "\n".join(feedback["failures"]))
    if feedback["completed"]:
        st.success(
            "Indexing complete: "
            + "; ".join(
                f"{item['filename']} ({item['chunks']:,} chunks)"
                for item in feedback["completed"]
            )
        )

uploaded = st.file_uploader(
    "PDF or CSV", type=["pdf", "csv"], accept_multiple_files=True
)
left_col, right_col = st.columns(2)
with left_col:
    index_clicked = st.button(
        "Index documents", type="primary", disabled=not uploaded
    )
with right_col:
    if st.session_state.indexed_documents:
        st.success(
            f"{len(st.session_state.indexed_documents)} document(s) available for questions."
        )
        st.caption(
            "Indexed files: "
            + ", ".join(
                document["filename"]
                for document in st.session_state.indexed_documents
            )
        )

for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Indexing step: parse document -> chunk -> embed -> persist to Chroma.
if index_clicked and uploaded:
    if chunk_overlap >= chunk_size:
        st.error("Overlap must be smaller than chunk size.")
        st.stop()

    failures = []
    completed_files = []
    progress_bar = st.progress(0.0, text="Starting indexing… 0.0%")
    status = st.status("Preparing to index documents…", expanded=True)
    st.session_state.pop("index_feedback", None)
    for file_index, file in enumerate(uploaded):
        status.update(
            label=f"Indexing {file.name} ({file.size / (1024 * 1024):.1f} MB)",
            state="running",
            expanded=True,
        )

        def show_progress(stage: str, fraction: float, *, index: int = file_index) -> None:
            overall_fraction = (index + fraction) / len(uploaded)
            percentage = overall_fraction * 100
            progress_text = f"{file.name}: {stage} · {percentage:.1f}% overall"
            status.update(
                label=progress_text,
                state="running",
                expanded=True,
            )
            progress_bar.progress(overall_fraction, text=progress_text)

        try:
            file_bytes = file.getvalue()
            result = ingest_document_file(
                file_bytes,
                file.name,
                settings,
                progress_callback=show_progress,
            )
            extension = file.name.rsplit(".", 1)[-1].lower()
            document = {
                "filename": file.name,
                "collection": result.collection,
                "file_type": extension,
                "ingest": result,
            }
            completed_files.append(
                {"filename": file.name, "chunks": result.chunks}
            )
            st.session_state.indexed_documents = [
                existing
                for existing in st.session_state.indexed_documents
                if existing["filename"] != file.name
            ]
            st.session_state.indexed_documents.append(document)
        except Exception as exc:
            failures.append(f"{file.name}: {exc}")
    st.session_state["index_feedback"] = {
        "completed": completed_files,
        "failures": failures,
    }
    if failures:
        status.update(label="Indexing finished with errors", state="error")
    else:
        status.update(label="Indexing complete", state="complete")
    if st.session_state.indexed_documents:
        st.session_state.collection = [
            document["collection"]
            for document in st.session_state.indexed_documents
        ]
        st.session_state.chat_history = []
    st.rerun()

st.divider()
question = st.text_input("Question about the indexed document")
ask = st.button(
    "Ask",
    disabled=not (st.session_state.indexed_documents and question.strip()),
)

# Query flow: same indexed collection, different runtime clients.
if ask and question.strip() and st.session_state.indexed_documents:
    history = build_history_pairs()
    route = agentic_route_sketch(question, has_index=True)
    collections = [
        document["collection"]
        for document in st.session_state.indexed_documents
    ]
    with st.spinner("Retrieve → generate"):
        try:
            result = answer_question_for_mode(
                question.strip(),
                collections,
                settings,
                rag_implementation,
                chat_history=history,
            )
        except Exception as exc:
            st.exception(exc)
            st.stop()

    st.session_state.chat_history.append({"role": "user", "content": question.strip()})
    st.session_state.chat_history.append({"role": "assistant", "content": result.answer})

    st.subheader("Answer")
    if result.refused:
        st.warning(result.answer)
    else:
        st.write(result.answer)
    st.caption(f"Agentic route sketch (not executed as a loop): `{route}`")

    with st.expander("Debug: retrieved chunks, scores, prompt", expanded=True):
        st.markdown(
            "If the answer is wrong, look here first. Bad chunks → retrieval problem. "
            "Good chunks, wrong answer → generation / prompt problem."
        )
        debug_sections = []
        for index, chunk in enumerate(result.chunks, start=1):
            section = chunk.section or "general"
            end_page = chunk.end_page if chunk.end_page is not None else chunk.page
            if chunk.source.lower().endswith(".csv"):
                location = (
                    f"row {chunk.page}"
                    if chunk.page == end_page
                    else f"rows {chunk.page}-{end_page}"
                )
            else:
                location = (
                    f"page {chunk.page}"
                    if chunk.page == end_page
                    else f"pages {chunk.page}-{end_page}"
                )
            debug_sections.append(
                f"[{index}] source={chunk.source} · {location} · "
                f"section={section} · cosine distance={chunk.distance:.4f}\n"
                f"{chunk.text}"
            )
        debug_report = (
            "\n\n".join(debug_sections)
            + "\n\nPrompt sent to the chat model:\n"
            + result.prompt
        )
        st.markdown("**Copy all retrieved chunks, scores, and prompt**")
        st.code(debug_report, language=None)

st.divider()
st.header("RAG evaluation")
st.caption(
    "Run a small labeled test set against the selected RAG implementation. "
    "Each expected evidence location is identified by its indexed filename and page/row."
)
st.markdown(
    "Add one row per test question. `relevant_locations` is a comma-separated list "
    "such as `policy.pdf:2` or `policy.pdf:2, policy.pdf:5`. "
    "Recall@k checks whether those gold locations appear in the retrieved top-k. "
    "Relevance and groundedness are scored by an LLM judge against your reference answer "
    "and retrieved evidence."
)
if st.session_state.indexed_documents:
    st.caption(
        "Indexed sources: "
        + ", ".join(document["filename"] for document in st.session_state.indexed_documents)
    )

if "evaluation_input" not in st.session_state:
    st.session_state["evaluation_input"] = [
        {"question": "", "reference_answer": "", "relevant_locations": ""}
    ]
evaluation_input = st.data_editor(
    st.session_state["evaluation_input"],
    num_rows="dynamic",
    hide_index=True,
    use_container_width=True,
    column_config={
        "question": st.column_config.TextColumn("Question", required=True),
        "reference_answer": st.column_config.TextColumn(
            "Reference answer", required=True
        ),
        "relevant_locations": st.column_config.TextColumn(
            "Gold locations (filename:page/row)", required=True
        ),
    },
    key="evaluation_cases_editor",
)
st.session_state["evaluation_input"] = evaluation_input
run_eval = st.button(
    "Run evaluation",
    type="primary",
    disabled=not st.session_state.indexed_documents,
)
if run_eval:
    indexed_sources = {
        document["filename"].casefold()
        for document in st.session_state.indexed_documents
    }
    cases = []
    for row_number, row in enumerate(evaluation_input, start=1):
        question_text = str(row.get("question") or "").strip()
        reference_answer = str(row.get("reference_answer") or "").strip()
        raw_labels = str(row.get("relevant_locations") or "").strip()
        if not question_text and not reference_answer and not raw_labels:
            continue
        if not question_text or not reference_answer or not raw_labels:
            st.error(
                f"Evaluation row {row_number} needs a question, reference answer, "
                "and at least one gold location."
            )
            st.stop()
        labels = tuple(label.strip() for label in raw_labels.split(",") if label.strip())
        if not labels:
            st.error(f"Evaluation row {row_number} needs at least one gold location.")
            st.stop()
        for label in labels:
            if ":" not in label:
                st.error(
                    f"Evaluation row {row_number}: `{label}` must use "
                    "`filename:page-or-row` format."
                )
                st.stop()
            source, page_text = label.rsplit(":", 1)
            if source.casefold() not in indexed_sources:
                st.error(
                    f"Evaluation row {row_number}: `{source}` is not among the "
                    "currently indexed filenames."
                )
                st.stop()
            if not page_text.isdigit() or int(page_text) < 1:
                st.error(
                    f"Evaluation row {row_number}: `{label}` must end with a "
                    "positive page or row number."
                )
                st.stop()
        cases.append(
            EvaluationCase(
                question=question_text,
                reference_answer=reference_answer,
                relevant_locations=labels,
            )
        )
    if not cases:
        st.error("Add at least one complete evaluation row before running.")
        st.stop()

    collections = [
        document["collection"] for document in st.session_state.indexed_documents
    ]
    outcomes = []
    with st.spinner(f"Evaluating {len(cases)} question(s)…"):
        for case in cases:
            try:
                outcomes.append(
                    run_evaluation_case(case, collections, settings, rag_implementation)
                )
            except Exception as exc:
                st.exception(exc)
                st.stop()
    st.session_state["evaluation_outcomes"] = outcomes

if st.session_state.get("evaluation_outcomes"):
    outcomes = st.session_state["evaluation_outcomes"]
    st.subheader("Evaluation results")
    average_recall = sum(item["recall_at_k"] for item in outcomes) / len(outcomes)
    average_relevance = sum(item["relevance"] for item in outcomes) / len(outcomes)
    average_groundedness = sum(item["groundedness"] for item in outcomes) / len(outcomes)
    metric_columns = st.columns(3)
    metric_columns[0].metric("Mean Recall@k", f"{average_recall:.0%}")
    metric_columns[1].metric("Mean relevance", f"{average_relevance:.2f} / 1")
    metric_columns[2].metric("Mean groundedness", f"{average_groundedness:.2f} / 1")
    st.dataframe(
        [
            {
                "Question": item["question"],
                "Recall@k": item["recall_at_k"],
                "Relevance": item["relevance"],
                "Groundedness": item["groundedness"],
                "Retrieval (ms)": round(item["retrieval_ms"], 1),
                "Generation (ms)": round(item["generation_ms"], 1),
                "Judge (ms)": round(item["evaluation_ms"], 1),
                "Total (ms)": round(item["total_ms"], 1),
            }
            for item in outcomes
        ],
        hide_index=True,
        use_container_width=True,
    )
    for index, item in enumerate(outcomes, start=1):
        with st.expander(f"Case {index}: {item['question']}"):
            st.markdown(f"**Answer:** {item['answer']}")
            st.markdown(
                f"**Relevance ({item['relevance']:.2f}):** "
                f"{item['relevance_reason']}"
            )
            st.markdown(
                f"**Groundedness ({item['groundedness']:.2f}):** "
                f"{item['groundedness_reason']}"
            )
            result = item["result"]
            st.markdown("**Retrieved evidence**")
            for chunk in result.chunks:
                st.markdown(
                    f"- `{chunk.source}:{chunk.page}` · distance "
                    f"{chunk.distance:.3f} — {chunk.text}"
                )
