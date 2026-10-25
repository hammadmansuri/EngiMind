from __future__ import annotations

import streamlit as st


def init_session_state() -> None:
    """Keep the UI state simple and predictable for the single-app demo."""
    defaults = {
        "collection": None,
        "ingest": None,
        "chat_history": [],
        "indexed_documents": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def build_history_pairs() -> list[tuple[str, str]]:
    """Convert the chat transcript into (question, answer) pairs for the prompt."""
    history: list[tuple[str, str]] = []
    current_question: None | str = None

    for item in st.session_state.chat_history:
        if item["role"] == "user":
            current_question = item["content"]
        elif item["role"] == "assistant" and current_question is not None:
            history.append((current_question, item["content"]))
            current_question = None

    return history
