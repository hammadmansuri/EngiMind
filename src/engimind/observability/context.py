from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from threading import Lock
from typing import Iterator
from uuid import uuid4

_request_id: ContextVar[str] = ContextVar("engimind_request_id", default="-")
_trace_id: ContextVar[str] = ContextVar("engimind_trace_id", default="-")


@dataclass
class RunRecorder:
    """Counters for one agent run. Mutated in place so copies of the context
    (threads, LangGraph executors) all update the same run."""

    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_calls: int = 0
    embedding_tokens: int = 0
    tool_calls: int = 0
    tool_errors: int = 0
    retries: int = 0
    handoffs: int = 0
    models: set[str] = field(default_factory=set)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def add(self, **increments: int) -> None:
        with self._lock:
            for name, amount in increments.items():
                setattr(self, name, getattr(self, name) + amount)

    def add_model(self, model: str) -> None:
        with self._lock:
            self.models.add(model)


@dataclass(frozen=True)
class RunFrame:
    run_id: str
    agent: str
    recorder: RunRecorder


_runs: ContextVar[tuple[RunFrame, ...]] = ContextVar("engimind_runs", default=())


def current_request_id() -> str:
    return _request_id.get()


def current_trace_id() -> str:
    return _trace_id.get()


def current_run() -> RunFrame | None:
    stack = _runs.get()
    return stack[-1] if stack else None


@contextmanager
def push_run(frame: RunFrame) -> Iterator[None]:
    token = _runs.set(_runs.get() + (frame,))
    try:
        yield
    finally:
        _runs.reset(token)


@contextmanager
def request_context(
    request_id: str | None = None, trace_id: str | None = None
) -> Iterator[str]:
    """Tag every log line in this block with correlation and trace ids.

    `request_id` is the correlation id (a caller may supply one). `trace_id`
    follows one workflow end to end. Both are reused when already active, so
    nested work shares them unless an id is passed explicitly.
    """
    active_request = _request_id.get()
    active_trace = _trace_id.get()
    chosen_request = request_id or (
        active_request if active_request != "-" else uuid4().hex[:12]
    )
    chosen_trace = trace_id or (active_trace if active_trace != "-" else uuid4().hex)
    request_token = _request_id.set(chosen_request)
    trace_token = _trace_id.set(chosen_trace)
    try:
        yield chosen_request
    finally:
        _trace_id.reset(trace_token)
        _request_id.reset(request_token)
