"""Agent-level observability: runs, LLM usage, tool calls, retries, handoffs.

Everything here is passive. Telemetry failures are swallowed, wrapped code runs
exactly as before, and exceptions always propagate unchanged.
"""

from __future__ import annotations

import functools
import logging
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Iterator
from uuid import uuid4

from engimind.observability.context import (
    RunFrame,
    RunRecorder,
    current_run,
    push_run,
    request_context,
)
from engimind.observability.tracing import describe_text, get_logger, log_event

logger = get_logger("agents")

STATUS_SUCCESS = "success"
STATUS_REFUSED = "refused"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)


def _emit(event: str, fields: dict[str, Any], level: int = logging.INFO) -> None:
    try:
        log_event(logger, event, level=level, **fields)
    except Exception:
        pass


class AgentRun:
    """Handle yielded by `agent_run` for attaching output and status."""

    def __init__(self, run_id: str, agent: str, parent_run_id: str | None) -> None:
        self.run_id = run_id
        self.agent = agent
        self.parent_run_id = parent_run_id
        self.status = STATUS_SUCCESS
        self.fields: dict[str, Any] = {}

    def set_output(self, text: str | None = None, **fields: Any) -> None:
        try:
            if text is not None:
                self.fields.update(describe_text("output", text))
            self.fields.update(fields)
        except Exception:
            pass

    def set_status(self, status: str) -> None:
        self.status = status

    def add(self, **fields: Any) -> None:
        self.fields.update(fields)


@contextmanager
def agent_run(
    agent: str, *, input_text: str | None = None, **fields: Any
) -> Iterator[AgentRun]:
    """Record one agent execution from start to finish.

    Emits `agent.run.started` and then `agent.run.completed` (status `success`
    or `refused`) or `agent.run.failed`. Nested runs form a tree through
    `parent_run_id` and share the workflow's `trace_id`.
    """
    with request_context():
        parent = current_run()
        run = AgentRun(uuid4().hex[:16], agent, parent.run_id if parent else None)
        recorder = RunRecorder()
        started = perf_counter()
        started_at = _now()
        try:
            base = dict(fields)
            if input_text is not None:
                base.update(describe_text("input", input_text))
        except Exception:
            base = {}
        with push_run(RunFrame(run.run_id, agent, recorder)):
            _emit(
                "agent.run.started",
                {
                    **base,
                    "agent": agent,
                    "run_id": run.run_id,
                    "parent_run_id": run.parent_run_id,
                    "started_at": started_at,
                },
            )
            error: BaseException | None = None
            try:
                yield run
            except BaseException as exc:
                error = exc
                raise
            finally:
                _finish_run(run, recorder, base, started, started_at, error)


def _finish_run(
    run: AgentRun,
    recorder: RunRecorder,
    base: dict[str, Any],
    started: float,
    started_at: str,
    error: BaseException | None,
) -> None:
    try:
        status = run.status
        failure: dict[str, Any] = {}
        level = logging.INFO
        if error is not None:
            status = STATUS_FAILED if isinstance(error, Exception) else STATUS_CANCELLED
            failure = {"error_type": type(error).__name__, "error": str(error)}
            level = logging.ERROR if status == STATUS_FAILED else logging.WARNING
        core = {
            "agent": run.agent,
            "run_id": run.run_id,
            "parent_run_id": run.parent_run_id,
            "status": status,
            "started_at": started_at,
            "duration_ms": _ms(started),
            "llm_calls": recorder.llm_calls,
            "prompt_tokens": recorder.prompt_tokens,
            "completion_tokens": recorder.completion_tokens,
            "total_tokens": recorder.total_tokens,
            "embedding_calls": recorder.embedding_calls,
            "embedding_tokens": recorder.embedding_tokens,
            "tool_calls": recorder.tool_calls,
            "tool_errors": recorder.tool_errors,
            "retries": recorder.retries,
            "handoffs": recorder.handoffs,
            "models": sorted(recorder.models),
        }
        _emit(
            "agent.run.failed" if status == STATUS_FAILED else "agent.run.completed",
            {**base, **run.fields, **core, **failure},
            level,
        )
    except Exception:
        pass


def _int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def usage_from_response(response: Any) -> dict[str, int | None]:
    """Read token usage from an OpenAI-style response (object or mapping)."""
    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, Mapping):
        usage = response.get("usage")

    def read(*names: str) -> int | None:
        for name in names:
            value = (
                usage.get(name) if isinstance(usage, Mapping) else getattr(usage, name, None)
            )
            if _int(value) is not None:
                return value
        return None

    prompt = read("prompt_tokens", "input_tokens")
    completion = read("completion_tokens", "output_tokens")
    total = read("total_tokens")
    if total is None and prompt is not None and completion is not None:
        total = prompt + completion
    return {"prompt": prompt, "completion": completion, "total": total}


def record_llm_call(
    *,
    model: str,
    purpose: str,
    duration_ms: float,
    usage: Mapping[str, int | None] | None = None,
    kind: str = "chat",
    provider: str = "openai",
    error: BaseException | None = None,
) -> None:
    """Account one model call against the active run and log it."""
    try:
        usage = usage or {}
        prompt = usage.get("prompt") or 0
        completion = usage.get("completion") or 0
        total = usage.get("total") or (prompt + completion)
        run = current_run()
        if run is not None:
            run.recorder.add_model(model)
            if kind == "embedding":
                run.recorder.add(embedding_calls=1, embedding_tokens=total)
            else:
                run.recorder.add(
                    llm_calls=1,
                    prompt_tokens=prompt,
                    completion_tokens=completion,
                    total_tokens=total,
                )
        failed = error is not None
        # Embedding batches are numerous during indexing; they are summed into
        # the run totals and only logged individually when debugging.
        level = (
            logging.ERROR
            if failed
            else logging.DEBUG
            if kind == "embedding"
            else logging.INFO
        )
        _emit(
            "llm.call.failed" if failed else "llm.call.completed",
            {
                "model": model,
                "provider": provider,
                "kind": kind,
                "purpose": purpose,
                "duration_ms": duration_ms,
                "prompt_tokens": usage.get("prompt"),
                "completion_tokens": usage.get("completion"),
                "total_tokens": usage.get("total"),
                **(
                    {"error_type": type(error).__name__, "error": str(error)}
                    if failed
                    else {}
                ),
            },
            level,
        )
    except Exception:
        pass


class LlmCall:
    def __init__(self) -> None:
        self.usage: dict[str, int | None] = {}
        self.model: str | None = None

    def record(self, response: Any) -> None:
        """Capture token usage (and the served model) from a response."""
        try:
            self.usage = usage_from_response(response)
            served = getattr(response, "model", None)
            if isinstance(served, str) and served:
                self.model = served
        except Exception:
            pass


@contextmanager
def llm_call(
    model: str, *, purpose: str, provider: str = "openai", kind: str = "chat"
) -> Iterator[LlmCall]:
    """Time and account one chat/embedding call; call `.record(response)` inside."""
    call = LlmCall()
    started = perf_counter()
    try:
        yield call
    except Exception as exc:
        record_llm_call(
            model=model,
            purpose=purpose,
            duration_ms=_ms(started),
            usage=call.usage,
            kind=kind,
            provider=provider,
            error=exc,
        )
        raise
    record_llm_call(
        model=call.model or model,
        purpose=purpose,
        duration_ms=_ms(started),
        usage=call.usage,
        kind=kind,
        provider=provider,
    )


@contextmanager
def tool_call(name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Record a tool invocation (retrieval, CSV query, MCP tool, ...)."""
    outcome: dict[str, Any] = {}
    started = perf_counter()
    try:
        yield outcome
    except Exception as exc:
        _record_tool(name, started, fields, outcome, exc)
        raise
    _record_tool(name, started, fields, outcome, None)


def _record_tool(
    name: str,
    started: float,
    fields: dict[str, Any],
    outcome: dict[str, Any],
    error: BaseException | None,
) -> None:
    try:
        run = current_run()
        if run is not None:
            run.recorder.add(tool_calls=1, tool_errors=1 if error else 0)
        _emit(
            "tool.call.failed" if error else "tool.call.completed",
            {
                **fields,
                **outcome,
                "tool": name,
                "duration_ms": _ms(started),
                **(
                    {"error_type": type(error).__name__, "error": str(error)}
                    if error
                    else {}
                ),
            },
            logging.ERROR if error else logging.INFO,
        )
    except Exception:
        pass


def traced_tool(
    name: str,
    *,
    describe: Callable[..., dict[str, Any]] | None = None,
    summarize: Callable[[Any], dict[str, Any]] | None = None,
    when: Callable[..., bool] | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator form of `tool_call`; leaves arguments and results untouched."""

    def decorator(function: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                skip = when is not None and not when(*args, **kwargs)
                fields = describe(*args, **kwargs) if describe and not skip else {}
            except Exception:
                skip, fields = False, {}
            if skip:
                return function(*args, **kwargs)
            with tool_call(name, **fields) as outcome:
                result = function(*args, **kwargs)
                try:
                    if summarize is not None:
                        outcome.update(summarize(result))
                except Exception:
                    pass
            return result

        return wrapper

    return decorator


def record_retry(kind: str, **fields: Any) -> None:
    try:
        run = current_run()
        if run is not None:
            run.recorder.add(retries=1)
        _emit("retry.attempted", {**fields, "kind": kind}, logging.WARNING)
    except Exception:
        pass


def record_handoff(
    to_agent: str, reason: str = "", *, from_agent: str | None = None
) -> None:
    """Log control passing from one agent (or node) to another."""
    try:
        run = current_run()
        if run is not None:
            run.recorder.add(handoffs=1)
        _emit(
            "agent.handoff",
            {
                "from_agent": from_agent or (run.agent if run else "-"),
                "to_agent": to_agent,
                "reason": reason,
            },
        )
    except Exception:
        pass
