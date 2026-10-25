"""LangChain callback that reports LLM usage to the active agent run.

Attached to the model at construction time, so chain invocation is untouched.
"""

from __future__ import annotations

from time import perf_counter
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler

from engimind.observability.agents import record_llm_call, usage_from_response


class UsageCallbackHandler(BaseCallbackHandler):
    raise_error = False

    def __init__(self, model: str, purpose: str = "answer") -> None:
        self.model = model
        self.purpose = purpose
        self._started: dict[UUID, float] = {}

    def on_chat_model_start(
        self, serialized: Any, messages: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._started[run_id] = perf_counter()

    def on_llm_start(
        self, serialized: Any, prompts: Any, *, run_id: UUID, **kwargs: Any
    ) -> None:
        self._started[run_id] = perf_counter()

    def on_llm_end(self, response: Any, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            started = self._started.pop(run_id, perf_counter())
            output = getattr(response, "llm_output", None) or {}
            usage = usage_from_response({"usage": output.get("token_usage")})
            model = output.get("model_name") or self.model
            if usage["total"] is None:
                for generations in getattr(response, "generations", []):
                    for generation in generations:
                        metadata = getattr(
                            getattr(generation, "message", None), "usage_metadata", None
                        )
                        if metadata:
                            usage = usage_from_response({"usage": dict(metadata)})
            record_llm_call(
                model=str(model),
                purpose=self.purpose,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                usage=usage,
            )
        except Exception:
            pass

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        try:
            started = self._started.pop(run_id, perf_counter())
            record_llm_call(
                model=self.model,
                purpose=self.purpose,
                duration_ms=round((perf_counter() - started) * 1000, 2),
                error=error,
            )
        except Exception:
            pass
