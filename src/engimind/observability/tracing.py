from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from hashlib import sha256
from time import perf_counter
from typing import Any, Iterator

from engimind.observability.context import request_context
from engimind.observability.logging_setup import LOGGER_NAME


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f"{LOGGER_NAME}.{name}")


def log_event(
    logger: logging.Logger,
    event: str,
    *,
    level: int = logging.INFO,
    exc_info: bool = False,
    **fields: Any,
) -> None:
    """Never lets a logging problem disturb the code being observed."""
    try:
        logger.log(
            level, event, extra={"event": event, "fields": fields}, exc_info=exc_info
        )
    except Exception:
        pass


_PREVIEW_CHARS = 300


def describe_text(label: str, text: str) -> dict[str, Any]:
    """Log the shape of user text, not the text, unless LOG_CONTENT is enabled.

    Questions can contain sensitive data, so by default only the length and a
    short hash (to correlate repeated questions) are written to disk. With
    LOG_CONTENT enabled a truncated preview is added (still redacted on write).
    """
    fields: dict[str, Any] = {
        f"{label}_chars": len(text),
        f"{label}_sha": sha256(text.encode("utf-8")).hexdigest()[:10],
    }
    if os.getenv("LOG_CONTENT", "").strip().lower() in {"1", "true", "yes"}:
        fields[label] = (
            text if len(text) <= _PREVIEW_CHARS else text[:_PREVIEW_CHARS] + "…"
        )
    return fields


def summarize_arguments(
    arguments: dict[str, Any], plain_keys: tuple[str, ...] = ("source",)
) -> dict[str, Any]:
    """Summarize tool arguments; free text is described, identifiers pass through."""
    summary: dict[str, Any] = {}
    for key, value in arguments.items():
        if isinstance(value, str) and key not in plain_keys:
            summary.update(describe_text(key, value))
        else:
            summary[key] = value
    return summary


@contextmanager
def span(logger: logging.Logger, name: str, **fields: Any) -> Iterator[dict[str, Any]]:
    """Time a unit of work and log `<name>.completed` or `<name>.failed`.

    The yielded dict lets the caller attach result fields (counts, scores) that
    are only known once the work has finished.
    """
    outcome: dict[str, Any] = {}
    with request_context():
        started = perf_counter()
        try:
            yield outcome
        except Exception as exc:
            log_event(
                logger,
                f"{name}.failed",
                level=logging.ERROR,
                duration_ms=_elapsed_ms(started),
                error_type=type(exc).__name__,
                error=str(exc),
                **{**fields, **outcome},
            )
            raise
        log_event(
            logger,
            f"{name}.completed",
            duration_ms=_elapsed_ms(started),
            **{**fields, **outcome},
        )


def _elapsed_ms(started: float) -> float:
    return round((perf_counter() - started) * 1000, 2)
