from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from engimind.observability.context import (
    current_request_id,
    current_run,
    current_trace_id,
)
from engimind.observability.redaction import redact, redact_value

LOGGER_NAME = "engimind"
LOG_FILE_NAME = "engimind.log"
_HANDLER_MARK = "_engimind_file_handler"


class JsonLineFormatter(logging.Formatter):
    """One JSON object per line, so logs can be grepped, jq'd, or shipped as-is."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, object] = {
            "ts": datetime.fromtimestamp(record.created, timezone.utc).isoformat(
                timespec="milliseconds"
            ),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
            "request_id": current_request_id(),
            "trace_id": current_trace_id(),
        }
        run = current_run()
        if run is not None:
            payload["run_id"] = run.run_id
            payload["agent"] = run.agent
        payload.update(getattr(record, "fields", {}))
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(redact_value(payload), default=str, ensure_ascii=False)


def log_directory() -> Path:
    return Path(os.getenv("LOG_DIR", "logs")).expanduser()


def configure_logging() -> Path | None:
    """Attach the rotating JSON file handler once and return the log file path.

    Safe to call repeatedly (Streamlit re-runs the script on every interaction).
    Logs go to a file only: stdout is the MCP stdio transport, so printing there
    would corrupt the protocol. Returns None if the log folder is not writable;
    the app keeps working without file logs.
    """
    logger = logging.getLogger(LOGGER_NAME)
    for handler in logger.handlers:
        if getattr(handler, _HANDLER_MARK, False):
            return Path(handler.baseFilename)

    level_name = os.getenv("LOG_LEVEL", "INFO").strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    try:
        directory = log_directory()
        directory.mkdir(parents=True, exist_ok=True)
        handler = RotatingFileHandler(
            directory / LOG_FILE_NAME,
            maxBytes=int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024))),
            backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
            encoding="utf-8",
            delay=True,
        )
    except OSError as exc:
        print(f"EngiMind: file logging disabled ({exc})", file=sys.stderr)
        return None

    setattr(handler, _HANDLER_MARK, True)
    handler.setFormatter(JsonLineFormatter())
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return Path(handler.baseFilename)


# Without configure_logging() the library stays silent (tests, imports).
logging.getLogger(LOGGER_NAME).addHandler(logging.NullHandler())
