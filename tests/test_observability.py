import json
import logging
import os
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from engimind.observability import (
    configure_logging,
    current_request_id,
    describe_text,
    get_logger,
    log_event,
    request_context,
    span,
    summarize_arguments,
)
from engimind.observability.logging_setup import LOGGER_NAME


def _reset_logger() -> None:
    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, "_engimind_file_handler", False):
            handler.close()
            logger.removeHandler(handler)


class ObservabilityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = TemporaryDirectory(ignore_cleanup_errors=True)
        self.addCleanup(self.tmp.cleanup)
        self.addCleanup(_reset_logger)
        env = patch.dict(os.environ, {"LOG_DIR": self.tmp.name}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("LOG_CONTENT", None)
        self.log_file = configure_logging()

    def records(self) -> list[dict]:
        for handler in logging.getLogger(LOGGER_NAME).handlers:
            handler.flush()
        lines = Path(self.log_file).read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines]

    def test_writes_json_lines_to_log_folder(self) -> None:
        log_event(get_logger("t"), "thing.happened", count=3)
        record = self.records()[0]
        self.assertEqual(Path(self.log_file).parent, Path(self.tmp.name))
        self.assertEqual(record["event"], "thing.happened")
        self.assertEqual(record["count"], 3)
        self.assertEqual(record["level"], "INFO")
        self.assertEqual(record["logger"], "engimind.t")

    def test_configure_is_idempotent(self) -> None:
        self.assertEqual(configure_logging(), self.log_file)
        log_event(get_logger("t"), "once")
        self.assertEqual(len(self.records()), 1)

    def test_span_logs_duration_and_outcome_with_shared_request_id(self) -> None:
        logger = get_logger("t")
        with span(logger, "outer", a=1) as outcome:
            with span(logger, "inner"):
                pass
            outcome["result"] = "ok"
        inner, outer = self.records()
        self.assertEqual(inner["event"], "inner.completed")
        self.assertEqual(outer["event"], "outer.completed")
        self.assertEqual(outer["result"], "ok")
        self.assertGreaterEqual(outer["duration_ms"], 0)
        self.assertEqual(inner["request_id"], outer["request_id"])
        self.assertNotEqual(outer["request_id"], "-")
        self.assertEqual(current_request_id(), "-")

    def test_failed_span_logs_error_and_reraises(self) -> None:
        with self.assertRaises(ValueError):
            with span(get_logger("t"), "work", item="x"):
                raise ValueError("boom")
        record = self.records()[0]
        self.assertEqual(record["event"], "work.failed")
        self.assertEqual(record["level"], "ERROR")
        self.assertEqual(record["error_type"], "ValueError")
        self.assertEqual(record["error"], "boom")
        self.assertEqual(record["item"], "x")

    def test_explicit_request_id_is_used(self) -> None:
        with request_context("req-123"):
            log_event(get_logger("t"), "tagged")
        self.assertEqual(self.records()[0]["request_id"], "req-123")

    def test_secrets_are_redacted(self) -> None:
        log_event(
            get_logger("t"),
            "upstream.error",
            error="Incorrect API key provided: sk-proj-abcdef123456",
        )
        text = Path(self.log_file).read_text(encoding="utf-8")
        self.assertNotIn("sk-proj-abcdef123456", text)
        self.assertIn("[REDACTED]", text)

    def test_text_is_not_logged_unless_enabled(self) -> None:
        fields = describe_text("question", "What is the secret plan?")
        self.assertEqual(fields["question_chars"], 24)
        self.assertNotIn("question", fields)
        with patch.dict(os.environ, {"LOG_CONTENT": "true"}):
            self.assertEqual(
                describe_text("question", "hello")["question"], "hello"
            )

    def test_tool_arguments_hide_free_text_but_keep_identifiers(self) -> None:
        summary = summarize_arguments({"query": "tiny", "source": "a.csv", "k": 3})
        self.assertNotIn("query", summary)
        self.assertEqual(summary["query_chars"], 4)
        self.assertEqual(summary["source"], "a.csv")
        self.assertEqual(summary["k"], 3)

    def test_unwritable_log_dir_does_not_crash(self) -> None:
        _reset_logger()
        blocker = Path(self.tmp.name) / "file"
        blocker.write_text("x")
        with patch.dict(os.environ, {"LOG_DIR": str(blocker / "logs")}):
            self.assertIsNone(configure_logging())
        log_event(get_logger("t"), "still.works")


if __name__ == "__main__":
    unittest.main()
