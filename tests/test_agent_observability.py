import json
import logging
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from engimind.application.agentic_rag import StateGraph, answer_question_agentic
from engimind.application.rag import RagResult
from engimind.evaluation.metrics import judge_answer
from engimind.observability import (
    agent_run,
    analytics,
    llm_call,
    record_handoff,
    record_retry,
    tool_call,
)
from engimind.observability.langchain_callback import UsageCallbackHandler
from engimind.observability.logging_setup import LOGGER_NAME, JsonLineFormatter
from engimind.observability.redaction import redact, redact_value


class ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.setFormatter(JsonLineFormatter())
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.lines.append(self.format(record))

    @property
    def events(self) -> list[dict]:
        return [json.loads(line) for line in self.lines]


class CaptureCase(unittest.TestCase):
    def setUp(self) -> None:
        self.handler = ListHandler()
        logger = logging.getLogger(LOGGER_NAME)
        self.previous_level = logger.level
        logger.addHandler(self.handler)
        logger.setLevel(logging.DEBUG)
        self.addCleanup(logger.removeHandler, self.handler)
        self.addCleanup(logger.setLevel, self.previous_level)

    def named(self, name: str) -> list[dict]:
        return [e for e in self.handler.events if e["event"] == name]


def fake_response(prompt=10, completion=5, model="served-model"):
    return SimpleNamespace(
        model=model,
        usage=SimpleNamespace(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=prompt + completion,
        ),
    )


class RunLifecycleTests(CaptureCase):
    def test_success_run_records_counters_and_ids(self) -> None:
        with agent_run("demo", input_text="hello") as run:
            with llm_call("m", purpose="answer") as call:
                call.record(fake_response())
            with tool_call("vector_search", k=3):
                pass
            record_retry("rate_limit")
            record_handoff("other", "because")
        done = self.named("agent.run.completed")[0]
        self.assertEqual(done["status"], "success")
        self.assertEqual(done["agent"], "demo")
        self.assertEqual(done["run_id"], run.run_id)
        self.assertEqual(done["total_tokens"], 15)
        self.assertEqual(done["llm_calls"], 1)
        self.assertEqual(done["tool_calls"], 1)
        self.assertEqual(done["retries"], 1)
        self.assertEqual(done["handoffs"], 1)
        self.assertEqual(done["models"], ["served-model"])
        self.assertEqual(done["input_chars"], 5)
        self.assertNotIn("input", done)
        self.assertTrue(done["trace_id"] and done["request_id"])
        self.assertIn("duration_ms", done)
        self.assertEqual(len(self.named("agent.run.started")), 1)

    def test_failed_run_reraises_original_exception(self) -> None:
        with self.assertRaises(KeyError):
            with agent_run("demo"):
                raise KeyError("boom")
        failed = self.named("agent.run.failed")[0]
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_type"], "KeyError")

    def test_cancelled_run_is_not_a_failure(self) -> None:
        with self.assertRaises(KeyboardInterrupt):
            with agent_run("demo"):
                raise KeyboardInterrupt()
        self.assertEqual(self.named("agent.run.completed")[0]["status"], "cancelled")
        self.assertEqual(self.named("agent.run.failed"), [])

    def test_refused_status(self) -> None:
        with agent_run("demo") as run:
            run.set_status("refused")
        self.assertEqual(self.named("agent.run.completed")[0]["status"], "refused")

    def test_nested_runs_share_trace_and_link_parent(self) -> None:
        with agent_run("outer") as outer:
            with agent_run("inner") as inner:
                with llm_call("m", purpose="x") as call:
                    call.record(fake_response())
        events = self.named("agent.run.completed")
        by_agent = {e["agent"]: e for e in events}
        self.assertEqual(by_agent["inner"]["parent_run_id"], outer.run_id)
        self.assertEqual(by_agent["inner"]["trace_id"], by_agent["outer"]["trace_id"])
        self.assertEqual(by_agent["outer"]["total_tokens"], 0)
        self.assertEqual(by_agent["inner"]["total_tokens"], 15)
        self.assertEqual(inner.run_id, by_agent["inner"]["run_id"])

    def test_failing_llm_call_is_logged_and_reraised(self) -> None:
        with self.assertRaises(ValueError):
            with agent_run("demo"):
                with llm_call("m", purpose="x"):
                    raise ValueError("bad")
        self.assertEqual(self.named("llm.call.failed")[0]["error_type"], "ValueError")

    def test_telemetry_failure_never_breaks_wrapped_code(self) -> None:
        with patch(
            "engimind.observability.agents.log_event", side_effect=RuntimeError("disk")
        ):
            with agent_run("demo"):
                with tool_call("t"):
                    value = 41 + 1
        self.assertEqual(value, 42)

    def test_mock_usage_is_ignored(self) -> None:
        from unittest.mock import MagicMock

        with agent_run("demo"):
            with llm_call("m", purpose="x") as call:
                call.record(MagicMock())
        self.assertEqual(self.named("agent.run.completed")[0]["total_tokens"], 0)


class RedactionTests(unittest.TestCase):
    def test_secrets_are_masked(self) -> None:
        text = "key sk-abcdefghijklmnopqrstuvwxyz123456 and Bearer abc.def.ghi-123456 mail a@b.com"
        masked = redact(text)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", masked)
        self.assertNotIn("a@b.com", masked)
        self.assertNotIn("abc.def.ghi-123456", masked)

    def test_sensitive_keys_masked_but_token_counts_kept(self) -> None:
        cleaned = redact_value(
            {
                "api_key": "x",
                "password": "y",
                "access_token": "z",
                "total_tokens": 12,
                "prompt_tokens": 7,
                "nested": {"authorization": "Bearer q"},
            }
        )
        self.assertNotIn(cleaned["api_key"], ("x",))
        self.assertNotEqual(cleaned["password"], "y")
        self.assertNotEqual(cleaned["access_token"], "z")
        self.assertEqual(cleaned["total_tokens"], 12)
        self.assertEqual(cleaned["prompt_tokens"], 7)
        self.assertNotEqual(cleaned["nested"]["authorization"], "Bearer q")

    def test_error_text_is_redacted_in_log_line(self) -> None:
        handler = ListHandler()
        logger = logging.getLogger(LOGGER_NAME)
        logger.addHandler(handler)
        previous = logger.level
        logger.setLevel(logging.INFO)
        self.addCleanup(logger.removeHandler, handler)
        self.addCleanup(logger.setLevel, previous)
        with self.assertRaises(RuntimeError):
            with agent_run("demo"):
                raise RuntimeError("failed with sk-abcdefghijklmnopqrstuvwxyz123456")
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", "\n".join(handler.lines))


class InstrumentedPathTests(CaptureCase):
    def test_judge_answer_records_usage_and_returns_unchanged_scores(self) -> None:
        response = fake_response(20, 10)
        response.choices = [
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(
                        {
                            "relevance": 1,
                            "groundedness": 0.5,
                            "relevance_reason": "ok",
                            "groundedness_reason": "partly",
                        }
                    )
                )
            )
        ]
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **_: response)
            )
        )
        scores = judge_answer("q", "ref", "ans", ["ctx"], openai_client=client, model="m")
        self.assertEqual(scores.relevance, 1)
        self.assertEqual(scores.groundedness, 0.5)
        done = self.named("agent.run.completed")[0]
        self.assertEqual(done["agent"], "eval_judge")
        self.assertEqual(done["total_tokens"], 30)

    @unittest.skipIf(StateGraph is None, "LangGraph is not installed")
    def test_agentic_graph_logs_node_runs_and_handoffs(self) -> None:
        settings = SimpleNamespace(
            openai_api_key="k",
            embedding_model="e",
            chat_model="c",
            chroma_path=".chroma",
            top_k=4,
            max_distance=0.55,
        )
        chunk = SimpleNamespace(
            text="Employees receive 20 days.",
            page=2,
            source="p.pdf",
            distance=0.3,
            section="s",
        )
        with patch(
            "engimind.application.agentic_rag.query_collection",
            side_effect=[[], [chunk]],
        ), patch("engimind.application.agentic_rag.OpenAI"), patch(
            "engimind.application.agentic_rag._generate_answer", return_value="A [p. 2]"
        ):
            result = answer_question_agentic(
                "How many vacation days?", "c", settings, openai_client=SimpleNamespace()
            )
        self.assertEqual(result.answer, "A [p. 2]")
        runs = {e["agent"] for e in self.named("agent.run.completed")}
        self.assertIn("agentic_rag", runs)
        self.assertIn("agentic_rag.retrieve", runs)
        self.assertTrue(self.named("agent.handoff"))
        self.assertEqual(len({e["trace_id"] for e in self.handler.events}), 1)


class CallbackTests(CaptureCase):
    def test_langchain_usage_callback_counts_tokens(self) -> None:
        from langchain_core.outputs import ChatGeneration, LLMResult
        from langchain_core.messages import AIMessage

        handler = UsageCallbackHandler("m")
        with agent_run("lc"):
            handler.on_chat_model_start({}, [[]], run_id="r1")
            message = AIMessage(
                content="hi",
                usage_metadata={"input_tokens": 4, "output_tokens": 2, "total_tokens": 6},
            )
            handler.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]), run_id="r1")
        self.assertEqual(self.named("agent.run.completed")[0]["total_tokens"], 6)


def event(name, ts, **fields):
    return {"event": name, "ts": ts.isoformat(), "level": "INFO", **fields}


class AnalyticsTests(unittest.TestCase):
    def setUp(self) -> None:
        base = datetime(2026, 1, 1, tzinfo=timezone.utc)
        t = lambda s: base + timedelta(seconds=s)  # noqa: E731
        trace = {"trace_id": "T1", "request_id": "R1"}
        self.now = t(100)
        self.events = [
            event("agent.run.started", t(0), run_id="a", agent="router", **trace),
            event("agent.run.started", t(1), run_id="b", agent="basic_rag", parent_run_id="a", **trace),
            event("tool.call.completed", t(2), run_id="b", agent="basic_rag", tool="vector_search", duration_ms=10, **trace),
            event("llm.call.completed", t(3), run_id="b", agent="basic_rag", model="m", prompt_tokens=8, completion_tokens=2, total_tokens=10, duration_ms=300, **trace),
            event("agent.run.completed", t(4), run_id="b", agent="basic_rag", parent_run_id="a", status="success", duration_ms=3000, total_tokens=10, llm_calls=1, tool_calls=1, **trace),
            event("agent.handoff", t(1.5), run_id="a", agent="router", from_agent="router", to_agent="basic_rag", **trace),
            event("agent.run.completed", t(5), run_id="a", agent="router", status="success", duration_ms=5000, handoffs=1, **trace),
            event("agent.run.started", t(10), run_id="c", agent="basic_rag", trace_id="T2", request_id="R2"),
            {**event("agent.run.failed", t(11), run_id="c", agent="basic_rag", status="failed", duration_ms=1000, error_type="RuntimeError", error="x", trace_id="T2", request_id="R2"), "level": "ERROR"},
            event("agent.run.started", t(20), run_id="d", agent="indexer", trace_id="T3", request_id="R3"),
        ]

    def test_build_runs_merges_and_computes_inclusive_totals(self) -> None:
        runs = {r["run_id"]: r for r in analytics.build_runs(self.events)}
        self.assertEqual(runs["a"]["incl_total_tokens"], 10)
        self.assertEqual(runs["a"]["total_tokens"], 0)
        self.assertEqual(runs["b"]["status"], "success")
        self.assertEqual(runs["c"]["status"], "failed")
        self.assertEqual(runs["d"]["status"], "incomplete")

    def test_summary_rates_exclude_incomplete(self) -> None:
        summary = analytics.summarize_runs(analytics.build_runs(self.events))
        self.assertEqual(summary["finished"], 3)
        self.assertAlmostEqual(summary["success_rate"], 2 / 3, places=3)
        self.assertAlmostEqual(summary["failure_rate"], 1 / 3, places=3)
        self.assertEqual(summary["incomplete"], 1)
        self.assertEqual(summary["tokens"], 10)

    def test_agent_tool_llm_metrics(self) -> None:
        runs = analytics.build_runs(self.events)
        agents = {r["agent"]: r for r in analytics.agent_metrics(runs)}
        self.assertEqual(agents["basic_rag"]["runs"], 2)
        self.assertEqual(agents["basic_rag"]["failed"], 1)
        self.assertEqual(analytics.tool_metrics(self.events)[0]["calls"], 1)
        self.assertEqual(analytics.llm_metrics(self.events)[0]["total_tokens"], 10)

    def test_filters(self) -> None:
        runs = analytics.build_runs(self.events)
        self.assertEqual(len(analytics.filter_runs(runs, agents=["indexer"])), 1)
        self.assertEqual(len(analytics.filter_runs(runs, statuses=["failed"])), 1)
        self.assertEqual(len(analytics.filter_runs(runs, run_id="b")), 1)
        self.assertEqual(len(analytics.filter_runs(runs, trace_id="T1")), 2)
        recent = analytics.filter_runs(runs, since=timedelta(seconds=95), now=self.now)
        self.assertEqual({r["run_id"] for r in recent}, {"c", "d"})
        self.assertEqual(len(analytics.filter_events(self.events, run_id="b")), 4)

    def test_error_summary_and_timeline(self) -> None:
        summary = analytics.error_summary(self.events)
        self.assertEqual(summary[0]["error_type"], "RuntimeError")
        self.assertEqual(summary[0]["last_run_id"], "c")
        rows = analytics.trace_timeline(self.events, "T1")
        kinds = [r["kind"] for r in rows]
        self.assertIn("handoff", kinds)
        self.assertEqual(rows[0]["offset_ms"], 0.0)
        child = next(r for r in rows if r["kind"] == "tool")
        self.assertEqual(child["depth"], 2)
        traces = {t["trace_id"]: t for t in analytics.trace_summaries(analytics.build_runs(self.events))}
        self.assertEqual(traces["T1"]["runs"], 2)
        self.assertEqual(traces["T2"]["status"], "failed")

    def test_load_events_reads_rotated_files_and_skips_bad_lines(self) -> None:
        from pathlib import Path
        from tempfile import TemporaryDirectory

        with TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
            old = {"event": "x", "ts": "2026-01-01T00:00:00+00:00"}
            new = {"event": "y", "ts": "2026-01-02T00:00:00+00:00"}
            Path(tmp, "engimind.log.1").write_text(json.dumps(old) + "\n", encoding="utf-8")
            Path(tmp, "engimind.log").write_text("not json\n" + json.dumps(new) + "\n", encoding="utf-8")
            events = analytics.load_events(tmp)
        self.assertEqual([e["event"] for e in events], ["x", "y"])


if __name__ == "__main__":
    unittest.main()


class DashboardAnalyticsTests(unittest.TestCase):
    def test_cost_handoffs_status_and_series(self):
        from datetime import datetime, timedelta, timezone

        from engimind.observability import analytics

        now = datetime.now(timezone.utc)
        ts = now.isoformat()
        events = [
            {"ts": ts, "event": "agent.run.started", "run_id": "r1", "agent": "a", "trace_id": "t"},
            {"ts": ts, "event": "llm.call.completed", "run_id": "r1", "model": "m",
             "prompt_tokens": 1_000_000, "completion_tokens": 500_000, "total_tokens": 1_500_000},
            {"ts": ts, "event": "llm.call.completed", "run_id": "r1", "model": "other", "total_tokens": 5},
            {"ts": ts, "event": "agent.handoff", "from_agent": "a", "to_agent": "b"},
            {"ts": ts, "event": "agent.run.completed", "run_id": "r1", "agent": "a",
             "status": "success", "duration_ms": 5, "trace_id": "t"},
        ]
        pricing = analytics.parse_pricing("m, 1, 2\n# c\nbad line")
        costs = analytics.cost_summary(events, pricing)
        assert costs["total"] == 2.0
        assert costs["by_run"] == {"r1": 2.0}
        assert costs["unpriced"] == ["other"]
        assert analytics.handoff_metrics(events)[0]["count"] == 1
        runs = analytics.build_runs(events)
        assert analytics.agent_status(runs, now=now + timedelta(seconds=1))[0]["state"] == "healthy"
        assert analytics.agent_status(runs, now=now + timedelta(days=1))[0]["state"] == "idle"
        assert analytics.time_series(runs, timedelta(minutes=1))[0]["success"] == 1
