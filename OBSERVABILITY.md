# Observability

EngiMind records every agent execution as structured JSON lines in `logs/`, and
ships a dashboard that reads those files. Observability is **passive**: it only
wraps existing calls, never changes arguments, return values, control flow or
exceptions, and a logging failure can never break the wrapped code.

## What is an "agent" here

| Agent name | Where | Role |
|---|---|---|
| `router` | `application/services/rag_service.py` | Picks CSV / Basic / LangChain / Agentic path |
| `basic_rag`, `langchain_rag`, `agentic_rag` | `application/` | Answer pipelines |
| `agentic_rag.retrieve` / `.rewrite` / `.clarify` / `.answer` | LangGraph nodes | One run per node visit |
| `csv_query_agent` | `application/rag.py` | Explains CSV query results |
| `indexer` | `application/rag.py` | Document ingestion |
| `eval_runner`, `eval_judge` | service / `evaluation/metrics.py` | Evaluation case and LLM judge |

MCP tool calls are logged as tools (`mcp:<name>`) inside a trace.

## Event types

| Event | Meaning |
|---|---|
| `agent.run.started` | A run began |
| `agent.run.completed` | Finished: status `success`, `refused` or `cancelled` |
| `agent.run.failed` | Raised an exception (status `failed`, `error_type`, `error`) |
| `llm.call.completed` / `.failed` | One chat call: model, prompt/completion/total tokens, `duration_ms` |
| `tool.call.completed` / `.failed` | Retrieval, CSV query, MCP tool: `tool`, `duration_ms` |
| `agent.handoff` | `from_agent` → `to_agent` with `reason` |
| `retry.attempted` | A retry (for example embedding rate limit) |

Embedding batches are summed into run totals (`embedding_calls`,
`embedding_tokens`) and logged individually only at `LOG_LEVEL=DEBUG`.

## IDs: following a workflow end to end

Every line carries:

- `trace_id` – one whole workflow (all agents, tools, LLM calls). The API
  accepts/returns `X-Trace-ID`.
- `request_id` – correlation ID of the request/interaction. API accepts/returns `X-Request-ID`.
- `run_id` / `parent_run_id` – one agent execution and its parent, forming a tree.
- `ts` – UTC timestamp, millisecond precision.

Run completion lines include: `agent`, `status`, `duration_ms`, `llm_calls`,
`prompt_tokens`, `completion_tokens`, `total_tokens`, `embedding_calls`,
`embedding_tokens`, `tool_calls`, `tool_errors`, `retries`, `handoffs`,
`models`, input/output summaries and any error.

Counters on a run are **its own**. A parent's cost including its children is
computed by the dashboard (`incl_*`), so nothing is double counted.

## Metric definitions

- **Success rate** = `success` runs / finished runs. **Failure rate** = `failed` / finished.
- **Refused** (“I can't find that in the documents”) is a valid completed
  answer and counted separately, not as a failure. `cancelled` (for example a
  Streamlit rerun interrupting a run) is also separate.
- **Incomplete** = started but never finished (still running or process killed).
- Latency uses p50/p95 over finished runs. Token and tool-call counts are sums of own counters.

## Dashboard

Start the app (`streamlit run app.py`) and open **Observability** in the sidebar
(`pages/Observability.py`). Filter by time window, agent, status, run ID prefix
and trace ID.

- **Overview** – KPIs, per-agent table and charts, per-tool and per-model tables, latency over time
- **Runs** – every run with status, tokens, retries, handoffs, IDs
- **Traces** – one row per workflow; pick one for an indented timeline of runs, handoffs, tools and LLM calls
- **Errors** – grouped by agent / error type with counts and last run ID; retries listed
- **Logs** – raw events filtered by level and text

It reads `engimind.log` and rotated `engimind.log.N` files, cached until files change.

## Privacy and redaction

- By default only input/output **length and a short hash** are logged, not text.
  Set `LOG_CONTENT=1` to add a 300-character preview.
- Everything is redacted before writing: OpenAI-style keys (`sk-…`), Bearer
  tokens, AWS/GitHub/JWT tokens, URL credentials, `key=value` secrets, emails,
  and any value under a sensitive key (`api_key`, `password`, `token`,
  `authorization`, …). Token **counts** (`total_tokens`) are not masked.

## Configuration

| Variable | Default | Purpose |
|---|---|---|
| `LOG_DIR` | `logs` | Log folder (use one absolute path for all processes) |
| `LOG_LEVEL` | `INFO` | `DEBUG` also logs each embedding batch |
| `LOG_CONTENT` | off | Add redacted text previews |
| `LOG_MAX_BYTES` / `LOG_BACKUP_COUNT` | 5 MB / 5 | Rotation |

Logging is enabled only by entry points (`app.py`, API, MCP server); tests and
imports write nothing.

## Known limits

- OpenAI SDK internal retries are invisible; only the app's own retries are logged.
- LangChain's OpenAI embeddings, and Ollama embeddings, report no token counts
  (chat tokens are captured via a callback on the LangChain model).
- The background indexing worker thread starts its own trace.
- On Windows a rotating file cannot be shared by several processes. If the API,
  MCP server and Streamlit run together, rotation can fail; raise `LOG_MAX_BYTES`
  or ship the files to a collector.
- The dashboard loads up to 200,000 recent events.

## Code map

`src/engimind/observability/`: `agents.py` (runs, LLM/tool/retry/handoff),
`context.py` (trace/run context), `redaction.py`, `logging_setup.py`,
`langchain_callback.py`, `analytics.py` (metrics from logs), `dashboard.py`.
Tests: `tests/test_observability.py`, `tests/test_agent_observability.py`.
