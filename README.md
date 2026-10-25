# EngiMind

EngiMind is a document question-answering application for PDF and CSV files. It combines semantic retrieval with source-aware answers, so users can inspect the pages or rows behind a response.

## Capabilities

- Ingest text-based PDFs and UTF-8 CSV files.
- Configure chunking, retrieval depth, and a maximum retrieval distance.
- Generate embeddings with OpenAI or a locally hosted Ollama model.
- Answer questions with page- or row-level source references.
- Query CSV data directly for supported exact-match, count, and date questions.
- Compare custom, LangChain, and agentic retrieval workflows.
- Evaluate retrieval and answer quality against labeled question sets.
- Inspect structured execution logs in the observability dashboard.
- Optionally expose HTTP API endpoints and read-only document tools over MCP.

## Requirements

- Python 3.10 or later.
- An OpenAI API key for answer generation and evaluation.
- For local embeddings, a running Ollama service with the configured embedding model installed.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e .
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. The default models are `text-embedding-3-small` and `gpt-4o-mini`.

To use Ollama for embeddings, set `EMBEDDING_PROVIDER=ollama`, configure `OLLAMA_BASE_URL` and `OLLAMA_EMBEDDING_MODEL`, and install that model locally. OpenAI remains the answer-generation and evaluation provider. Re-index documents after changing the embedding provider or model.

## Run the application

```powershell
streamlit run app.py
```

Upload one or more text-based PDFs or UTF-8 CSVs, index them, and ask questions in the application. Scanned PDFs are not OCR-processed. Indexed content and conversation state are local to the running application session.

### Optional HTTP API

```powershell
uvicorn engimind.api.main:app --reload
```

### Optional MCP server

```powershell
python -m engimind.mcp_server
```

The MCP server provides read-only document listing, search, and question-answering tools over standard input/output. See [MCP_TOOLS.md](./MCP_TOOLS.md) for client setup and tool details.

## Architecture

The Streamlit UI and optional API/MCP entry points share the application workflows in `src/engimind/`. PDF and CSV loaders prepare source-aware chunks; embeddings and Chroma support semantic retrieval; CSV rows are also persisted for structured queries. The package includes separate custom, LangChain, and agentic RAG paths, plus evaluation and observability components.

## Configuration

Copy `.env.example` to `.env` and set the required provider credentials and any desired model or retrieval settings. `.env` is excluded from version control; do not commit credentials or private documents.

## Quality and operational notes

- Run the test suite with `python -m pytest`.
- Logs, vector indexes, and local environment files are excluded from version control.
- The API and indexing metadata use local/in-process storage in this implementation; they are not configured for durable multi-instance operation.
- Retrieval quality and response latency depend on the selected models, data, and settings.

See [RAG_EVALUATION.md](./RAG_EVALUATION.md) for evaluation details, [OBSERVABILITY.md](./OBSERVABILITY.md) for execution telemetry, and [PRODUCTION_RAG_CHECKLIST.md](./PRODUCTION_RAG_CHECKLIST.md) for deployment considerations.
