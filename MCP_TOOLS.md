# Tools and the MCP server

This guide explains what "tools" and the Model Context Protocol (MCP) are, and
how EngiMind uses them to let AI assistants search and ask questions about your
indexed PDFs and CSVs.

## 1. The concepts

**Tool.** A function an AI model is allowed to call. The model sees a name, a
description, and an input schema. When it decides it needs the function, the
host application runs it and hands the result back to the model. Tools turn a
model that can only talk into one that can act: fetch data, search, calculate.

**MCP (Model Context Protocol).** An open standard for connecting AI
applications to tools and data. Without it, every assistant needs custom
integration code for every tool. With it, you write one **MCP server**, and any
**MCP client** (VS Code Copilot, Claude Desktop, MCP Inspector, your own code)
can discover and call its tools.

```text
 MCP client (VS Code, Claude Desktop, ...)      MCP server (this repo)
 ┌────────────────────────────────────┐          ┌───────────────────────────┐
 │ model decides: "I need the document │  stdio   │ engimind.mcp_server        │
 │ evidence" → calls search_documents  │ ───────▶ │  → engimind.tools          │
 │                                      │ ◀─────── │  → Chroma / Ollama / OpenAI│
 └────────────────────────────────────┘  result   └───────────────────────────┘
```

MCP servers can offer **tools** (actions), **resources** (readable data), and
**prompts** (templates). EngiMind exposes tools only.

The transport used here is **stdio**: the client starts the server as a child
process and exchanges JSON messages over its standard input/output. Nothing
listens on a network port.

## 2. How this fits the RAG project

The RAG pipeline already knows how to retrieve and answer. Tools wrap that
existing code as small, safe operations an assistant can call:

| Tool | Does | Calls |
|---|---|---|
| `list_documents` | Lists indexed documents and whether each is searchable | Chroma metadata |
| `search_documents` | Returns the most relevant passages, no generated answer | `query_collection` |
| `ask_document` | Returns a grounded, cited answer plus its evidence | `answer_question` |

An assistant typically calls `list_documents` first, then either
`search_documents` (to inspect evidence itself) or `ask_document` (to get a
finished answer).

## 3. Code layout

```text
src/engimind/
├── tools/
│   └── document_tools.py   # DocumentTools: plain Python, no MCP imports
└── mcp_server/
    ├── server.py           # registers DocumentTools methods as MCP tools
    └── __main__.py         # enables: python -m engimind.mcp_server
```

The design separates two concerns:

- **`engimind.tools`** holds the behavior: validation, source filtering,
  retrieval, and answering. It returns plain dicts and lists, and does not know
  MCP exists. The same class could back an HTTP endpoint or an OpenAI
  function-calling loop.
- **`engimind.mcp_server`** is a thin adapter: it registers the methods, adds
  read-only annotations, and translates errors for the protocol.

MCP derives each tool's input schema from the method's type hints, and its
description from the docstring. That is why the methods have precise hints and
docstrings with an `Args:` section.

## 4. The tools

### `list_documents()`

Returns one entry per indexed document:

```json
{
  "source": "matches.csv",
  "collection": "pdf_b1b766bf0629bfa7ce9d01311c4ee3ea",
  "chunks": 128,
  "embedding_provider": "ollama",
  "embedding_model": "all-minilm",
  "searchable": true
}
```

`searchable` is `false` when the document was indexed with a different
embedding provider/model than the server is configured to use. Vectors from
different embedding models are not comparable, so the other tools skip those
documents. The fix is to re-index the document with the current setting.

### `search_documents(query, k=4, source=None)`

Retrieves passages only; no language model writes an answer.

| Argument | Meaning |
|---|---|
| `query` | Natural-language search text (required, up to 2,000 characters) |
| `k` | Number of passages, clamped to 1-20 |
| `source` | Optional document name from `list_documents`, matched case-insensitively |

```json
{
  "query": "annual leave",
  "results": [
    {
      "source": "policy.pdf",
      "location": "page=2",
      "section": "Leave",
      "distance": 0.2314,
      "text": "Staff receive 20 days of annual leave..."
    }
  ]
}
```

`location` is `page=N` or `pages=A-B` for PDFs, and `row=N` or `rows=A-B` for
CSVs. `distance` is cosine distance; lower is closer.

### `ask_document(question, source=None, k=None)`

Runs the full pipeline: exact CSV query when applicable, otherwise retrieval,
then a grounded answer with citations.

```json
{
  "answer": "Staff receive 20 days of annual leave [p. 2].",
  "refused": false,
  "sources": [ { "source": "policy.pdf", "location": "page=2", "...": "..." } ],
  "timings_ms": { "retrieval": 41.2, "generation": 812.5 }
}
```

`refused: true` means retrieval found nothing close enough, so the system
declined to answer instead of guessing. The prompt sent to the model is
deliberately **not** returned.

## 5. Running the server

Install the project, then start the server directly to confirm it launches:

```powershell
pip install -e .
python -m engimind.mcp_server      # or: engimind-mcp
```

It waits silently for a client on stdin; press `Ctrl+C` to stop. Normally you do
not start it by hand: the client launches it.

### Configure a client

Use absolute paths. Clients start the server from an arbitrary working
directory, so relative paths such as the default `.chroma` will not resolve.

**VS Code** (`.vscode/mcp.json`):

```json
{
  "servers": {
    "engimind": {
      "type": "stdio",
      "command": "C:\\Users\\hamma\\source\\repos\\EngiMind\\.venv\\Scripts\\python.exe",
      "args": ["-m", "engimind.mcp_server"],
      "env": {
        "OPENAI_API_KEY": "${input:openai-key}",
        "CHROMA_PATH": "C:\\Users\\hamma\\source\\repos\\EngiMind\\.chroma",
        "EMBEDDING_PROVIDER": "ollama"
      }
    }
  },
  "inputs": [
    {
      "type": "promptString",
      "id": "openai-key",
      "description": "OpenAI API key",
      "password": true
    }
  ]
}
```

**Claude Desktop** (`claude_desktop_config.json`) uses the same command, args,
and env under a top-level `mcpServers` key instead of `servers`.

### Configuration rules that matter

| Setting | Rule |
|---|---|
| `command` | The `python.exe` inside this project's `.venv`, so dependencies resolve |
| `CHROMA_PATH` | Absolute path to the same `.chroma` folder the Streamlit app writes |
| `EMBEDDING_PROVIDER` / model | Must match how the documents were indexed, or they are skipped as not searchable |
| `OPENAI_API_KEY` | Pass it in the client's `env` block. Clients give servers a **restricted environment**, so a key set only in your terminal is not inherited |

`OPENAI_API_KEY` is required even with Ollama embeddings, because the shared
settings loader requires it and `ask_document` uses OpenAI for the answer.

## 6. Errors the client sees

Anticipated problems are reported with a readable message so the model can
correct itself:

| Situation | Message (abridged) |
|---|---|
| Empty or oversized input | `query must not be empty.` |
| Unknown document name | `Unknown source 'x'. Indexed documents: ...` |
| Nothing indexed | `No documents are indexed. Index a PDF or CSV first.` |
| Embedding mismatch | `No matching document was indexed with the current embedding setting (...). Re-index it...` |
| Missing configuration | `OPENAI_API_KEY is missing...` |

Unexpected failures, such as an upstream API outage, return only a generic
`Error executing tool <name>`. The SDK withholds details from the client and
logs the traceback on the server's stderr, which avoids leaking internals.

## 7. Safety design

- **Read-only.** All three tools carry `readOnlyHint`, and none writes data.
  Indexing, re-indexing, and deleting are intentionally not exposed: a model
  that can be prompt-injected by document text should not be able to change or
  destroy the index.
- **No file paths or URLs as input.** Tools accept only text and a document
  *name*, which is matched against what is already indexed.
- **Bounded input.** Query length and `k` are capped.
- **Treat retrieved text as untrusted.** Passages come from uploaded documents
  and could contain instructions aimed at the model. The server returns them as
  data; the client model must not obey them.
- **No network port.** stdio only. If you later add an HTTP transport, add
  authentication first; see `PRODUCTION_RAG_CHECKLIST.md`.
- **Do not re-index while a client is querying.** Re-indexing deletes and
  recreates a collection, so concurrent reads can fail.

## 8. Try it

With the MCP Inspector (requires Node.js):

```powershell
npx @modelcontextprotocol/inspector C:\Users\hamma\source\repos\EngiMind\.venv\Scripts\python.exe -m engimind.mcp_server
```

Set `OPENAI_API_KEY`, `CHROMA_PATH`, and `EMBEDDING_PROVIDER` in the Inspector's
environment fields, then call `list_documents` and `search_documents`.

In an MCP-aware chat, try prompts such as:

- "List my indexed documents."
- "Search matches.csv for games between Sunrisers Hyderabad and Royal
  Challengers Bangalore."
- "Using policy.pdf, how many annual leave days do staff get? Cite the page."

To test from Python, use the MCP SDK's `stdio_client` and `ClientSession`:
start the server with `python -m engimind.mcp_server`, call `initialize()`,
then `list_tools()` and `call_tool(name, arguments)`. Pass the client an `env`
containing `OPENAI_API_KEY`, `CHROMA_PATH`, and `EMBEDDING_PROVIDER`.

## 9. Adding a new tool

1. Add a method to `DocumentTools` (or a new class in `engimind/tools/`) with
   full type hints and a docstring containing an `Args:` section. Return
   JSON-serializable data and raise `ValueError` for bad input.
2. Register it in `create_server` in `mcp_server/server.py` by adding it to the
   tuple of methods.
3. Add tests in `tests/test_tools.py`, including a case for invalid input.
4. Document it in section 4 of this file.

Keep new tools read-only unless you have a specific reason and a review of the
risks in section 7.

## 10. Troubleshooting

| Symptom | Cause and fix |
|---|---|
| Client shows the server failed to start | Wrong `command`; use the `.venv` `python.exe` with an absolute path |
| `OPENAI_API_KEY is missing` | Add it to the client's `env` block |
| `list_documents` is empty | `CHROMA_PATH` points to a different folder than the app's index |
| Every document shows `searchable: false` | Provider/model differs from indexing; align `EMBEDDING_PROVIDER` or re-index |
| `Ollama embedding request failed` | Ollama is not running, or the model is missing; run `ollama pull all-minilm` |
| `ModuleNotFoundError: mcp.server.fastmcp` | You are on MCP SDK v2; this repo uses `mcp.server.mcpserver`. Install with `pip install -e .` |
| Garbled protocol errors | Something wrote to stdout. A stdio server must never `print`; use logging, which goes to stderr |

## Related guides

- [README.md](./README.md): setup and code map
- [PRODUCTION_RAG_CHECKLIST.md](./PRODUCTION_RAG_CHECKLIST.md): access control and
  hardening before exposing tools beyond your machine
- [RAG_EVALUATION.md](./RAG_EVALUATION.md): measuring answer quality
- [LANGGRAPH_AGENTIC_RAG.md](./LANGGRAPH_AGENTIC_RAG.md): the in-app agentic flow
