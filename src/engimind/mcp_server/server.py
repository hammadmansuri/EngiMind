from __future__ import annotations

from functools import wraps
from typing import Any, Callable

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations

from engimind.observability import (
    configure_logging,
    request_context,
    summarize_arguments,
    tool_call,
)
from engimind.tools import DocumentTools


INSTRUCTIONS = (
    "Search and ask questions about PDF and CSV documents indexed by EngiMind. "
    "Call list_documents first to see available sources. Use search_documents to "
    "inspect evidence and ask_document for a cited answer. All tools are read-only."
)

_READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)


def _report_expected_errors(function: Callable[..., Any]) -> Callable[..., Any]:
    """Show anticipated failures (bad input, missing config) to the model.

    The SDK hides the text of any other exception, so unexpected errors such as
    upstream API failures stay generic instead of leaking internals.
    """

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return function(*args, **kwargs)
        except (ValueError, RuntimeError) as exc:
            raise ToolError(str(exc)) from exc

    return wrapper


def _observed(function: Callable[..., Any]) -> Callable[..., Any]:
    """Log every tool call with its duration and outcome."""

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        with request_context(), tool_call(
            f"mcp:{function.__name__}", **summarize_arguments(kwargs)
        ):
            return function(*args, **kwargs)

    return wrapper


def create_server(tools: DocumentTools | None = None) -> MCPServer:
    tools = tools or DocumentTools()
    server = MCPServer("engimind", instructions=INSTRUCTIONS)
    for method in (tools.list_documents, tools.search_documents, tools.ask_document):
        server.add_tool(
            _report_expected_errors(_observed(method)), annotations=_READ_ONLY
        )
    return server


def main() -> None:
    configure_logging()
    create_server().run(transport="stdio")
