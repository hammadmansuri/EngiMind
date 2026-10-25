import re
import time
import json
from collections.abc import Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from openai import OpenAI, RateLimitError

from engimind.core.config import Settings
from engimind.observability import record_llm_call, record_retry, usage_from_response

_BATCH = 64
_MAX_RATE_LIMIT_RETRIES = 5


def embed_texts(
    client: OpenAI | None,
    settings: Settings,
    texts: list[str],
    *,
    on_retry: Callable[[str], None] | None = None,
) -> list[list[float]]:
    vectors: list[list[float]] = []
    started = time.perf_counter()
    tokens = 0
    for i in range(0, len(texts), _BATCH):
        batch = texts[i : i + _BATCH]
        if getattr(settings, "embedding_provider", "openai") == "ollama":
            vectors.extend(_ollama_embed(settings, batch))
            continue
        if client is None:
            raise ValueError("An OpenAI client is required for OpenAI embeddings.")
        for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
            try:
                response = client.embeddings.create(
                    model=settings.embedding_model, input=batch
                )
                break
            except RateLimitError as exc:
                if not _is_retryable_rate_limit(exc):
                    raise
                if attempt >= _MAX_RATE_LIMIT_RETRIES:
                    raise
                delay = _retry_delay(exc, attempt)
                record_retry(
                    "embedding_rate_limit",
                    delay_s=round(delay, 2),
                    attempt=attempt + 1,
                    max_retries=_MAX_RATE_LIMIT_RETRIES,
                    model=settings.embedding_model,
                )
                if on_retry is not None:
                    on_retry(
                        f"Embedding rate limit reached; retrying in {delay:.1f}s "
                        f"(retry {attempt + 1}/{_MAX_RATE_LIMIT_RETRIES})"
                    )
                time.sleep(delay)
        ordered = sorted(response.data, key=lambda row: row.index)
        vectors.extend(item.embedding for item in ordered)
        tokens += usage_from_response(response)["total"] or 0
    if texts:
        record_llm_call(
            model=settings.embedding_model,
            purpose="embed",
            kind="embedding",
            provider=getattr(settings, "embedding_provider", "openai"),
            duration_ms=round((time.perf_counter() - started) * 1000, 2),
            usage={"total": tokens or None},
        )
    return vectors


def _ollama_embed(settings: Settings, texts: list[str]) -> list[list[float]]:
    payload = json.dumps(
        {"model": settings.embedding_model, "input": texts}
    ).encode("utf-8")
    request = Request(
        f"{settings.ollama_base_url}/api/embed",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read())
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama embedding request failed (HTTP {exc.code}): {detail}"
        ) from exc
    except URLError as exc:
        raise RuntimeError(
            f"Could not connect to Ollama at {settings.ollama_base_url}. "
            "Start Ollama and ensure the embedding model is pulled."
        ) from exc
    except (TimeoutError, json.JSONDecodeError) as exc:
        raise RuntimeError("Ollama embedding request timed out or returned invalid JSON.") from exc

    embeddings = result.get("embeddings")
    if (
        not isinstance(embeddings, list)
        or len(embeddings) != len(texts)
        or any(not isinstance(vector, list) or not vector for vector in embeddings)
    ):
        raise RuntimeError("Ollama returned an invalid embedding response.")
    if any(
        any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector)
        for vector in embeddings
    ):
        raise RuntimeError("Ollama returned a non-numeric embedding vector.")
    dimensions = len(embeddings[0])
    if any(len(vector) != dimensions for vector in embeddings):
        raise RuntimeError("Ollama returned embedding vectors with inconsistent dimensions.")
    return embeddings


def _retry_delay(error: RateLimitError, attempt: int) -> float:
    headers = error.response.headers
    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms is not None:
        try:
            return min(60.0, max(0.1, float(retry_after_ms) / 1000))
        except ValueError:
            pass

    retry_after = headers.get("retry-after")
    if retry_after is not None:
        try:
            return min(60.0, max(0.1, float(retry_after)))
        except ValueError:
            match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*(ms|s)?\s*", retry_after)
            if match:
                delay = float(match.group(1))
                if match.group(2) == "ms":
                    delay /= 1000
                return min(60.0, max(0.1, delay))

    return min(30.0, 2.0**attempt)


def _is_retryable_rate_limit(error: RateLimitError) -> bool:
    body = error.body
    if not isinstance(body, dict):
        return True
    details = body.get("error")
    if not isinstance(details, dict):
        return True
    code = details.get("code")
    return code in (None, "rate_limit_exceeded")
