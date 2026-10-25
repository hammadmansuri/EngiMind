import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embedding_model: str
    chat_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    chroma_path: str
    max_distance: float
    csv_rows_per_chunk: int = 1
    embedding_provider: str = "openai"
    ollama_base_url: str = "http://localhost:11434"


def load_settings(
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    top_k: int | None = None,
    max_distance: float | None = None,
    csv_rows_per_chunk: int | None = None,
    embedding_provider: str | None = None,
) -> Settings:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Set it for answer generation and Eval "
            "judging; Ollama can provide local embeddings."
        )
    provider = (
        embedding_provider or os.getenv("EMBEDDING_PROVIDER", "openai")
    ).strip().lower()
    if provider not in {"openai", "ollama"}:
        raise ValueError("EMBEDDING_PROVIDER must be either 'openai' or 'ollama'.")
    model = (
        os.getenv("OLLAMA_EMBEDDING_MODEL", "all-minilm")
        if provider == "ollama"
        else os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    return Settings(
        openai_api_key=key,
        embedding_model=model,
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        chunk_size=_int("CHUNK_SIZE", 800, chunk_size),
        chunk_overlap=_int("CHUNK_OVERLAP", 150, chunk_overlap),
        top_k=_int("TOP_K", 4, top_k),
        chroma_path=os.getenv("CHROMA_PATH", ".chroma"),
        max_distance=(
            max_distance
            if max_distance is not None
            else float(os.getenv("MAX_DISTANCE", "0.55"))
        ),
        csv_rows_per_chunk=_int("CSV_ROWS_PER_CHUNK", 1, csv_rows_per_chunk),
        embedding_provider=provider,
        ollama_base_url=os.getenv(
            "OLLAMA_BASE_URL", "http://localhost:11434"
        ).rstrip("/"),
    )


def _int(name: str, default: int, override: int | None) -> int:
    if override is not None:
        return override
    return int(os.getenv(name, str(default)))
