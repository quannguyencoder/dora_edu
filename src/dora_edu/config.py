"""System configurations and environment variables loading."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables or a ``.env`` file.

    Every value has a sensible default so that the pure-logic layers
    (chunking, prompt building) can be imported and unit-tested without any
    credential being present.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # --- Messaging channels -------------------------------------------------
    telegram_bot_token: str | None = None
    discord_bot_token: str | None = None
    zalo_access_token: str | None = None
    zalo_oa_secret: str | None = None
    zalo_webhook_host: str = "0.0.0.0"
    zalo_webhook_port: int = Field(default=8080, ge=1, le=65535)

    # --- LLM generation -----------------------------------------------------
    #: Which backend `build_generator()` constructs: "openai" or "gemini".
    llm_provider: str = "openai"
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    gemini_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"
    #: Comma-separated, tried in order after ``llm_model`` is rate-limited
    #: (HTTP 429). The Gemini free tier quotas each model separately, so a
    #: lightweight model with a much higher daily quota keeps the bot
    #: answering once the primary model's quota is exhausted, instead of
    #: refusing every question until the quota resets. Only used by the
    #: Gemini backend; parsed with :func:`parse_model_list`.
    gemini_fallback_models: str = "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    llm_temperature: float = 0.3
    #: Generous headroom for "thinking" models (e.g. Gemini), which spend a
    #: chunk of this budget on internal reasoning before the visible answer.
    llm_max_tokens: int = 2048
    llm_timeout_seconds: float = 30.0

    # --- Vector store -------------------------------------------------------
    chroma_db_path: Path = Path("./vector_db")
    chroma_collection_name: str = "moet_textbooks"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # --- Retrieval tuning ---------------------------------------------------
    retrieval_top_k: int = Field(default=10, ge=1, le=20)
    max_retrieval_distance: float = Field(default=1.1, gt=0.0)
    # Acts as a safety cap, not a target: the chunker now cuts primarily at
    # heading-like lines ("a. Lựa chọn đề tài", "1. TRƯỚC KHI VIẾT", "BÀI 1")
    # so a labelled subsection stays in one chunk, and only falls back to
    # this character budget when a single subsection is still too long.
    chunk_size: int = Field(default=700, ge=200)
    chunk_overlap: int = Field(default=100, ge=0)

    # --- Session ------------------------------------------------------------
    session_max_turns: int = Field(default=6, ge=1)
    #: Persists each student's grade/subject across bot restarts (conversation
    #: history is not persisted -- it is cheap to rebuild and not worth it).
    session_db_path: Path = Path("./data/sessions.db")
    log_level: str = "INFO"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_must_fit_in_chunk(cls, value: int, info) -> int:
        """Reject an overlap large enough to make chunking loop forever."""
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @field_validator("chroma_db_path", "session_db_path")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        """Expand ``~`` and resolve the persistence directory to an absolute path."""
        return value.expanduser().resolve()


def parse_model_list(raw: str) -> list[str]:
    """Split a comma-separated setting like ``gemini_fallback_models`` into model ids.

    A plain string field (rather than ``list[str]``) is used for this kind of
    setting because pydantic-settings parses a ``list[str]`` env value as
    JSON, which rejects an ordinary comma-separated ``.env`` value outright.

    Args:
        raw: The raw comma-separated setting value.

    Returns:
        Non-empty, whitespace-trimmed model ids, in order.
    """
    return [model.strip() for model in raw.split(",") if model.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
