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
    zalo_access_token: str | None = None
    zalo_oa_secret: str | None = None
    zalo_webhook_host: str = "0.0.0.0"
    zalo_webhook_port: int = Field(default=8080, ge=1, le=65535)

    # --- LLM generation -----------------------------------------------------
    openai_api_key: str | None = None
    openai_base_url: str | None = None
    llm_model: str = "gpt-4o-mini"
    llm_temperature: float = 0.3
    llm_max_tokens: int = 800
    llm_timeout_seconds: float = 30.0

    # --- Vector store -------------------------------------------------------
    chroma_db_path: Path = Path("./vector_db")
    chroma_collection_name: str = "moet_textbooks"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    # --- Retrieval tuning ---------------------------------------------------
    retrieval_top_k: int = Field(default=5, ge=1, le=20)
    max_retrieval_distance: float = Field(default=1.1, gt=0.0)
    chunk_size: int = Field(default=900, ge=200)
    chunk_overlap: int = Field(default=150, ge=0)

    # --- Session ------------------------------------------------------------
    session_max_turns: int = Field(default=6, ge=1)
    log_level: str = "INFO"

    @field_validator("chunk_overlap")
    @classmethod
    def _overlap_must_fit_in_chunk(cls, value: int, info) -> int:
        """Reject an overlap large enough to make chunking loop forever."""
        chunk_size = info.data.get("chunk_size")
        if chunk_size is not None and value >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return value

    @field_validator("chroma_db_path")
    @classmethod
    def _expand_path(cls, value: Path) -> Path:
        """Expand ``~`` and resolve the persistence directory to an absolute path."""
        return value.expanduser().resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton."""
    return Settings()
