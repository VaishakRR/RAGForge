from __future__ import annotations

from pathlib import Path

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration, loaded from environment variables or .env."""

    model_config = SettingsConfigDict(
        env_prefix="RAGFORGE_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    ollama_base_url: str = Field(
        default="",
        validation_alias=AliasChoices("OLLAMA_BASE_URL", "RAGFORGE_OLLAMA_BASE_URL"),
        description="Ollama endpoint. Empty means auto-detect.",
    )
    chat_model: str = "qwen2.5-coder:3b-instruct-q8_0"
    embed_model: str = "nomic-embed-text"
    allow_fallback_embeddings: bool = True

    data_dir: Path = Path("data")
    max_upload_mb: int = 50

    chunk_size: int = 900
    chunk_overlap: int = 150
    min_chunk_chars: int = 40

    top_k: int = 6
    dense_k: int = 12
    bm25_k: int = 12
    mmr_lambda: float = 0.8

    temperature: float = 0.2
    max_context_chars: int = 12000
    max_history_messages: int = 6
    num_ctx: int = 8192

    request_timeout: float = 180.0

    @property
    def database_path(self) -> Path:
        return self.data_dir / "ragforge.db"

    @property
    def uploads_dir(self) -> Path:
        return self.data_dir / "uploads"

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)


settings = Settings()
