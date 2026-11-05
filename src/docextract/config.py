"""Central application configuration loaded from environment variables or `.env`."""

from __future__ import annotations

from decimal import Decimal
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_prefix="DOCEXTRACT_",
        extra="ignore",
    )

    # --- LLM Extraction ---
    llm_model: str = "gemini/gemini-3.5-flash"
    llm_temperature: float = 0.0
    llm_timeout_s: float = 60.0
    max_repair_attempts: int = 2
    consistency_samples: int = 1

    # --- Storage & Ingest Limits ---
    database_url: str = f"sqlite:///{(PROJECT_ROOT / '.data' / 'docextract.db').as_posix()}"
    storage_dir: Path = PROJECT_ROOT / ".data" / "storage"
    max_upload_bytes: int = 20 * 1024 * 1024  # 20 MB
    max_pages: int = 25
    target_dpi: int = 300

    # --- Validation & Confidence Thresholds ---
    min_field_confidence: float = 0.75
    rounding_tolerance: Decimal = Decimal("0.02")

    # --- Outbound Webhooks ---
    webhook_url: str | None = None
    webhook_secret: str | None = None
    webhook_max_retries: int = 3


@lru_cache
def get_settings() -> Settings:
    return Settings()
