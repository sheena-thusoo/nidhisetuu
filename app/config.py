"""Central configuration. Every secret comes from environment variables (or a .env file).

Missing third-party keys never crash the app: the owning component degrades to MOCK MODE
and logs "[MOCK MODE]" so the mock/real distinction is always visible.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- app ---
    app_name: str = "NidhiSetu API"
    app_version: str = "1.0.0"
    env: str = "development"  # development | test | production
    log_level: str = "INFO"
    log_json: bool = True
    cors_origins: str = "*"

    # --- storage ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "nidhisetu"

    # --- auth ---
    jwt_secret: str = "dev-insecure-secret-change-me"
    jwt_algorithm: str = "HS256"
    jwt_expires_minutes: int = 60 * 24
    admin_api_key: str | None = None

    # --- rate limiting (slowapi) ---
    rate_limit_enabled: bool = True
    rate_limit_match: str = "30/minute"
    rate_limit_research: str = "15/minute"
    rate_limit_auth: str = "20/minute"

    # --- agentic layer (Groq / Llama 3.3 70B) ---
    groq_api_key: str | None = None
    groq_model: str = "llama-3.3-70b-versatile"
    groq_temperature: float = 0.0
    tool_timeout_seconds: float = 15.0

    # --- RAG / vector store ---
    pinecone_api_key: str | None = None
    pinecone_index: str = "nidhisetu-schemes"
    pinecone_embedding_model: str = "multilingual-e5-large"
    pinecone_embedding_dim: int = 1024
    mock_embedding_dim: int = 256
    chunk_size: int = 900
    chunk_overlap: int = 150
    rag_top_k: int = 4

    # --- external research (Tavily) ---
    tavily_api_key: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    tavily_max_results: int = 2

    # --- background jobs (Redis / RQ) ---
    redis_url: str | None = None
    rq_queue_name: str = "ingestion"
    force_inline_jobs: bool = False

    # --- freshness / versioning policy ---
    freshness_fresh_days: int = 180
    freshness_aging_days: int = 365
    new_scheme_window_days: int = 7

    # --- data ---
    data_dir: Path = PROJECT_ROOT / "data"

    # ------------------------------------------------------------------ paths
    @property
    def scheme_rules_dir(self) -> Path:
        return self.data_dir / "scheme_rules"

    @property
    def sample_docs_dir(self) -> Path:
        return self.data_dir / "sample_documents"

    @property
    def partners_file(self) -> Path:
        return self.data_dir / "partners_sample.json"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    # ------------------------------------------------------------------ modes
    @property
    def groq_enabled(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def pinecone_enabled(self) -> bool:
        return bool(self.pinecone_api_key)

    @property
    def tavily_enabled(self) -> bool:
        return bool(self.tavily_api_key)

    @property
    def redis_enabled(self) -> bool:
        return bool(self.redis_url) and not self.force_inline_jobs

    def mock_mode_flags(self) -> dict[str, bool]:
        """True == component runs in MOCK MODE (no live credential configured)."""
        return {
            "groq": not self.groq_enabled,
            "pinecone": not self.pinecone_enabled,
            "tavily": not self.tavily_enabled,
            "redis_rq": not self.redis_enabled,
        }


@lru_cache
def get_settings() -> Settings:
    return Settings()
