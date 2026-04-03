from __future__ import annotations

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    neo4j_uri: str = "neo4j://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = SecretStr("changeme")
    neo4j_database: str = "neo4j"
    log_level: str = "INFO"

    # HuggingFace Inference API
    huggingface_api_key: SecretStr | None = None

    # Embedding model
    embedding_model: str = "sentence-transformers/all-mpnet-base-v2"
    embedding_dim: int = 768

    # GAM-RAG constants
    r_pos: float = 0.5
    r_neg: float = 1.0
    q_task: float = 0.05
    q_time: float = 0.05
    pi_init: float = 1.0
    time_decay_rate: float = 0.02  # perplexity increase per day of absence


@lru_cache
def get_settings() -> Settings:
    return Settings()
