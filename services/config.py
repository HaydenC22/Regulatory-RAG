from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgresql://postgres:postgres@localhost:5432/regulatory_rag"

    llm_provider: str = "gemini"  # "gemini" (free tier, default — see ADR-0004) or "anthropic"

    gemini_api_key: str = ""
    anthropic_api_key: str = ""
    llm_model_cheap: str = "gemini-flash-lite-latest"
    llm_model_quality: str = "gemini-flash-lite-latest"

    langsmith_api_key: str = ""
    langsmith_project: str = "regulatory-rag-p2"
    langsmith_tracing: bool = False

    embedding_provider: str = "local"
    embedding_model: str = "BAAI/bge-base-en-v1.5"
    embedding_dim: int = 768

    rerank_threshold: float = 0.0
    retrieval_top_k: int = 5
    rate_limit_per_minute: int = 20

    corpus_manifest_path: str = "corpus/manifest.yaml"
    corpus_raw_dir: str = "corpus/raw"
    corpus_processed_dir: str = "corpus/processed"


@lru_cache
def get_settings() -> Settings:
    return Settings()
