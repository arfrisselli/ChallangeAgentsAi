"""
Load settings from .env. Never log or expose secret values.
"""
import os
from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI / Azure
    openai_api_key: Optional[str] = None
    azure_openai_api_key: Optional[str] = None
    azure_openai_endpoint: Optional[str] = None
    azure_openai_deployment: Optional[str] = None

    # Web search
    serpapi_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    google_cse_id: Optional[str] = None
    google_api_key: Optional[str] = None

    # Weather
    openweathermap_api_key: Optional[str] = None

    # Postgres
    postgres_host: str = "postgres"
    postgres_port: int = 5432
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_db: str = "challenge_db"

    # Chroma
    chroma_host: str = "chroma"
    chroma_port: int = 8000

    # Tracer
    langchain_tracing_v2: bool = False
    langchain_api_key: Optional[str] = None
    otel_exporter_otlp_endpoint: Optional[str] = None

    # App
    log_level: str = "INFO"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    streamlit_port: int = 8501

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def chroma_http_host(self) -> str:
        return f"http://{self.chroma_host}:{self.chroma_port}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
