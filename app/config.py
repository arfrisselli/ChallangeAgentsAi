"""
Load settings from .env. Never log or expose secret values.
All values come from environment variables (populated via .env file).
"""
import os
from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # OpenAI / Azure
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    azure_openai_api_key: Optional[str] = Field(default=None, description="Azure OpenAI API key")
    azure_openai_endpoint: Optional[str] = Field(default=None, description="Azure OpenAI endpoint")
    azure_openai_deployment: Optional[str] = Field(default=None, description="Azure OpenAI deployment name")

    # Web search
    serpapi_api_key: Optional[str] = Field(default=None, description="SerpAPI key")
    tavily_api_key: Optional[str] = Field(default=None, description="Tavily search key")
    google_cse_id: Optional[str] = Field(default=None, description="Google CSE ID")
    google_api_key: Optional[str] = Field(default=None, description="Google API key")

    # Weather
    openweathermap_api_key: Optional[str] = Field(default=None, description="OpenWeatherMap API key")

    # Postgres -- all values from .env, no hardcoded credentials
    postgres_host: str = Field(description="Postgres host (e.g. 'postgres' in Docker)")
    postgres_port: int = Field(default=5432, description="Postgres port")
    postgres_user: str = Field(description="Postgres username")
    postgres_password: str = Field(description="Postgres password")
    postgres_db: str = Field(description="Postgres database name")

    # Chroma
    chroma_host: str = Field(description="Chroma host (e.g. 'chroma' in Docker)")
    chroma_port: int = Field(default=8000, description="Chroma port")

    # Tracer
    langchain_tracing_v2: bool = Field(default=False, description="Enable LangChain tracing")
    langchain_api_key: Optional[str] = Field(default=None, description="LangSmith API key")
    otel_exporter_otlp_endpoint: Optional[str] = Field(default=None, description="OTEL endpoint")

    # App
    log_level: str = Field(default="INFO", description="Log level")
    api_host: str = Field(default="0.0.0.0", description="FastAPI bind host")
    api_port: int = Field(default=8000, description="FastAPI port")
    streamlit_port: int = Field(default=8501, description="Streamlit port")

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
