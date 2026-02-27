"""Shared types for tool inputs/outputs. Tools use pydantic for tool-calling mapping."""

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class WebSearchResult:
    """Consolidated web search result: natural language summary + source links."""
    summary: str
    links: list[str]


@dataclass
class VectorSearchResult:
    """Single chunk from vector search."""
    content: str
    metadata: dict[str, Any]


@dataclass
class WeatherResult:
    """Weather API result for natural language response."""
    summary: str
    raw_data: Optional[dict[str, Any]] = None
