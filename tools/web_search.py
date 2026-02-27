"""
Web search tool: SerpAPI, Tavily, or Google CSE (LangChain wrapper).
Invoked by the model when the user asks about current/factual web information.
Returns consolidated natural language text + list of source links.
"""
import os
import re
from typing import Optional

from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from pydantic import BaseModel, Field

from tools.base import WebSearchResult

_NOISE_PATTERNS = re.compile(
    r"(?i)"
    r"(cnpj|cep\s*:?\s*\d|telefone\s*:?\s*\(|fone\s*:?\s*\(|whatsapp|"
    r"rodap[eé]|cookie|acessibilidade|pol[ií]tica de privacidade|"
    r"termos? de uso|fale conosco|ouvidoria|copyright|©|"
    r"facebook\s+twitter|linkedin\s+twitter|instagram\s+facebook|"
    r"icone marca|favicon|\.png\b|\.jpg\b|\.svg\b|"
    r"pular para o conte[uú]do|skip to content|"
    r"todos os direitos|all rights reserved|"
    r"portal da prefeitura|menu\s+oculto|gabinete do prefeito|"
    r"n[uú]cleo de comunica[cç][aã]o|hor[aá]rios?\s+de\s+funcionamento|"
    r"lista de respons[aá]veis|acesso [àa] informa[cç][aã]o)"
)


def _clean_web_content(text: str, max_chars: int = 1500) -> str:
    """Strip noise from scraped web pages: menus, footers, contacts, markdown headers, etc."""
    if not text:
        return ""
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)

    lines = text.split("\n")
    kept = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 25:
            continue
        if _NOISE_PATTERNS.search(stripped):
            continue
        if stripped.count("|") > 2:
            continue
        norm = stripped.lower()[:60]
        if norm in seen:
            continue
        seen.add(norm)
        kept.append(stripped)

    clean = " ".join(kept)
    clean = re.sub(r"\s{2,}", " ", clean)
    clean = re.sub(r"(\b\w[\w\s]{5,50})\1+", r"\1", clean)
    return clean[:max_chars].strip()


class WebSearchInput(BaseModel):
    """Input for web search. Query is the user's search question."""
    query: str = Field(description="Search query or question to look up on the web")


def _get_tavily_tool():
    """Build Tavily tool if API key is set."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return None
    return TavilySearchResults(
        max_results=5,
        search_depth="advanced",
        include_answer=True,
        api_key=api_key,
    )


def _run_web_search(query: str) -> WebSearchResult:
    t = _get_tavily_tool()
    if not t:
        return WebSearchResult(
            summary="Web search is not configured. Set TAVILY_API_KEY in .env.",
            links=[],
        )
    try:
        results = t.invoke({"query": query})
        if isinstance(results, list):
            summaries = [
                _clean_web_content(r.get("content", r.get("snippet", str(r))))
                for r in results
            ]
            summaries = [s for s in summaries if s]
            links = [r.get("url", "") for r in results if r.get("url")]
            return WebSearchResult(
                summary="\n".join(summaries) if summaries else "No results found.",
                links=links[:10],
            )
        if isinstance(results, dict) and "answer" in results:
            res_list = results.get("results") or []
            links = [r.get("url", "") for r in res_list if isinstance(r, dict) and r.get("url")][:10]
            return WebSearchResult(summary=results.get("answer", ""), links=links)
        return WebSearchResult(summary=str(results), links=[])
    except Exception as e:
        msg = str(e).lower()
        if "rate" in msg or "limit" in msg:
            return WebSearchResult(
                summary="Rate limit reached for web search. Please try again later.",
                links=[],
            )
        return WebSearchResult(
            summary=f"Search failed: {e}. Please try again or rephrase.",
            links=[],
        )


@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Search the web for current information. Use when the user asks about recent events,
    facts not in the knowledge base, or general web search. Returns information with source URLs.
    """
    result = _run_web_search(query)
    parts = [result.summary]
    if result.links:
        parts.append("\n\n**Sources:**")
        for i, link in enumerate(result.links[:3], 1):
            parts.append(f"[{i}] {link}")
    return "\n".join(parts)


def get_web_search_tool():
    """Return the LangChain tool for binding to the agent."""
    return web_search
