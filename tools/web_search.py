"""
Web search tool using Tavily API directly (not LangChain wrapper).
Uses include_answer=True for pre-synthesized answers without needing LLM.
"""
import logging
import os
import re
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field
from tavily import TavilyClient

from tools.base import WebSearchResult

logger = logging.getLogger(__name__)

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
    r"lista de respons[aá]veis|acesso [àa] informa[cç][aã]o|"
    r"ative o javascript|enable javascript|download on the app\s*store|"
    r"google play|app store|escolha a cor|escolha o tamanho|"
    r"c[oó]digo fornecido|iframe|embed\s+code|"
    r"\d{2,4}x\d{2,4}\b|"
    r"voltar ao topo|back to top|leia mais|read more|saiba mais|"
    r"clique aqui|click here|menu principal|main menu|"
    r"logotipo|banner|slider|carousel|"
    r"rel[oó]gio online|hora exata|fuso hor[aá]rio|timezone|"
    r"inscreva.se|subscribe|newsletter|"
    r"compartilh[ae]|share this|tweet this)"
)


def _clean_web_content(text: str, max_chars: int = 1000) -> str:
    """Aggressively strip noise from scraped web pages."""
    if not text:
        return ""
    text = re.sub(r"#{1,6}\s*", "", text)
    text = re.sub(r"^\d+\.\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"\[\.{2,3}\]", "", text)
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\\[nrt]", " ", text)

    lines = text.split("\n")
    kept = []
    seen = set()
    for line in lines:
        stripped = line.strip()
        if len(stripped) < 30:
            continue
        if _NOISE_PATTERNS.search(stripped):
            continue
        if stripped.count("|") > 2:
            continue
        if stripped.count("—") > 2:
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


def _run_web_search(query: str) -> WebSearchResult:
    """
    Search via Tavily API directly. Uses include_answer=True to get
    a pre-synthesized answer that works without LLM.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return WebSearchResult(
            summary="Web search is not configured. Set TAVILY_API_KEY in .env.",
            links=[],
        )

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(
            query=query,
            max_results=5,
            search_depth="advanced",
            include_answer=True,
        )

        tavily_answer = (response.get("answer") or "").strip()
        results_list = response.get("results") or []
        links = [r.get("url", "") for r in results_list if r.get("url")][:5]

        if tavily_answer:
            return WebSearchResult(
                summary=tavily_answer,
                links=links,
                answer=tavily_answer,
            )

        summaries = [
            _clean_web_content(r.get("content", ""))
            for r in results_list
        ]
        summaries = [s for s in summaries if s]

        return WebSearchResult(
            summary="\n".join(summaries) if summaries else "Nenhum resultado encontrado.",
            links=links,
        )

    except Exception as e:
        logger.error("web_search_error: %s", str(e)[:200])
        msg = str(e).lower()
        if "rate" in msg or "limit" in msg:
            return WebSearchResult(
                summary="Limite de requisições atingido. Tente novamente em alguns instantes.",
                links=[],
            )
        return WebSearchResult(
            summary=f"Erro na busca: {str(e)[:100]}. Tente reformular a pergunta.",
            links=[],
        )


@tool(args_schema=WebSearchInput)
def web_search(query: str) -> str:
    """
    Search the web for current information. Use when the user asks about recent events,
    facts not in the knowledge base, or general web search. Returns information with source URLs.
    """
    result = _run_web_search(query)
    if result.answer:
        parts = [result.answer]
    else:
        parts = [result.summary]
    if result.links:
        parts.append("\n\nFontes:")
        for i, link in enumerate(result.links[:3], 1):
            parts.append(f"[{i}] {link}")
    return "\n".join(parts)


def get_web_search_tool():
    """Return the LangChain tool for binding to the agent."""
    return web_search
