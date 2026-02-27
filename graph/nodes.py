"""
LangGraph nodes: UserNode, PlannerNode, ExecutorNode, MemoryNode, FallbackSearchNode, and ConversationNode.
Each node has a single responsibility; fallback route sends "no context" questions to web search.
"""
import os
import time
from typing import Any, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

from graph.state import GraphState
from graph.persona import detect_conversation_type, get_conversational_response

# Structlog if available, else standard logging (no secrets)
try:
    import structlog
    log = structlog.get_logger()
except Exception:
    import logging
    log = logging.getLogger(__name__)


def _get_llm():
    """Chat model from env (OpenAI or Azure)."""
    if os.getenv("AZURE_OPENAI_API_KEY"):
        from langchain_openai import AzureChatOpenAI
        return AzureChatOpenAI(
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            deployment_name=os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4"),
            api_version="2024-02-15-preview",
            temperature=0,
        )
    return ChatOpenAI(
        model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
        api_key=os.getenv("OPENAI_API_KEY"),
        temperature=0,
    )


def user_node(state: GraphState) -> dict[str, Any]:
    """
    UserNode: Normalizes and forwards user input into state. Called at graph entry.
    Does not call any tools; only prepares state for Planner.
    """
    return state


def planner_node(state: GraphState) -> dict[str, Any]:
    """
    PlannerNode: Decides routing - conversation, weather, web fallback, or tool execution.
    Called after UserNode. Routes based on query type.
    """
    messages = state.get("messages") or []
    last = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    if not last or not getattr(last, "content", None):
        return {"need_web_fallback": True, "is_conversational": False, "is_weather_query": False}
    query = (last.content or "").strip()[:500]
    
    # 1. Check if conversational/social query (name, greetings, etc.)
    conversation_type = detect_conversation_type(query)
    if conversation_type:
        log.info("planner_node", is_conversational=True, conversation_type=conversation_type, duration_sec=0)
        return {"need_web_fallback": False, "is_conversational": True, "is_weather_query": False}
    
    # 2. Fast keyword-based routing for weather queries (NO LLM call!)
    weather_keywords = [
        "clima", "weather", "temperatura", "temperature", "tempo", "forecast",
        "previsão", "chuva", "rain", "frio", "quente", "hot", "cold", "sol", "sun",
        "nublado", "cloudy", "vento", "wind", "°c", "celsius", "fahrenheit"
    ]
    query_lower = query.lower()
    if any(keyword in query_lower for keyword in weather_keywords):
        log.info("planner_node", is_weather_query=True, duration_sec=0, reason="weather_keyword_match")
        return {"need_web_fallback": False, "is_conversational": False, "is_weather_query": True}
    
    # 3. LLM-based routing for other queries
    start = time.perf_counter()
    llm = _get_llm()
    
    prompt = (
        "You are a router. Given the user question below, answer with exactly one word:\n\n"
        "EXECUTE: for questions that need specialized tools:\n"
        "  * Internal docs/FAQs → use search_docs tool\n"
        "  * Database queries/SQL → use sql_db tool\n"
        "  Examples: 'What products are in database?', 'Search internal docs'\n\n"
        "WEB_FALLBACK: ONLY for general knowledge, current events, news\n"
        "  Examples: 'Who won the election?', 'Latest AI news'\n\n"
        "User question: " + query
    )
    try:
        out = llm.invoke([HumanMessage(content=prompt)])
        text = (out.content or "").strip().upper()
        need_fallback = "WEB_FALLBACK" in text or "FALLBACK" in text
    except Exception:
        need_fallback = True
    duration = time.perf_counter() - start
    log.info("planner_node", need_web_fallback=need_fallback, duration_sec=round(duration, 3))
    return {"need_web_fallback": need_fallback, "is_conversational": False, "is_weather_query": False}


def fallback_search_node(state: GraphState) -> dict[str, Any]:
    """
    FallbackSearchNode: Web search with Tavily answer (pre-synthesized) as primary path.
    Falls back to LLM synthesis, then to honest message if both fail.
    """
    from tools.web_search import _run_web_search
    messages = state.get("messages") or []
    last = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    query = (getattr(last, "content", None) or "").strip() if last else ""
    if not query:
        return {"messages": [AIMessage(content="Não recebi uma pergunta. Por favor, pergunte algo.")]}

    start = time.perf_counter()
    result = _run_web_search(query)
    duration = time.perf_counter() - start
    log.info("fallback_search_node", tool="web_search", duration_sec=round(duration, 3),
             has_tavily_answer=bool(result.answer))

    context = result.answer if result.answer else result.summary[:1000]

    synthesis_prompt = f"""Com base nas informações abaixo, forneça uma resposta concisa e direta à pergunta do usuário em 2-4 frases.

REGRA OBRIGATÓRIA: Responda SEMPRE em português brasileiro (PT-BR), independentemente do idioma da pergunta ou das informações fornecidas. Traduza tudo para PT-BR.

Pergunta do usuário: {query}

Informações encontradas:
{context}

Instruções:
- Sintetize apenas as informações CHAVE
- Seja conciso e natural (2-4 frases no máximo)
- RESPONDA SEMPRE EM PORTUGUÊS BRASILEIRO, traduzindo se necessário
- Cite fontes como [1], [2] se usar múltiplas

Resposta (em PT-BR):"""

    try:
        llm = _get_llm()
        synthesis = llm.invoke([HumanMessage(content=synthesis_prompt)])
        answer = synthesis.content or "Não foi possível sintetizar os resultados."
    except Exception as e:
        log.error("fallback_search_synthesis_error", error=str(e), query=query[:100])
        if result.answer:
            answer = result.answer
        else:
            answer = ("Desculpe, não consegui gerar uma resposta sintetizada no momento. "
                      "Tente novamente ou reformule sua pergunta.")

    if result.links and "[" not in answer:
        answer += "\n\nFontes:\n" + "\n".join(
            f"[{i+1}] {link}" for i, link in enumerate(result.links[:3]))

    return {"messages": [AIMessage(content=answer)]}


def _build_tools():
    """Build list of LangChain tools for Executor: web_search, search_docs, sql_db, weather_api."""
    from tools.web_search import get_web_search_tool
    from tools.vector_search import get_search_docs_tool
    from tools.weather_api import get_weather_tool
    from app.config import get_settings
    from tools.sql_db import get_sql_db_tool
    settings = get_settings()
    tools = [
        get_web_search_tool(),
        get_search_docs_tool(),
        get_weather_tool(),
        get_sql_db_tool(settings.postgres_dsn),
    ]
    return tools


def executor_node(state: GraphState) -> dict[str, Any]:
    """
    ExecutorNode: Runs the agent with all tools (web_search, search_docs, sql_db, weather_api).
    Invoked when Planner chose EXECUTE. Uses ReAct/tool-calling to produce final answer.
    """
    from langchain_core.messages import SystemMessage
    
    tools = _build_tools()
    llm = _get_llm()
    
    # System prompt for concise, synthesized responses (Perplexity-style) with language matching
    system_prompt = """Você é um assistente de IA útil que fornece respostas concisas e precisas.

REGRA OBRIGATÓRIA: Responda SEMPRE em português brasileiro (PT-BR), independentemente do idioma da pergunta.

Ao responder perguntas:
1. Seja direto e conciso - sintetize as informações, não repita os outputs das ferramentas
2. Responda em 2-4 frases no máximo, a menos que mais detalhes sejam solicitados
3. Se usar múltiplas fontes, cite com [1], [2] etc. ao final
4. Formato: Resposta primeiro, depois lista de fontes se aplicável

Exemplos:
- Clima: "Em Londres, está 15°C e nublado com chuva leve esperada."
- Banco de dados: "Há 3 produtos no banco: Widget A (R$ 10,50), Widget B (R$ 25,00) e Gadget X (R$ 99,99)."
- Web: "Node.js é um runtime JavaScript assíncrono baseado no V8 [1]. É ideal para APIs e aplicações em tempo real [2]."

Mantenha as respostas naturais, úteis e objetivas, SEMPRE em PT-BR."""

    agent = create_react_agent(llm, tools)
    
    # Prepend system message to messages
    messages = list(state.get("messages") or [])
    messages_with_system = [SystemMessage(content=system_prompt)] + messages
    
    start = time.perf_counter()
    result = agent.invoke({"messages": messages_with_system})
    duration = time.perf_counter() - start
    log.info("executor_node", duration_sec=round(duration, 3))
    new_messages = result.get("messages", [])
    
    # Return only NEW messages (excluding system + original messages)
    num_original = len(messages_with_system)
    if len(new_messages) > num_original:
        return {"messages": new_messages[num_original:]}
    
    # If no new messages, return the last AI response
    return {"messages": [m for m in new_messages if isinstance(m, AIMessage)][-1:] if new_messages else []}


def conversation_node(state: GraphState) -> dict[str, Any]:
    """
    ConversationNode: Handles social/conversational queries with consistent identity.
    Returns predefined responses for greetings, identity questions, etc.
    """
    messages = state.get("messages") or []
    last = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    query = (getattr(last, "content", None) or "").strip() if last else ""
    
    start = time.perf_counter()
    conversation_type = detect_conversation_type(query)
    response = get_conversational_response(conversation_type, query)
    duration = time.perf_counter() - start
    
    log.info("conversation_node", conversation_type=conversation_type, duration_sec=round(duration, 3))
    return {"messages": [AIMessage(content=response)]}


_WEATHER_DESC_PT = {
    "clear sky": "Céu limpo",
    "few clouds": "Poucas nuvens",
    "scattered clouds": "Nuvens dispersas",
    "broken clouds": "Nublado parcial",
    "overcast clouds": "Nublado",
    "shower rain": "Chuva rápida",
    "rain": "Chuva",
    "light rain": "Chuva leve",
    "moderate rain": "Chuva moderada",
    "heavy intensity rain": "Chuva forte",
    "thunderstorm": "Tempestade",
    "snow": "Neve",
    "light snow": "Neve leve",
    "mist": "Névoa",
    "haze": "Neblina",
    "fog": "Nevoeiro",
    "drizzle": "Garoa",
    "light intensity drizzle": "Garoa leve",
    "smoke": "Fumaça",
    "dust": "Poeira",
    "sand": "Areia",
    "tornado": "Tornado",
    "squall": "Ventania",
}


def _translate_weather_desc(desc: str) -> str:
    """Translate OpenWeatherMap description to PT-BR."""
    lower = desc.lower().strip()
    if lower in _WEATHER_DESC_PT:
        return _WEATHER_DESC_PT[lower]
    for key, val in _WEATHER_DESC_PT.items():
        if key in lower:
            return val
    return desc.title()


def _extract_city(query: str) -> tuple[Optional[str], Optional[str]]:
    """Extract city (and optional country) from weather query. Works on original casing."""
    import re
    q = query.strip()

    # Pattern: preposition + city name (may include /, , for country)
    # e.g. "clima em Londrina/PR", "weather in Paris, France", "tempo de São Paulo"
    match = re.search(
        r'\b(?:em|in|de|for|para)\s+'
        r'([A-ZÀ-ÚÑ][a-záàâãéèêíïóôõöúçñ]+(?:[\s\-]+[A-ZÀ-Úa-záàâãéèêíïóôõöúçñ]+)*'
        r'(?:[/,]\s*[A-Za-zÀ-ÚÑà-úñ]+)?)',
        q
    )
    if match:
        raw = match.group(1).strip().rstrip('?!.')
        if '/' in raw:
            parts = raw.split('/', 1)
            return parts[0].strip(), parts[1].strip()
        if ',' in raw:
            parts = raw.split(',', 1)
            return parts[0].strip(), parts[1].strip()
        return raw, None

    # Fallback: find capitalized proper nouns (skip common PT/EN/weather words)
    skip = {
        'Como', 'Qual', 'What', 'How', 'The', 'Uma', 'Está', 'Sera', 'Vai',
        'Will', 'Does', 'Can', 'Que', 'Por', 'Não', 'Para', 'Hoje', 'Amanhã',
        'Clima', 'Weather', 'Tempo', 'Temperature', 'Temperatura', 'Previsão',
    }
    candidates = []
    for w in q.split():
        clean = re.sub(r'[?.!,/]', '', w)
        if clean and clean[0].isupper() and clean not in skip and len(clean) > 2:
            candidates.append(clean)
    if candidates:
        return ' '.join(candidates), None

    return None, None


def weather_node(state: GraphState) -> dict[str, Any]:
    """
    WeatherNode: Extracts city from question and calls weather API directly (NO LLM!).
    Fast, cheap, and works even without OpenAI credits.
    """
    from tools.weather_api import get_weather_impl

    messages = state.get("messages") or []
    last = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
    query = (getattr(last, "content", None) or "").strip() if last else ""

    start = time.perf_counter()

    city, country = _extract_city(query)

    if not city:
        error_msg = "Desculpe, não consegui identificar a cidade. Por favor, especifique a cidade (ex: 'clima em São Paulo')."
        duration = time.perf_counter() - start
        log.info("weather_node", city=None, duration_sec=round(duration, 3), error="no_city_found")
        return {"messages": [AIMessage(content=error_msg)]}

    result = get_weather_impl(city, country)

    if result.raw_data:
        main = result.raw_data.get("main", {})
        daily = result.raw_data.get("daily", {})
        temp = main.get("temp", "N/A")
        temp_min = daily.get("temp_min")
        temp_max = daily.get("temp_max")
        desc_raw = (result.raw_data.get("weather") or [{}])[0].get("description", "")
        desc = _translate_weather_desc(desc_raw)
        feels_like = main.get("feels_like")
        humidity = main.get("humidity")
        wind_speed = result.raw_data.get("wind", {}).get("speed")
        city_name = result.raw_data.get("name", city)

        response = f"🌤️ Em {city_name}: {desc}, {temp}°C"
        if temp_min is not None and temp_max is not None:
            response += f" (mín {temp_min}°C / máx {temp_max}°C)"
        if feels_like is not None:
            response += f". Sensação térmica: {feels_like}°C"
        if humidity is not None:
            response += f". Umidade: {humidity}%"
        if wind_speed is not None:
            response += f". Vento: {wind_speed} m/s"

        response += "."
    else:
        response = result.summary

    duration = time.perf_counter() - start
    log.info("weather_node", city=city, country=country, duration_sec=round(duration, 3))
    return {"messages": [AIMessage(content=response)]}


def memory_node(state: GraphState) -> dict[str, Any]:
    """
    MemoryNode: Updates conversation context (e.g. keeps last N messages for context).
    Called after Executor or Fallback to persist turn. Can be extended for summarization.
    """
    messages = state.get("messages") or []
    # Keep last 20 messages as context for future turns
    context_messages = messages[-20:] if len(messages) > 20 else messages
    return {"context": str(len(context_messages)) + " messages in context"}
