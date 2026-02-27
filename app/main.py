"""
FastAPI backend: chat with streaming (SSE), health checks, graph initialization.
Logs are structured (request_id, conversation_id, node, tool, duration); tracer configurable via .env.
"""
import logging
import os
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage, AIMessage
from pydantic import BaseModel, Field
import openai

from app.config import get_settings
from graph.graph import get_graph

# Structured logging: use standard logging with extra dict (JSON-safe; no secrets)
log = logging.getLogger(__name__)
logging.basicConfig(level=getattr(logging, os.getenv("LOG_LEVEL", "INFO")))

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize graph and connections on startup."""
    get_graph()
    yield
    # Teardown if needed


app = FastAPI(title="ChallangeAgentsAi", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=8000)
    conversation_id: str | None = Field(default=None, description="Optional conversation id for context")


class ChatChunk(BaseModel):
    type: str = "token"
    content: str = ""


def _stream_graph(message: str, conversation_id: str | None):
    """Invoke graph and stream the final AI message content token-by-token (or chunked)."""
    graph = get_graph()
    config = {"configurable": {"thread_id": conversation_id or str(uuid.uuid4())}}
    initial = {"messages": [HumanMessage(content=message)]}
    full_content = []
    
    try:
        for event in graph.stream(initial, config=config, stream_mode="values"):
            # event is the state dict when using stream_mode="values"
            if isinstance(event, dict):
                messages = event.get("messages") or []
                for m in messages:
                    if isinstance(m, AIMessage) and m.content:
                        full_content.append(m.content)
    except openai.RateLimitError as e:
        # Friendly error message for quota exceeded
        error_msg = (
            "❌ Desculpe, a quota da API OpenAI foi excedida.\n\n"
            "Por favor, verifique seus créditos em:\n"
            "https://platform.openai.com/settings/organization/billing\n\n"
            "Você pode adicionar créditos ou atualizar a chave da API no arquivo .env"
        )
        log.error("openai_quota_exceeded", extra={"error": str(e)[:200]})
        for char in error_msg:
            yield ChatChunk(type="token", content=char)
        yield ChatChunk(type="done", content="")
        return
    except Exception as e:
        # Generic error handling
        error_msg = f"❌ Erro ao processar sua mensagem: {str(e)[:100]}"
        log.error("graph_stream_error", extra={"error": str(e)[:200]})
        for char in error_msg:
            yield ChatChunk(type="token", content=char)
        yield ChatChunk(type="done", content="")
        return
    
    # Stream the last AI response as chunks (simplified: by char)
    last_content = full_content[-1] if full_content else "I couldn't generate a response."
    for i in range(0, len(last_content), 1):
        yield ChatChunk(type="token", content=last_content[i : i + 1])
    yield ChatChunk(type="done", content="")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    response = await call_next(request)
    response.headers["x-request-id"] = rid
    return response


@app.post("/chat/stream")
async def chat_stream(req: ChatRequest, request: Request):
    """Stream chat response as NDJSON (one JSON object per line)."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    conv_id = req.conversation_id or str(uuid.uuid4())
    start = time.perf_counter()
    log.info("chat_stream_start", extra={"request_id": request_id, "conversation_id": conv_id})

    def gen():
        for chunk in _stream_graph(req.message, conv_id):
            yield chunk.model_dump_json() + "\n"

    return StreamingResponse(
        gen(),
        media_type="application/x-ndjson",
        headers={"x-request-id": request_id, "x-conversation-id": conv_id},
    )


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    """Non-streaming chat: returns full response once done."""
    request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
    conv_id = req.conversation_id or str(uuid.uuid4())
    start = time.perf_counter()
    log.info("chat_start", extra={"request_id": request_id, "conversation_id": conv_id})
    graph = get_graph()
    config = {"configurable": {"thread_id": conv_id}}
    initial = {"messages": [HumanMessage(content=req.message)]}
    final_state = None
    
    try:
        for event in graph.stream(initial, config=config, stream_mode="values"):
            if isinstance(event, dict):
                final_state = event
    except openai.RateLimitError as e:
        # Friendly error message for quota exceeded
        error_msg = (
            "❌ Desculpe, a quota da API OpenAI foi excedida.\n\n"
            "Por favor, verifique seus créditos em:\n"
            "https://platform.openai.com/settings/organization/billing\n\n"
            "Você pode adicionar créditos ou atualizar a chave da API no arquivo .env"
        )
        log.error("openai_quota_exceeded", extra={"error": str(e)[:200]})
        duration = time.perf_counter() - start
        log.info("chat_error", extra={"request_id": request_id, "duration_sec": round(duration, 3)})
        return {"response": error_msg, "conversation_id": conv_id, "error": "rate_limit_exceeded"}
    except Exception as e:
        # Generic error handling
        error_msg = f"❌ Erro ao processar sua mensagem: {str(e)[:100]}"
        log.error("graph_stream_error", extra={"error": str(e)[:200]})
        duration = time.perf_counter() - start
        log.info("chat_error", extra={"request_id": request_id, "duration_sec": round(duration, 3)})
        return {"response": error_msg, "conversation_id": conv_id, "error": "internal_error"}
    
    messages = (final_state or {}).get("messages") or []
    content = ""
    for m in reversed(messages):
        if isinstance(m, AIMessage) and m.content:
            content = m.content
            break
    duration = time.perf_counter() - start
    log.info("chat_done", extra={"request_id": request_id, "duration_sec": round(duration, 3)})
    return {"response": content, "conversation_id": conv_id}


@app.get("/health")
async def health():
    """Basic health check."""
    return {"status": "ok"}


@app.get("/health/ready")
async def health_ready():
    """Readiness: Postgres and Chroma connectivity."""
    settings = get_settings()
    checks = {}
    try:
        from sqlalchemy import create_engine, text
        e = create_engine(settings.postgres_dsn)
        with e.connect() as c:
            c.execute(text("SELECT 1"))
        checks["postgres"] = "ok"
    except Exception as e:
        checks["postgres"] = str(e)[:100]
    try:
        import chromadb
        client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
        client.heartbeat()
        checks["chroma"] = "ok"
    except Exception as e:
        checks["chroma"] = str(e)[:100]
    return {"status": "ok" if all(v == "ok" for v in checks.values()) else "degraded", "checks": checks}
