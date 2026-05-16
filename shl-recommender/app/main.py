"""
main.py — FastAPI application exposing /health and /chat endpoints.

Usage:
    uvicorn app.main:app --host 0.0.0.0 --port 8000

Environment variables (set in .env or deployment config):
    LLM_PROVIDER    = google | groq | openai   (default: google)
    GOOGLE_API_KEY  = your Gemini API key
    GROQ_API_KEY    = your Groq API key        (if using groq)
    OPENAI_API_KEY  = your OpenAI API key      (if using openai)
"""

import time
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, field_validator

from .agent import run_agent
from .catalog import get_engine

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("shl-recommender")

# ── Lifespan (startup pre-warming) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the catalog and build the FAISS index at startup."""
    logger.info("Loading SHL catalog and building search index...")
    try:
        engine = get_engine()
        logger.info(f"Catalog ready: {len(engine.items)} assessments indexed.")
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Server will start but /chat will fail until catalog is available.")
    yield
    logger.info("Shutting down.")

# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="SHL Assessment Recommender",
    description="Conversational agent for recommending SHL assessments.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request / Response models ─────────────────────────────────────────────────

class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def validate_role(cls, v):
        if v not in ("user", "assistant"):
            raise ValueError("role must be 'user' or 'assistant'")
        return v

    @field_validator("content")
    @classmethod
    def validate_content(cls, v):
        if not v or not v.strip():
            raise ValueError("content must not be empty")
        if len(v) > 8000:
            raise ValueError("content exceeds 8000 character limit")
        return v.strip()


class ChatRequest(BaseModel):
    messages: list[Message]

    @field_validator("messages")
    @classmethod
    def validate_messages(cls, v):
        if not v:
            raise ValueError("messages list must not be empty")
        if len(v) > 20:
            raise ValueError("Too many messages (max 20)")
        # Last message must be from user
        if v[-1].role != "user":
            raise ValueError("Last message must be from user")
        return v


class Recommendation(BaseModel):
    name: str
    url: str
    test_type: str


class ChatResponse(BaseModel):
    reply: str
    recommendations: list[Recommendation]
    end_of_conversation: bool


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    """Readiness check. Returns 200 when service is up."""
    try:
        engine = get_engine()
        catalog_size = len(engine.items)
    except Exception:
        catalog_size = 0
    return {"status": "ok", "catalog_size": catalog_size}


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main conversational endpoint.

    Accepts full stateless conversation history and returns:
    - reply: agent's text response
    - recommendations: list of 0-10 SHL assessments
    - end_of_conversation: whether the agent considers the task done
    """
    t0 = time.time()

    # Convert Pydantic models to plain dicts for the agent
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Turn cap guard (the evaluator caps at 8 turns total)
    # Count only user turns already answered (all but the last user message)
    answered_turns = sum(1 for m in messages[:-1] if m["role"] == "assistant")
    if answered_turns >= 7:
        # Force a recommendation on the last allowed turn
        messages[-1]["content"] = (
            messages[-1]["content"]
            + "\n\n[SYSTEM NOTE: This is the final allowed turn. "
            "You MUST provide a recommendation now even if context is limited. "
            "Use your best judgment.]"
        )

    try:
        result = run_agent(messages)
    except Exception as e:
        logger.exception(f"Agent error: {e}")
        raise HTTPException(status_code=500, detail=f"Agent error: {str(e)}")

    elapsed = time.time() - t0
    logger.info(
        f"Chat turn | {len(messages)} msgs | "
        f"{len(result.get('recommendations', []))} recs | "
        f"{elapsed:.2f}s"
    )

    # Validate response structure (safety net)
    reply = result.get("reply") or "I encountered an issue generating a response."
    recommendations = result.get("recommendations") or []
    eoc = bool(result.get("end_of_conversation", False))

    # Enforce: if recommendations present, mark end_of_conversation True
    # only if LLM said so — don't force it
    validated_recs = []
    for rec in recommendations[:10]:
        validated_recs.append(
            Recommendation(
                name=str(rec.get("name", "")),
                url=str(rec.get("url", "")),
                test_type=str(rec.get("test_type", "")),
            )
        )

    return ChatResponse(
        reply=reply,
        recommendations=validated_recs,
        end_of_conversation=eoc,
    )


# ── Error handlers ────────────────────────────────────────────────────────────

@app.exception_handler(422)
async def validation_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    logger.exception(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please try again."},
    )
