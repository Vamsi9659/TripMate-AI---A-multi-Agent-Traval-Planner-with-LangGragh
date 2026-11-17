import asyncio
import logging
import os
import secrets
from datetime import date
from pathlib import Path
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, Request
from starlette.middleware.sessions import SessionMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from backend import (
    WorkflowConflictError,
    apply_trip_dates_to_request,
    close_database_pool,
    get_checkpointer_status,
    initialize_database,
    resume_travel_agent,
    run_travel_agent,
)
from mcp_client import extract_destination

BASE_DIR = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(initialize_database)
    try:
        yield
    finally:
        await asyncio.to_thread(close_database_pool)

_production = os.getenv("APP_ENV", "development").strip().lower() == "production"
_session_secret = os.getenv("APP_SECRET_KEY")
if not _session_secret:
    if _production:
        raise RuntimeError("APP_SECRET_KEY is required when APP_ENV=production.")
    _session_secret = secrets.token_urlsafe(48)
    logger.warning("APP_SECRET_KEY is not set; using an ephemeral development session secret.")

app = FastAPI(
    title="TripMate AI",
    description=(
        "LangGraph Multi-Agent Travel Planner with Supervisor, Guardrails, "
        "Human-in-the-Loop, and FastAPI Frontend"
    ),
    version="2.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    SessionMiddleware,
    secret_key=_session_secret,
    session_cookie="tripmate_session",
    max_age=60 * 60 * 24 * 7,
    same_site="strict",
    https_only=_production,
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "strict-origin-when-cross-origin")
    response.headers.setdefault("X-Frame-Options", "DENY")
    response.headers.setdefault("Permissions-Policy", "camera=(), microphone=(), geolocation=()")
    response.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; script-src 'self'; style-src 'self'; "
        "img-src 'self' data:; connect-src 'self'; object-src 'none'; "
        "base-uri 'self'; frame-ancestors 'none'",
    )
    return response



def _workflow_error_response(exc: Exception) -> JSONResponse:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status_code = status_code or getattr(response, "status_code", None)
    provider_error_code = getattr(exc, "code", None)
    error_body = getattr(exc, "body", None)
    if provider_error_code is None and isinstance(error_body, dict):
        provider_error = error_body.get("error")
        if isinstance(provider_error, dict):
            provider_error_code = provider_error.get("code")
    error_text = str(exc).casefold()

    if isinstance(exc, WorkflowConflictError):
        return JSONResponse(
            status_code=409,
            content={
                "success": False,
                "error": (
                    "This draft is no longer waiting for approval. Generate a "
                    "new draft before approving or requesting a revision."
                ),
            },
        )

    if status_code == 413 or any(
        marker in error_text
        for marker in ("request too large for model", "tpm limit")
    ):
        return JSONResponse(
            status_code=413,
            content={
                "success": False,
                "error": (
                    "Groq rejected an oversized request. Please shorten the "
                    "travel request and try again."
                ),
            },
        )

    if status_code == 429 or any(
        marker in error_text
        for marker in ("rate limit", "rate_limit_exceeded", "tokens per minute")
    ):
        return JSONResponse(
            status_code=429,
            content={
                "success": False,
                "error": (
                    "Groq is temporarily rate-limited. Please wait briefly "
                    "before trying again."
                ),
            },
        )

    if status_code in {401, 403}:
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": (
                    "The Groq API rejected the configured credentials. "
                    "Check GROQ_API_KEY in the server environment."
                ),
            },
        )

    if status_code == 404 or provider_error_code == "model_not_found":
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": (
                    "The configured Groq model is unavailable. Check GROQ_MODEL "
                    "against the active models available to your Groq account."
                ),
            },
        )

    if status_code == 400:
        return JSONResponse(
            status_code=502,
            content={
                "success": False,
                "error": (
                    "The configured Groq model rejected the request. Check "
                    "GROQ_MODEL and the request size, then try again."
                ),
            },
        )

    if type(exc).__module__.startswith(("psycopg", "psycopg_pool")):
        return JSONResponse(
            status_code=503,
            content={
                "success": False,
                "error": (
                    "PostgreSQL checkpoint storage is temporarily unavailable. "
                    "Please try again shortly."
                ),
            },
        )

    if (
        isinstance(exc, (TimeoutError, asyncio.TimeoutError))
        or "timeout" in type(exc).__name__.casefold()
    ):
        return JSONResponse(
            status_code=504,
            content={
                "success": False,
                "error": "A travel-planning service timed out. Please try again.",
            },
        )

    return JSONResponse(
        status_code=502,
        content={
            "success": False,
            "error": (
                "Travel planning failed because a required service did not "
                "complete successfully. Please try again."
            ),
        },
    )


class TravelRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    thread_id: str | None = Field(default=None, max_length=200)
    departure_date: date | None = None
    return_date: date | None = None


class ApprovalRequest(BaseModel):
    thread_id: str = Field(min_length=1, max_length=200)
    approved: bool
    feedback: str = Field(default="", max_length=2000)


def _owned_thread_ids(request: Request) -> set[str]:
    values = request.session.get("thread_ids", [])
    if not isinstance(values, list):
        return set()
    return {value for value in values if isinstance(value, str)}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.post("/api/travel")
def travel_planner(request_data: TravelRequest, request: Request):
    try:
        if request_data.thread_id and request_data.thread_id not in _owned_thread_ids(request):
            return JSONResponse(
                status_code=403,
                content={"success": False, "error": "This conversation does not belong to this browser session."},
            )

        user_message = request_data.message.strip()

        if not user_message:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Message cannot be empty.",
                },
            )

        try:
            extract_destination(user_message)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": str(exc),
                },
            )

        try:
            user_message = apply_trip_dates_to_request(
                user_message,
                departure_date=request_data.departure_date,
                return_date=request_data.return_date,
            )
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": str(exc),
                },
            )

        result = run_travel_agent(
            user_input=user_message,
            thread_id=request_data.thread_id,
        )

        thread_ids = _owned_thread_ids(request)
        thread_ids.add(result["thread_id"])
        request.session["thread_ids"] = list(thread_ids)[-20:]
        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        logger.error(
            "Travel workflow failed (thread_id=%s, error_type=%s, status=%s).",
            request_data.thread_id,
            type(exc).__name__,
            getattr(exc, "status_code", None),
        )
        return _workflow_error_response(exc)


@app.post("/api/travel/approve")
def approve_travel_plan(request_data: ApprovalRequest, request: Request):
    try:
        if request_data.thread_id not in _owned_thread_ids(request):
            return JSONResponse(
                status_code=403,
                content={"success": False, "error": "This conversation does not belong to this browser session."},
            )

        if not request_data.approved and not request_data.feedback.strip():
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "error": "Please provide revision feedback when rejecting the draft.",
                },
            )

        result = resume_travel_agent(
            thread_id=request_data.thread_id,
            approved=request_data.approved,
            feedback=request_data.feedback,
        )

        return JSONResponse(
            content={
                "success": True,
                **result,
            }
        )

    except Exception as exc:
        logger.error(
            "Travel approval workflow failed "
            "(thread_id=%s, error_type=%s, status=%s).",
            request_data.thread_id,
            type(exc).__name__,
            getattr(exc, "status_code", None),
        )
        return _workflow_error_response(exc)


@app.get("/health")
async def health_check():
    storage = get_checkpointer_status()
    return {
        "status": "ok" if storage["backend"] == "postgresql" else "degraded",
        "message": "TripMate AI API is running",
        "checkpointer": storage,
        "features": [
            "supervisor_agent",
            "input_guardrail",
            "human_in_the_loop",
        ],
    }


@app.get("/favicon.ico")
async def favicon():
    return JSONResponse(content={})


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8000,
        reload=True,
    )
