import asyncio
import json
import logging
import os
import re
import shutil
import sys
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import certifi
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
import pycountry
import requests


# =========================================================
# Environment setup
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Support both environment-variable names.
AVIATION_STACK_API_KEY = (
    os.getenv("AVIATION_STACK_API_KEY")
    or os.getenv("AVIATIONSTACK_API_KEY")
)

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
AVIATIONSTACK_API_URL = "https://api.aviationstack.com/v1/flightsFuture"
AVIATIONSTACK_TIMEOUT_SECONDS = 15
logger = logging.getLogger(__name__)

WEATHER_SERVER_PATH = BASE_DIR / "custom_weather_mcp_server.py"


def _find_uvx_command() -> str:
    command = shutil.which("uvx")
    if command:
        return command

    executable_name = "uvx.exe" if os.name == "nt" else "uvx"
    environment_command = Path(sys.executable).resolve().parent / executable_name
    if environment_command.is_file():
        return str(environment_command)

    return "uvx"


UVX_COMMAND = _find_uvx_command()
MCP_TIMEOUT_SECONDS = 30


def _require_env(name: str, value: str | None) -> str:
    """Return an environment value or raise a readable setup error."""

    if not value:
        raise RuntimeError(
            f"{name} is missing. "
            f"Add {name}=your_key to the project .env file."
        )

    return value


def _subprocess_env(**updates: str | None) -> dict[str, str]:
    """
    Preserve the current Windows/Conda environment and add MCP API keys.
    """

    env = os.environ.copy()

    for key, value in updates.items():
        if value:
            env[key] = value

    return env


# =========================================================
# MCP client
# =========================================================

client = MultiServerMCPClient(
    {
        "tavily": {
            "transport": "streamable_http",
            "url": (
                "https://mcp.tavily.com/mcp/"
                f"?tavilyApiKey={quote_plus(TAVILY_API_KEY or '')}"
            ),
        },

        "aviationstack": {
            "transport": "stdio",
            "command": UVX_COMMAND,
            "args": [
                "--from",
                "aviationstack-mcp==1.8.1",
                "aviationstack-mcp",
            ],
            "env": _subprocess_env(
                AVIATION_STACK_API_KEY=AVIATION_STACK_API_KEY,
                AVIATIONSTACK_API_KEY=AVIATION_STACK_API_KEY,
            ),
        },

        "weather": {
            "transport": "stdio",

            # Uses the Python executable from the active Conda environment.
            "command": sys.executable,

            # Uses the weather server inside the current project folder.
            "args": [
                str(WEATHER_SERVER_PATH),
            ],

            "env": _subprocess_env(
                OPENWEATHER_API_KEY=OPENWEATHER_API_KEY,
            ),
        },
    }
)


async def _get_server_tool(
    server_name: str,
    tool_name: str,
):
    """
    Load one tool from one MCP server.

    This prevents a broken weather or AviationStack server from
    crashing an unrelated Tavily request.
    """

    if server_name == "tavily":
        _require_env(
            "TAVILY_API_KEY",
            TAVILY_API_KEY,
        )

    elif server_name == "aviationstack":
        _require_env(
            "AVIATION_STACK_API_KEY",
            AVIATION_STACK_API_KEY,
        )

        if not Path(UVX_COMMAND).is_file() and shutil.which("uvx") is None:
            raise RuntimeError(
                "uvx was not found. Install uv, reopen the terminal, "
                "activate the travel environment, and run "
                "`uvx --version`."
            )

    elif server_name == "weather":
        _require_env(
            "OPENWEATHER_API_KEY",
            OPENWEATHER_API_KEY,
        )

        if not WEATHER_SERVER_PATH.is_file():
            raise FileNotFoundError(
                f"Weather MCP server not found: "
                f"{WEATHER_SERVER_PATH}"
            )

    # Important: load only the requested MCP server.
    tools = await asyncio.wait_for(
        client.get_tools(server_name=server_name),
        timeout=MCP_TIMEOUT_SECONDS,
    )

    tool = next(
        (
            item
            for item in tools
            if item.name == tool_name
        ),
        None,
    )

    if tool is None:
        available_tools = (
            ", ".join(
                sorted(item.name for item in tools)
            )
            or "none"
        )

        raise RuntimeError(
            f"MCP tool '{tool_name}' was not found "
            f"on server '{server_name}'. "
            f"Available tools: {available_tools}"
        )

    return tool


# =========================================================
# MCP connection test
# =========================================================

async def get_all_tools() -> None:
    """
    Test every MCP server independently.

    One failed server will not stop the remaining tests.
    """

    for server_name in (
        "tavily",
        "aviationstack",
        "weather",
    ):
        try:
            tools = await asyncio.wait_for(
                client.get_tools(server_name=server_name),
                timeout=MCP_TIMEOUT_SECONDS,
            )

            tool_names = (
                ", ".join(
                    tool.name
                    for tool in tools
                )
                or "no tools"
            )

            print(
                f"{server_name}: OK -> {tool_names}"
            )

        except Exception as exc:
            print(
                f"{server_name}: FAILED -> "
                f"{type(exc).__name__}"
            )


# =========================================================
# Tavily MCP
# =========================================================

async def tavily_mcp_search(query: str):
    search_tool = await _get_server_tool(
        "tavily",
        "tavily_search",
    )

    return await asyncio.wait_for(
        search_tool.ainvoke({"query": query}),
        timeout=MCP_TIMEOUT_SECONDS,
    )


# =========================================================
# AviationStack MCP
# =========================================================

async def aviation_mcp_call(
    tool_name: str,
    tool_args: dict[str, Any] | None = None,
):
    if tool_name == "future_flights_arrival_departure_schedule":
        return await _aviationstack_future_schedule(tool_args or {})

    aviation_tool = await _get_server_tool(
        "aviationstack",
        tool_name,
    )

    return await asyncio.wait_for(
        aviation_tool.ainvoke(tool_args or {}),
        timeout=MCP_TIMEOUT_SECONDS,
    )


async def _aviationstack_future_schedule(
    tool_args: dict[str, Any],
) -> dict[str, Any]:
    api_key = _require_env("AVIATION_STACK_API_KEY", AVIATION_STACK_API_KEY)
    airport_iata = str(tool_args.get("airport_iata_code") or "").strip().upper()
    schedule_type = str(tool_args.get("schedule_type") or "").strip().lower()
    requested_date = str(tool_args.get("date") or "").strip()
    airline_iata = str(tool_args.get("airline_iata") or "").strip().upper()
    flight_limit = tool_args.get("number_of_flights", 5)

    if not re.fullmatch(r"[A-Z]{3}", airport_iata):
        raise ValueError("AviationStack requires a valid three-letter IATA airport code.")
    if schedule_type not in {"arrival", "departure"}:
        raise ValueError("AviationStack schedule_type must be arrival or departure.")
    if isinstance(flight_limit, bool) or not isinstance(flight_limit, int) or not 1 <= flight_limit <= 100:
        raise ValueError("AviationStack number_of_flights must be between 1 and 100.")
    try:
        schedule_date = date.fromisoformat(requested_date)
    except ValueError:
        raise ValueError("AviationStack schedule date must use YYYY-MM-DD.") from None
    today = date.today()
    if not today < schedule_date <= today + timedelta(days=366):
        raise ValueError(
            "AviationStack future schedules support dates from tomorrow through about 12 months ahead."
        )

    params: dict[str, Any] = {
        "access_key": api_key,
        "iataCode": airport_iata,
        "type": schedule_type,
        "date": schedule_date.isoformat(),
        "limit": flight_limit,
    }
    if airline_iata:
        params["airline_iata"] = airline_iata

    try:
        response = await asyncio.to_thread(
            requests.get,
            AVIATIONSTACK_API_URL,
            params=params,
            timeout=AVIATIONSTACK_TIMEOUT_SECONDS,
        )
    except requests.RequestException as exc:
        logger.warning(
            "AviationStack request transport failed (error_type=%s, airport=%s, date=%s).",
            type(exc).__name__,
            airport_iata,
            schedule_date.isoformat(),
        )
        return {
            "ok": False,
            "error": "AviationStack request could not reach the provider.",
        }

    status_code = response.status_code
    try:
        payload = response.json()
    except ValueError:
        logger.warning(
            "AviationStack returned invalid JSON (http_status=%s, airport=%s, date=%s).",
            status_code,
            airport_iata,
            schedule_date.isoformat(),
        )
        return {
            "ok": False,
            "status_code": status_code,
            "error": f"AviationStack returned invalid JSON (HTTP {status_code}).",
        }

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError:
            payload = {"error": payload}

    if not isinstance(payload, dict):
        logger.warning(
            "AviationStack returned an unexpected JSON shape (http_status=%s, airport=%s, date=%s).",
            status_code,
            airport_iata,
            schedule_date.isoformat(),
        )
        return {
            "ok": False,
            "status_code": status_code,
            "error": "AviationStack returned an unexpected response structure.",
        }

    provider_error = payload.get("error")
    if status_code >= 400 or provider_error:
        if isinstance(provider_error, dict):
            provider_error = (
                provider_error.get("message")
                or provider_error.get("code")
                or "provider API error"
            )
        error_text = re.sub(r"\s+", " ", str(provider_error or "HTTP error"))[:240]
        logger.warning(
            "AviationStack API error (http_status=%s, airport=%s, date=%s, detail=%s).",
            status_code,
            airport_iata,
            schedule_date.isoformat(),
            error_text,
        )
        return {
            "ok": False,
            "status_code": status_code,
            "error": f"HTTP {status_code}: {error_text}",
        }

    rows = payload.get("data")
    if rows is None:
        rows = []
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        logger.warning(
            "AviationStack response data was not a list of records (http_status=%s, airport=%s, date=%s).",
            status_code,
            airport_iata,
            schedule_date.isoformat(),
        )
        return {
            "ok": False,
            "status_code": status_code,
            "error": "AviationStack returned an unexpected response structure.",
        }

    logger.info(
        "AviationStack schedule request completed (http_status=%s, airport=%s, date=%s, records=%s).",
        status_code,
        airport_iata,
        schedule_date.isoformat(),
        len(rows),
    )
    return {"ok": True, "data": rows}


# =========================================================
# Weather MCP
# =========================================================

async def weather_mcp_search(city: str):
    weather_tool = await _get_server_tool(
        "weather",
        "get_current_weather",
    )

    return await asyncio.wait_for(
        weather_tool.ainvoke({"city": city}),
        timeout=MCP_TIMEOUT_SECONDS,
    )


async def forecast_mcp_search(city: str):
    forecast_tool = await _get_server_tool(
        "weather",
        "get_forecast",
    )

    return await asyncio.wait_for(
        forecast_tool.ainvoke({"city": city}),
        timeout=MCP_TIMEOUT_SECONDS,
    )


# =========================================================
# Destination extractor
# =========================================================

_DESTINATION_STOP = re.compile(
    r"(?:,\s*|\s+)\b(?:from|for|with|including|include|under|within|"
    r"budget|maximum(?:\s+budget)?|starting|departing|leaving|returning|"
    r"on|check[\s-]?in|check[\s-]?out|departure(?:\s+date)?|"
    r"travel(?:\s+date)?|then|lasting|over)\b|[?!;\n]|(?<!\bSt)\.(?=\s|$)",
    re.IGNORECASE,
)
_DESTINATION_ACTION_PREFIX = re.compile(
    r"^(?:(?:please\s+)?(?:plan|create|organize|book|arrange|suggest|"
    r"take|do|visit|explore|travel|go|fly|head)\s+)+",
    re.IGNORECASE,
)
_DESTINATION_FILLER_PREFIX = re.compile(
    r"^(?:(?:a|an|the|complete|full|relaxing|budget|family|solo|"
    r"short|long|international)\s+)+",
    re.IGNORECASE,
)
_DURATION_PREFIX = re.compile(
    r"^\d+\s*[- ]?\s*(?:days?|nights?|weeks?)\s+",
    re.IGNORECASE,
)


def _is_country_name(value: str) -> bool:
    if value.casefold().replace(".", "") == "uae":
        return True

    try:
        pycountry.countries.lookup(value)
    except LookupError:
        return False
    return True


def _clean_destination(candidate: str) -> str:
    stop = _DESTINATION_STOP.search(candidate)
    if stop:
        candidate = candidate[: stop.start()]

    candidate = candidate.strip(" \t\r\n\"'`.,!?;:")
    candidate = " ".join(candidate.split())
    candidate = _DESTINATION_ACTION_PREFIX.sub("", candidate).strip()
    candidate = _DESTINATION_FILLER_PREFIX.sub("", candidate).strip()
    candidate = _DURATION_PREFIX.sub("", candidate).strip()
    candidate = candidate.strip(" \t\r\n\"'`.,!?;:")

    if candidate.casefold() in {"", "a", "an", "the", "complete", "full"}:
        return ""

    if re.search(r"\s+(?:or|/)\s+", candidate, re.IGNORECASE):
        raise ValueError(
            "The destination is ambiguous. Please choose one destination."
        )

    parts = [part.strip() for part in candidate.split(",")]
    if len(parts) > 1 and _is_country_name(parts[-1]):
        candidate = ", ".join(parts[:-1]).strip()

    return candidate


def extract_destination(query: str) -> str:
    """Extract an explicitly named destination without using an LLM."""
    if not isinstance(query, str) or not query.strip():
        raise ValueError(
            "The destination could not be extracted. Please name a destination."
        )

    explicit_patterns = (
        r"\b(?:trip|vacation|holiday)\s+(?:to|in)\s+(.+)",
        r"\b(?:stay(?:ing)?|spend(?:ing)?)\b.*?\b(?:in|at)\s+(.+)",
        r"\b(?:destination(?:\s+is)?|to|visit(?:ing)?|explore|"
        r"travel(?:ing)?\s+to|go\s+to|fly\s+to|head\s+to)\s+(.+)",
    )

    for pattern in explicit_patterns:
        for match in re.finditer(pattern, query, re.IGNORECASE):
            candidate = match.group(1).strip()
            if re.match(
                r"(?:plan|book)\b",
                candidate,
                re.IGNORECASE,
            ):
                continue

            candidate = _DESTINATION_ACTION_PREFIX.sub("", candidate).strip()
            candidate = re.sub(r"^(?:to\s+)+", "", candidate, flags=re.IGNORECASE)
            destination = _clean_destination(candidate)
            if destination:
                return destination

    duration_trip = re.search(
        r"\b\d+\s*[- ]?\s*(?:days?|nights?|weeks?)\s+(.+?)\s+"
        r"(?:trip|vacation|holiday)\b",
        query,
        re.IGNORECASE,
    )
    if duration_trip:
        destination = _clean_destination(duration_trip.group(1))
        if destination:
            return destination

    trip_matches = list(
        re.finditer(
            r"\b([A-Za-z][A-Za-z0-9.'’ ,/\-]*?)\s+"
            r"(?:trip|vacation|holiday)\b",
            query,
            re.IGNORECASE,
        )
    )
    for match in trip_matches:
        candidate = _DESTINATION_ACTION_PREFIX.sub("", match.group(1)).strip()
        candidate = _DESTINATION_FILLER_PREFIX.sub("", candidate).strip()
        candidate = _DURATION_PREFIX.sub("", candidate).strip()
        destination = _clean_destination(candidate)
        if destination:
            return destination

    raise ValueError(
        "The destination could not be extracted. Please name one destination, "
        "for example, 'Plan a trip to Paris'."
    )