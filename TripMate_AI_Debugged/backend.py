import os
import certifi
import logging
import math
import re
from datetime import date, datetime, timedelta
from pathlib import Path
from functools import lru_cache
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().with_name(".env"))
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

from typing import Any, TypedDict, Annotated
import operator
import uuid
import asyncio
import json
import airportsdata
import pycountry
import requests
from urllib.parse import urlsplit
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.types import Command, interrupt
from langchain_core.messages import (
    AnyMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from langchain_groq import ChatGroq


from mcp_client import (
    tavily_mcp_search,
    extract_destination,
    weather_mcp_search,
    forecast_mcp_search,
    aviation_mcp_call,
)

logger = logging.getLogger(__name__)


def get_database_url():
    database_url = os.getenv("DATABASE_URL")

    if not database_url:
        raise ValueError(
            "DATABASE_URL is missing. "
            "Please add your Render PostgreSQL External Database URL to .env"
        )

    if not re.search(r"(?:[?&])sslmode=", database_url, re.IGNORECASE):
        separator = "&" if "?" in database_url else "?"
        database_url = f"{database_url}{separator}sslmode=require"

    return database_url


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b").strip()

if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY is missing. Please add it to your .env file.")
if not GROQ_MODEL:
    raise ValueError("GROQ_MODEL cannot be empty.")

# =========================
# LLM
# =========================
llm = ChatGroq(
    model=GROQ_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0,
    timeout=30,
    max_retries=0,
)

# =========================
# State - original fields kept, new control fields added
# =========================
class TravelState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], operator.add]
    user_query: str

    # Supervisor + guardrail state
    guardrail_allowed: bool
    guardrail_reason: str
    selected_agents: list[str]
    trip_constraints: dict[str, Any]
    supervisor_reasoning: str

    # Original specialist results
    flight_results: str
    flight_details: list[dict[str, Any]]
    flight_search_url: str
    hotel_results: str
    hotel_details: list[dict[str, str]]
    hotel_search_url: str
    weather_results: str
    weather_forecast: list[dict[str, Any]]
    itinerary: str

    # New budget + HITL state
    budget_results: str
    budget_breakdown: str
    budget_currency_conversion: dict[str, Any] | None
    approval_request: str
    approved: bool
    human_feedback: str
    final_response: str

    llm_calls: int


# =========================
# Shared helpers
# =========================
KNOWN_AGENTS = {
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
}

AGENT_ORDER = [
    "flight_agent",
    "hotel_agent",
    "weather_agent",
    "budget_agent",
    "itinerary_agent",
]


def _llm_text(
    system_prompt: str,
    user_prompt: str,
    max_tokens: int = 120,
) -> str:
    response = llm.invoke(
        [
            SystemMessage(content=_context_text(system_prompt, 300)),
            HumanMessage(content=_context_text(user_prompt, 1800)),
        ],
        max_tokens=max_tokens,
    )
    return str(response.content)


def _context_text(value: Any, max_chars: int) -> str:
    """Bound tool and agent context before it is included in a Groq prompt."""
    if isinstance(value, (dict, list, tuple)):
        text = json.dumps(value, ensure_ascii=True, default=str)
    else:
        text = str(value or "")

    if len(text) <= max_chars:
        return text

    marker = " [context truncated]"
    return text[: max_chars - len(marker)] + marker


def _is_groq_provider_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    response = getattr(exc, "response", None)
    status_code = status_code or getattr(response, "status_code", None)
    module_name = type(exc).__module__
    return (
        module_name.startswith("groq")
        or module_name.startswith("langchain_groq")
        or status_code is not None
    )


def _json_from_llm(text: str) -> dict[str, Any]:
    """Extract a complete JSON object, tolerating surrounding model prose."""
    decoder = json.JSONDecoder()
    for start, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("The model did not return a JSON object.")


def _empty_constraints() -> dict[str, Any]:
    return {
        "destination": "",
        "origin": "",
        "duration": "",
        "travelers": "",
        "budget": "",
        "travel_style": "",
        "special_preferences": [],
    }


def _extract_request_facts(query: str) -> dict[str, Any]:
    facts: dict[str, Any] = {}

    try:
        facts["destination"] = extract_destination(query)
    except ValueError:
        pass

    origin_match = re.search(
        r"\bfrom\s+(.+?)\s+to\b",
        query,
        re.IGNORECASE,
    )
    if origin_match:
        facts["origin"] = origin_match.group(1).split(",", 1)[0].strip()
    else:
        origin_match = re.search(
            r"\bfrom\s+([^,.;\n]+)",
            query,
            re.IGNORECASE,
        )
        if origin_match:
            facts["origin"] = re.split(
                r"\b(?:for|with|budget|maximum|including|include|on)\b",
                origin_match.group(1),
                maxsplit=1,
                flags=re.IGNORECASE,
            )[0].strip()

    parsed_dates = _explicit_trip_dates(query)
    if parsed_dates:
        facts["departure_date"] = parsed_dates[0].isoformat()
        facts["check_in_date"] = parsed_dates[0].isoformat()
        if len(parsed_dates) > 1:
            facts["return_date"] = parsed_dates[1].isoformat()
            facts["check_out_date"] = parsed_dates[1].isoformat()

    duration_match = re.search(
        r"\b(\d+)\s*[- ]?\s*(days?|nights?)\b",
        query,
        re.IGNORECASE,
    )
    if duration_match:
        duration_count = int(duration_match.group(1))
        duration_unit = duration_match.group(2).lower().rstrip("s")
        facts["duration"] = (
            f"{duration_count} {duration_unit}"
            f"{'' if duration_count == 1 else 's'}"
        )

    adult_match = re.search(r"\b(\d+)\s*adults?\b", query, re.IGNORECASE)
    child_match = re.search(r"\b(\d+)\s*(?:children|child|kids?)\b", query, re.IGNORECASE)
    if adult_match or child_match:
        adults = int(adult_match.group(1)) if adult_match else 0
        children = int(child_match.group(1)) if child_match else 0
        facts["adult_count"] = adults
        facts["child_count"] = children
        facts["traveler_count"] = adults + children
        party_labels = []
        if adults:
            party_labels.append(f"{adults} {'adult' if adults == 1 else 'adults'}")
        if children:
            party_labels.append(f"{children} {'child' if children == 1 else 'children'}")
        facts["travelers"] = " and ".join(party_labels)
    else:
        party_match = re.search(
            r"\b(\d+)\s*(?:people|persons?|travelers?|travellers?)\b",
            query,
            re.IGNORECASE,
        )
        if party_match:
            count = int(party_match.group(1))
            facts["traveler_count"] = count
            facts["travelers"] = f"{count} travelers"
        else:
            family_match = re.search(r"\bfamily\s+of\s+(\d+)\b", query, re.IGNORECASE)
            if family_match:
                count = int(family_match.group(1))
                facts["traveler_count"] = count
                facts["travelers"] = f"{count} travelers"

    budget_match = re.search(
        r"\b(?:maximum\s+)?budget(?:\s+of|\s+is|:)?\s*"
        r"((?:₹\s*[\d,]+(?:\.\d+)?\s*(?:INR|USD|EUR|GBP)?|"
        r"(?:INR|USD|EUR|GBP)\s*[\d,]+(?:\.\d+)?)|"
        r"[\d,]+(?:\.\d+)?\s*(?:INR|USD|EUR|GBP|lakhs?|lacs?))",
        query,
        re.IGNORECASE,
    )
    if budget_match:
        facts["budget"] = budget_match.group(1).strip()

    return facts


_MONTH_NAME = (
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)"
)


def _month_day(month: str, day_number: str, year: str) -> date:
    for month_format in ("%B", "%b"):
        try:
            return datetime.strptime(
                f"{month} {int(day_number)} {year}",
                f"{month_format} %d %Y",
            ).date()
        except ValueError:
            continue
    raise ValueError("Invalid month name.")


def _explicit_trip_dates(query: str) -> list[date]:
    """Parse explicit numeric or named dates without inferring relative dates."""
    named_range = re.search(
        rf"\b(?P<month1>{_MONTH_NAME})\s+(?P<day1>\d{{1,2}})(?:st|nd|rd|th)?"
        rf"(?:,?\s*(?P<year1>20\d{{2}}))?\s*"
        rf"(?:to|through|[-–])\s*"
        rf"(?:(?P<month2>{_MONTH_NAME})\s+)?"
        rf"(?P<day2>\d{{1,2}})(?:st|nd|rd|th)?"
        rf"(?:,?\s*(?P<year2>20\d{{2}}))?\b",
        query,
        re.IGNORECASE,
    )
    if named_range:
        values = named_range.groupdict()
        year = values["year2"] or values["year1"]
        if year:
            month1 = values["month1"]
            month2 = values["month2"] or month1
            return [
                _month_day(month1, values["day1"], year),
                _month_day(month2, values["day2"], year),
            ]

    named_dates = re.findall(
        rf"\b({_MONTH_NAME})\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d{{2}})\b",
        query,
        re.IGNORECASE,
    )
    if named_dates:
        parsed = []
        for month, day_number, year in named_dates[:2]:
            try:
                parsed.append(_month_day(month, day_number, year))
            except ValueError:
                continue
        if parsed:
            return parsed

    date_tokens = re.findall(
        r"\b(20\d{2}-\d{2}-\d{2}|\d{1,2}[/-]\d{1,2}[/-]20\d{2})\b",
        query,
    )
    parsed_dates = []
    for token in date_tokens[:2]:
        for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%m/%d/%Y"):
            try:
                parsed = (
                    date.fromisoformat(token)
                    if date_format == "%Y-%m-%d"
                    else datetime.strptime(token, date_format).date()
                )
                parsed_dates.append(parsed)
                break
            except ValueError:
                continue
    return parsed_dates


def apply_trip_dates_to_request(
    user_input: str,
    departure_date: date | None = None,
    return_date: date | None = None,
) -> str:
    """Validate explicit trip dates and include form dates in workflow text."""
    facts = _extract_request_facts(user_input)
    parsed_departure = (
        date.fromisoformat(facts["departure_date"])
        if facts.get("departure_date")
        else None
    )
    parsed_return = (
        date.fromisoformat(facts["return_date"])
        if facts.get("return_date")
        else None
    )

    if departure_date and parsed_departure and departure_date != parsed_departure:
        raise ValueError("The supplied departure date conflicts with the date in your request.")
    if return_date and parsed_return and return_date != parsed_return:
        raise ValueError("The supplied return date conflicts with the date in your request.")

    departure = departure_date or parsed_departure
    returning = return_date or parsed_return
    today = date.today()
    if departure and departure <= today:
        raise ValueError("Departure date must be in the future.")
    if returning and returning <= today:
        raise ValueError("Return date must be in the future.")
    if returning and not departure:
        raise ValueError("A departure date is required when a return date is supplied.")
    if returning and departure and returning < departure:
        raise ValueError("Return date must be on or after the departure date.")

    duration_match = re.fullmatch(
        r"(\d+)\s+(days?|nights?)",
        str(facts.get("duration", "")),
        re.IGNORECASE,
    )
    if returning and departure and duration_match:
        duration = int(duration_match.group(1))
        unit = duration_match.group(2).casefold()
        expected_days = duration if unit.startswith("night") else max(duration - 1, 0)
        if (returning - departure).days != expected_days:
            raise ValueError(
                f"Return date must be {expected_days} day(s) after departure "
                "to match the explicitly requested trip duration."
            )

    if not departure_date and not return_date:
        return user_input
    if parsed_departure is None and departure_date and return_date:
        return f"{user_input.rstrip()} from {departure_date.isoformat()} to {return_date.isoformat()}"
    if parsed_departure is None and departure_date:
        return f"{user_input.rstrip()} Departure date: {departure_date.isoformat()}."
    if parsed_return is None and return_date:
        return f"{user_input.rstrip()} to {return_date.isoformat()}"
    return user_input


def _request_facts_text(query: str) -> str:
    facts = _extract_request_facts(query)
    if not facts:
        return "No additional trip details were explicitly stated."
    return json.dumps(facts, ensure_ascii=True)


# =========================
# Supervisor Agent + Input Guardrail
# =========================
def supervisor_agent(state: TravelState):
    query = state["user_query"]
    request_facts = _extract_request_facts(query)

    # Destination extraction is best-effort here. _extract_request_facts()
    # already catches ValueError, so do not call the same failing extractor
    # again without handling it.
    destination = str(request_facts.get("destination") or "").strip()
    if not destination:
        try:
            destination = str(extract_destination(query) or "").strip()
        except (ValueError, TypeError) as exc:
            logger.info("Could not extract destination from request: %s", exc)
            destination = ""
    if destination:
        destination = _resolve_destination(destination)["city"]

    llm_calls = state.get("llm_calls", 0)

    guardrail_prompt = f"""Classify this request. Allow travel planning or travel information (including flights, hotels, weather, budgets, visas, transport, sightseeing, food, packing, and itineraries). Block unrelated requests and harmful or illegal instructions. Do not block valid travel requests for missing details.
Return strict JSON: {{"allowed": true, "reason": ""}}
Request: {_context_text(query, 4000)}"""

    try:
        guardrail_raw = _llm_text(
            "You are TripMate's input guardrail. Return strict JSON only.",
            guardrail_prompt,
        )
        guardrail_result = _json_from_llm(guardrail_raw)
        allowed_value = guardrail_result.get("allowed")
        if not isinstance(allowed_value, bool):
            raise ValueError("The guardrail response did not include a boolean decision.")
        allowed = allowed_value
        guardrail_reason = str(guardrail_result.get("reason", "")).strip()
        llm_calls += 1
    except Exception as exc:
        if _is_groq_provider_error(exc):
            raise
        logger.warning(
            "Travel input guardrail failed (error_type=%s).",
            type(exc).__name__,
        )
        raise RuntimeError(
            "TripMate could not validate this request safely. Please try again."
        ) from None

    if not allowed:
        reason = guardrail_reason or (
            "TripMate AI can only help with travel-planning requests. "
            "Please ask about a destination, flight, hotel, weather, budget, "
            "or itinerary."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
            "llm_calls": llm_calls,
        }

    is_trip_plan = bool(
        re.search(r"\b(?:trip|itinerary|vacation|holiday)\b", query, re.IGNORECASE)
        and re.search(
            r"\b(?:plan|create|build|organize|design|suggest)\b",
            query,
            re.IGNORECASE,
        )
    )
    if is_trip_plan:
        constraints = _empty_constraints()
        constraints.update(request_facts)
        constraints["destination"] = destination
        return {
            "guardrail_allowed": True,
            "guardrail_reason": guardrail_reason,
            "selected_agents": AGENT_ORDER.copy(),
            "trip_constraints": constraints,
            "supervisor_reasoning": (
                "A complete trip plan requires flight, hotel, weather, budget, "
                "and itinerary research."
            ),
            "messages": [AIMessage(content="Supervisor created the agent plan.")],
            "llm_calls": llm_calls,
        }

    supervisor_prompt = f"""Select needed agents: flight_agent (flights/routes), hotel_agent (stays), weather_agent (weather/packing), budget_agent (cost/limits), itinerary_agent (always).
Return strict JSON: {{"selected_agents": [], "trip_constraints": {{"destination": "", "origin": "", "duration": "", "budget": "", "travel_style": "", "special_preferences": []}}, "reasoning": ""}}
Request: {_context_text(query, 4000)}"""

    try:
        supervisor_raw = _llm_text(
            "Route travel work to specialist agents. Return strict JSON only.",
            supervisor_prompt,
        )
        parsed = _json_from_llm(supervisor_raw)
        requested_agents = parsed.get("selected_agents", [])
        if not isinstance(requested_agents, list):
            requested_agents = []
        selected_agents = [
            name for name in AGENT_ORDER
            if name in requested_agents and name in KNOWN_AGENTS
        ]

        # The itinerary agent integrates whichever specialist results were selected.
        if "itinerary_agent" not in selected_agents:
            selected_agents.append("itinerary_agent")

        constraints = _empty_constraints()
        parsed_constraints = parsed.get("trip_constraints", {})
        if isinstance(parsed_constraints, dict):
            constraints.update(parsed_constraints)
        constraints.update(request_facts)
        constraints["destination"] = destination

        reasoning = str(parsed.get("reasoning", "")).strip()
        llm_calls += 1
    except Exception as exc:
        if _is_groq_provider_error(exc):
            raise
        logger.warning(
            "Supervisor routing fallback used (error_type=%s).",
            type(exc).__name__,
        )
        # Original workflow behavior is preserved as the fallback.
        selected_agents = AGENT_ORDER.copy()
        constraints = _empty_constraints()
        constraints.update(request_facts)
        constraints["destination"] = destination
        reasoning = (
            "Supervisor parsing failed, so the original full travel workflow "
            "was selected as a safe fallback."
        )

    return {
        "guardrail_allowed": True,
        "guardrail_reason": guardrail_reason,
        "selected_agents": selected_agents,
        "trip_constraints": constraints,
        "supervisor_reasoning": reasoning,
        "messages": [AIMessage(content="Supervisor created the agent plan.")],
        "llm_calls": llm_calls,
    }


# =========================
# Guardrail blocked response
# =========================
def guardrail_blocked_agent(state: TravelState):
    reason = state.get("final_response") or state.get("guardrail_reason") or (
        "This request was blocked by the travel input guardrail."
    )
    return {
        "final_response": reason,
        "messages": [AIMessage(content=reason)],
    }


# =========================
# Flight Agent
# =========================
_COUNTRY_GATEWAYS = {
    "AE": ("Dubai", "DXB"),
    "AR": ("Buenos Aires", "EZE"),
    "AT": ("Vienna", "VIE"),
    "AU": ("Sydney", "SYD"),
    "BD": ("Dhaka", "DAC"),
    "BR": ("Sao Paulo", "GRU"),
    "CA": ("Toronto", "YYZ"),
    "CH": ("Zurich", "ZRH"),
    "CN": ("Beijing", "PEK"),
    "DE": ("Frankfurt", "FRA"),
    "EG": ("Cairo", "CAI"),
    "ES": ("Madrid", "MAD"),
    "FR": ("Paris", "CDG"),
    "GB": ("London", "LHR"),
    "GR": ("Athens", "ATH"),
    "HK": ("Hong Kong", "HKG"),
    "ID": ("Jakarta", "CGK"),
    "IE": ("Dublin", "DUB"),
    "IN": ("Delhi", "DEL"),
    "IT": ("Rome", "FCO"),
    "JP": ("Tokyo", "HND"),
    "KH": ("Phnom Penh", "PNH"),
    "KR": ("Seoul", "ICN"),
    "LK": ("Colombo", "CMB"),
    "MV": ("Male", "MLE"),
    "MX": ("Mexico City", "MEX"),
    "MY": ("Kuala Lumpur", "KUL"),
    "NP": ("Kathmandu", "KTM"),
    "NL": ("Amsterdam", "AMS"),
    "NZ": ("Auckland", "AKL"),
    "PH": ("Manila", "MNL"),
    "QA": ("Doha", "DOH"),
    "SA": ("Riyadh", "RUH"),
    "SG": ("Singapore", "SIN"),
    "TH": ("Bangkok", "BKK"),
    "TR": ("Istanbul", "IST"),
    "TW": ("Taipei", "TPE"),
    "US": ("New York", "JFK"),
    "VN": ("Hanoi", "HAN"),
    "ZA": ("Johannesburg", "JNB"),
}
_CITY_PRIMARY_AIRPORTS = {
    "bangkok": "BKK",
    "beijing": "PEK",
    "delhi": "DEL",
    "dubai": "DXB",
    "hyderabad": "HYD",
    "kualalumpur": "KUL",
    "london": "LHR",
    "manila": "MNL",
    "newyork": "JFK",
    "paris": "CDG",
    "phuket": "HKT",
    "rome": "FCO",
    "singapore": "SIN",
    "tokyo": "HND",
}


@lru_cache(maxsize=1)
def _airport_dataset() -> dict[str, dict[str, Any]]:
    return airportsdata.load("IATA")


def _airport_iata_for_city(city: str) -> str | None:
    airports = _airport_dataset()
    direct_code = city.strip().upper()
    if direct_code in airports:
        return direct_code

    normalized_city = re.sub(r"[^a-z0-9]", "", city.casefold())
    preferred_code = _CITY_PRIMARY_AIRPORTS.get(
        re.sub(r"[^a-z0-9]", "", city.casefold())
    )
    if preferred_code in airports:
        return preferred_code

    matches = [
        (code, airport)
        for code, airport in airports.items()
        if re.sub(
            r"[^a-z0-9]",
            "",
            str(airport.get("city", "")).casefold(),
        )
        == normalized_city
    ]
    if not matches:
        return None

    matches.sort(
        key=lambda item: (
            "international" not in str(item[1].get("name", "")).casefold(),
            item[0],
        )
    )
    return matches[0][0]


def _resolve_destination(value: str) -> dict[str, str | None]:
    """Resolve a city or country to a canonical city, country, and airport."""
    candidate = value.strip()
    if not candidate:
        return {"city": "", "country": None, "country_code": None, "iata": None}

    airports = _airport_dataset()
    direct_code = candidate.upper()
    if direct_code in airports:
        airport = airports[direct_code]
        country_code = airport.get("country")
        country = pycountry.countries.get(alpha_2=country_code) if country_code else None
        return {
            "city": str(airport.get("city") or candidate),
            "country": country.name if country else None,
            "country_code": country_code,
            "iata": direct_code,
        }

    country = None
    try:
        country = pycountry.countries.lookup(candidate)
    except LookupError:
        if candidate.casefold() == "uae":
            country = pycountry.countries.get(alpha_2="AE")

    if country:
        gateway = _COUNTRY_GATEWAYS.get(country.alpha_2)
        if gateway:
            city, iata = gateway
            return {
                "city": city,
                "country": country.name,
                "country_code": country.alpha_2,
                "iata": iata if iata in airports else _airport_iata_for_city(city),
            }

        country_airports = [
            (code, airport)
            for code, airport in airports.items()
            if airport.get("country") == country.alpha_2 and airport.get("city")
        ]
        if country_airports:
            country_airports.sort(
                key=lambda item: (
                    "international" not in str(item[1].get("name", "")).casefold(),
                    item[0],
                )
            )
            code, airport = country_airports[0]
            return {
                "city": str(airport["city"]),
                "country": country.name,
                "country_code": country.alpha_2,
                "iata": code,
            }
        return {
            "city": candidate,
            "country": country.name,
            "country_code": country.alpha_2,
            "iata": None,
        }

    code = _airport_iata_for_city(candidate)
    if code:
        airport = airports[code]
        country_code = airport.get("country")
        country = pycountry.countries.get(alpha_2=country_code) if country_code else None
        return {
            "city": str(airport.get("city") or candidate),
            "country": country.name if country else None,
            "country_code": country_code,
            "iata": code,
        }

    return {"city": candidate, "country": None, "country_code": None, "iata": None}


def _flight_search_url(
    origin: str,
    destination: str,
    departure_date: str | None = None,
    return_date: str | None = None,
    travelers: int = 1,
) -> str:
    from urllib.parse import quote_plus
    query = (
        f"Flights from {origin or 'your origin'} "
        f"to {destination or 'your destination'}"
        f"{f' on {departure_date}' if departure_date else ''}"
        f"{f' returning {return_date}' if return_date else ''}"
        f" for {travelers} traveler(s)"
    )
    return "https://www.google.com/travel/flights?q=" + quote_plus(query)


def _hotel_search_url(
    destination: str,
    check_in: str | None = None,
    check_out: str | None = None,
    travelers: int = 1,
) -> str:
    from urllib.parse import quote_plus

    query = (
        f"hotels in {destination or 'your destination'}"
        f"{f' check-in {check_in}' if check_in else ''}"
        f"{f' check-out {check_out}' if check_out else ''}"
        f" for {travelers} traveler(s)"
    )
    return "https://www.google.com/search?q=" + quote_plus(query)


def _requested_flight_iata(query: str) -> str | None:
    if not re.search(
        r"\b(?:flight\s+status|status|track(?:ing)?|delayed|landed|arrived)\b",
        query,
        re.IGNORECASE,
    ):
        return None
    match = re.search(r"\b([A-Z]{2,3}\s*\d{1,4}[A-Z]?)\b", query, re.IGNORECASE)
    return re.sub(r"\s+", "", match.group(1)).upper() if match else None


def _aviation_response_error(result: Any) -> str | None:
    decoded = _decode_aviation_result(result)
    if isinstance(decoded, list):
        return next(
            (
                message
                for item in decoded
                if (message := _aviation_response_error(item))
            ),
            None,
        )
    if not isinstance(decoded, dict):
        return None
    error = decoded.get("error")
    if error is not None:
        if isinstance(error, dict):
            return str(error.get("message") or error.get("code") or error)
        return str(error)
    if decoded.get("ok") is False or decoded.get("isError") is True:
        return str(decoded.get("message") or "AviationStack returned an error response.")
    return None


def flight_agent(state: TravelState):
    facts = _extract_request_facts(state.get("user_query", ""))
    constraints = state.get("trip_constraints", {})
    query = state.get("user_query", "")
    origin = str(facts.get("origin") or constraints.get("origin") or "").strip()
    destination_input = str(
        facts.get("destination") or constraints.get("destination") or ""
    ).strip()
    destination_details = _resolve_destination(destination_input)
    destination = str(destination_details["city"] or destination_input)
    traveler_count = int(facts.get("traveler_count") or facts.get("adult_count") or 1)
    departure_date = facts.get("departure_date")
    return_date = facts.get("return_date")
    duration = facts.get("duration") or constraints.get("duration")
    flight_details: list[dict[str, Any]] = []

    flight_iata = _requested_flight_iata(query)
    if flight_iata:
        try:
            status_args = {"flight_iata": flight_iata}
            if departure_date:
                status_args["flight_date"] = departure_date
            status_result = asyncio.run(
                aviation_mcp_call("get_flight_status", status_args)
            )
            error = _aviation_response_error(status_result)
            if error:
                raise RuntimeError(error)
            status_rows = _aviation_data_rows(status_result)
            status_rows = [
                row for row in status_rows
                if isinstance(row.get("flight_status"), str)
                and row["flight_status"].strip()
            ]
            if status_rows:
                flight_details = [
                    _aviation_flight_table_row(row)
                    for row in status_rows[:5]
                ]
                flight_data = (
                    "LIVE AviationStack flight status only (not a flight search, "
                    "future schedule, fare, availability, or booking):\n"
                    + _context_text(status_rows[:5], 2200)
                )
            else:
                flight_data = (
                    f"UNAVAILABLE: AviationStack returned no flight-status "
                    f"records for {flight_iata}."
                )
        except Exception as exc:
            flight_data = _integration_failure("AviationStack flight status", exc)
        return {
            "flight_results": flight_data,
            "flight_details": flight_details,
            "messages": [
                AIMessage(
                    content=(
                        "Flight status retrieved."
                        if flight_data.startswith("LIVE AviationStack flight status")
                        else "Flight status is unavailable."
                    )
                )
            ],
        }

    sections: list[str] = []
    origin_details = _resolve_destination(origin) if origin else {}
    origin = str(origin_details.get("city") or origin)
    origin_code = origin_details.get("iata")
    destination_code = destination_details["iata"]

    if not departure_date:
        sections.append(
            "UNAVAILABLE: No departure date was supplied. AviationStack future "
            "schedules require a travel date, so no flight schedule, fare, or "
            "availability is inferred. No flight options table is shown."
        )
    elif not origin_code or not destination_code:
        sections.append(
            "UNAVAILABLE: AviationStack future schedule requires specific "
            "origin and destination airports; local city-to-IATA resolution "
            f"returned origin={origin_code or 'unresolved'}, "
            f"destination={destination_code or 'unresolved'}."
        )
    else:
        try:
            requested_date = date.fromisoformat(departure_date)
        except ValueError:
            sections.append(
                "UNAVAILABLE: The departure date must use YYYY-MM-DD."
            )
        else:
            schedule_legs = [
                ("outbound", origin_code, destination_code, requested_date)
            ]
            if return_date:
                try:
                    requested_return = date.fromisoformat(return_date)
                except ValueError:
                    sections.append(
                        "UNAVAILABLE: The return date must use YYYY-MM-DD."
                    )
                else:
                    if requested_return <= requested_date:
                        sections.append(
                            "UNAVAILABLE: The return date must be after the departure date."
                        )
                    else:
                        schedule_legs.append(
                            (
                                "return",
                                destination_code,
                                origin_code,
                                requested_return,
                            )
                        )

            for leg_name, airport_code, arrival_code, scheduled_date in schedule_legs:
                if not date.today() < scheduled_date <= date.today() + timedelta(days=366):
                    sections.append(
                        f"UNAVAILABLE: AviationStack cannot verify the {leg_name} "
                        "schedule outside its supported window (tomorrow through "
                        "about 12 months ahead)."
                    )
                    continue
                try:
                    schedule_result = asyncio.run(
                        aviation_mcp_call(
                            "future_flights_arrival_departure_schedule",
                            {
                                "airport_iata_code": airport_code,
                                "schedule_type": "departure",
                                "airline_iata": "",
                                "date": scheduled_date.isoformat(),
                                "number_of_flights": 100,
                            },
                        )
                    )
                    error = _aviation_response_error(schedule_result)
                    if error:
                        raise RuntimeError(error)
                    schedule_rows = _aviation_data_rows(schedule_result)
                    matching_rows = [
                        row
                        for row in schedule_rows
                        if _aviation_arrival_iata(row) == arrival_code
                    ]
                    if matching_rows:
                        flight_details.extend(
                            _aviation_flight_table_row(
                                row,
                                schedule_status="Schedule listed by AviationStack",
                                departure_iata=airport_code,
                                arrival_iata=arrival_code,
                            )
                            for row in matching_rows[:10]
                        )
                        sections.append(
                            f"LIVE: AviationStack future departure schedule "
                            f"({leg_name} leg; provider response; verify with airline; "
                            "not a fare or booking):\n"
                            + _context_text(matching_rows[:10], 1800)
                        )
                    else:
                        sections.append(
                            f"UNAVAILABLE: AviationStack returned no future "
                            f"departures matching the {leg_name} route."
                        )
                except Exception as exc:
                    sections.append(
                        _integration_failure(
                            f"AviationStack {leg_name} future schedule",
                            exc,
                        )
                    )

    sections.append(
        "UNAVAILABLE: Live ticket prices are not provided. AviationStack supplies "
        "flight schedule/status information only; no fare or seat availability is inferred."
    )
    if origin or destination:
        sections.append(
            "Google Flights search link (external search; not a quote or availability check): "
            + _flight_search_url(
                origin,
                destination,
                departure_date,
                return_date,
                traveler_count,
            )
        )
    requested_duration = f"; duration: {duration}" if duration else ""
    sections.append(
        f"Requested route: {origin or 'origin not specified'} to {destination or 'destination not specified'}; "
        f"departure date: {departure_date or 'not specified'}; return date: {return_date or 'not specified'}; "
        f"travelers: {traveler_count}{requested_duration}."
    )
    flight_data = "\n\n".join(sections)
    has_live = any(line.startswith("LIVE:") for line in flight_data.splitlines())
    return {
        "flight_results": flight_data,
        "flight_details": flight_details,
        "flight_search_url": (
            _flight_search_url(
                origin,
                destination,
                departure_date,
                return_date,
                traveler_count,
            )
            if origin or destination
            else ""
        ),
        "messages": [AIMessage(content="Flight data retrieved." if has_live else "Flight route/search information processed; live offers may be unavailable.")],
    }


def _aviation_arrival_iata(row: dict[str, Any]) -> str | None:
    for key in ("arrival_iata", "arrival_airport_iata", "arrival_airport_code", "arrival_iata_code"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip().upper()
    arrival = row.get("arrival")
    if isinstance(arrival, dict):
        for key in ("iata", "iata_code", "code"):
            value = arrival.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
    return None


def _aviation_flight_table_row(
    row: dict[str, Any],
    schedule_status: str | None = None,
    departure_iata: str | None = None,
    arrival_iata: str | None = None,
) -> dict[str, Any]:
    def value_from(*candidates: Any) -> str | None:
        for candidate in candidates:
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
            if isinstance(candidate, (int, float)):
                return str(candidate)
        return None

    airline = row.get("airline")
    if isinstance(airline, dict):
        airline = airline.get("name")
    flight = row.get("flight")
    departure = row.get("departure")
    arrival = row.get("arrival")
    stop_value = value_from(
        row.get("stops"),
        row.get("number_of_stops"),
        row.get("stop_count"),
        flight.get("stops") if isinstance(flight, dict) else None,
    )
    if stop_value is None:
        stopovers = row.get("stopovers")
        if isinstance(stopovers, list):
            stop_value = str(len(stopovers))
    return {
        "departure_airport": value_from(
            departure.get("airport") if isinstance(departure, dict) else None,
            row.get("departure_airport"),
        ),
        "departure_iata": value_from(
            departure.get("iata") if isinstance(departure, dict) else None,
            departure.get("iataCode") if isinstance(departure, dict) else None,
            departure_iata,
        ),
        "arrival_airport": value_from(
            arrival.get("airport") if isinstance(arrival, dict) else None,
            row.get("arrival_airport"),
        ),
        "arrival_iata": value_from(
            row.get("arrival_airport_code"),
            arrival.get("iata") if isinstance(arrival, dict) else None,
            arrival.get("iataCode") if isinstance(arrival, dict) else None,
            arrival_iata,
        ),
        "airline": value_from(airline, row.get("airline_name")),
        "flight_number": value_from(
            row.get("flight_number"),
            row.get("flight_iata"),
            flight.get("iata") if isinstance(flight, dict) else None,
            flight.get("iataNumber") if isinstance(flight, dict) else None,
        ),
        "departure": value_from(
            row.get("departure_scheduled_time"),
            departure.get("scheduled") if isinstance(departure, dict) else None,
            departure.get("scheduledTime") if isinstance(departure, dict) else None,
        ),
        "arrival": value_from(
            row.get("arrival_scheduled_time"),
            arrival.get("scheduled") if isinstance(arrival, dict) else None,
            arrival.get("scheduledTime") if isinstance(arrival, dict) else None,
        ),
        "status": value_from(
            row.get("flight_status"),
            row.get("status"),
            schedule_status,
        ),
        "duration": value_from(
            row.get("duration"),
            row.get("flight_duration"),
            flight.get("duration") if isinstance(flight, dict) else None,
        ),
        "stops": stop_value,
        "price_inr": None,
    }


def _aviation_data_rows(result: Any) -> list[dict[str, Any]]:
    """Normalize MCP text/JSON wrappers and validate AviationStack data rows."""
    decoded = _decode_aviation_result(result)
    error = _aviation_response_error(decoded)
    if error:
        raise RuntimeError(error)
    if isinstance(decoded, dict):
        for key in ("data", "results", "items", "routes", "airports"):
            if key in decoded:
                rows = decoded[key]
                if not isinstance(rows, list) or any(
                    not isinstance(row, dict) for row in rows
                ):
                    raise RuntimeError(
                        "AviationStack returned an unexpected response structure."
                    )
                return rows
        raise RuntimeError("AviationStack returned an unexpected response structure.")
    if isinstance(decoded, list):
        if all(isinstance(row, dict) for row in decoded):
            return decoded
        nested_rows = []
        for item in decoded:
            nested_rows.extend(_aviation_data_rows(item))
        return nested_rows
    raise RuntimeError("AviationStack returned an unexpected response structure.")


def _decode_aviation_result(result: Any) -> Any:
    if isinstance(result, str):
        try:
            return _decode_aviation_result(json.loads(result))
        except json.JSONDecodeError:
            return result
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return _decode_aviation_result(result["text"])
        return result
    if isinstance(result, (list, tuple)):
        decoded = [_decode_aviation_result(item) for item in result]
        return decoded[0] if len(decoded) == 1 else decoded
    return result


def _is_empty_search_result(result: Any) -> bool:
    if result is None:
        return True
    if isinstance(result, str):
        return result.strip().lower() in {"", "[]", "{}", "null", "none"}
    if isinstance(result, (list, tuple)):
        return not result or all(_is_empty_search_result(item) for item in result)
    if isinstance(result, dict):
        for key in ("results", "data", "items"):
            if key in result:
                return _is_empty_search_result(result[key])
        if "text" in result:
            return _is_empty_search_result(result["text"])
        return not result
    return False


def _search_response_failed(result: Any) -> bool:
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except json.JSONDecodeError:
            return False
        return _search_response_failed(parsed)
    if isinstance(result, (list, tuple)):
        return any(_search_response_failed(item) for item in result)
    if isinstance(result, dict):
        if result.get("ok") is False or result.get("isError") is True:
            return True
        if "error" in result:
            return True
        return "text" in result and _search_response_failed(result["text"])
    return False


def _integration_failure(provider: str, exc: Exception) -> str:
    logger.warning(
        "%s integration failed (error_type=%s).",
        provider,
        type(exc).__name__,
    )
    if (
        isinstance(exc, TimeoutError)
        or "timeout" in type(exc).__name__.casefold()
        or "timed out" in str(exc).casefold()
    ):
        detail = "request timed out"
    else:
        message = str(exc)
        lowered_message = message.casefold()
        missing_key = next(
            (
                key
                for key in (
                    "TAVILY_API_KEY",
                    "AVIATION_STACK_API_KEY",
                    "AVIATIONSTACK_API_KEY",
                )
                if f"{key} is missing" in message
            ),
            None,
        )
        status_code = getattr(exc, "status_code", None)
        response = getattr(exc, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        if status_code is None:
            status_match = re.search(
                r"\b(?:http(?:\s+status)?|status(?:\s+code)?)\s*[:=]?\s*(\d{3})\b",
                message,
                re.IGNORECASE,
            )
            if status_match:
                status_code = int(status_match.group(1))

        if missing_key:
            detail = f"{missing_key} is not configured"
        elif "function_access_restricted" in lowered_message:
            detail = "subscription plan does not permit this API function"
        elif status_code == 429:
            detail = "provider rate limit reached (HTTP 429)"
        elif status_code == 401:
            detail = "authentication was rejected (HTTP 401)"
        elif status_code == 403:
            detail = "provider access was denied (HTTP 403)"
        elif isinstance(status_code, int):
            detail = f"request failed with HTTP {status_code}"
        elif any(
            marker in lowered_message for marker in ("rate limit", "too many requests", "rate_limit")
        ):
            detail = "provider rate limit reached"
        elif any(
            marker in lowered_message
            for marker in (
                "invalid api key",
                "invalid_access_key",
                "invalid access key",
                "authentication failed",
                "unauthorized",
            )
        ):
            detail = "authentication was rejected; check the configured API key"
        elif any(
            marker in lowered_message
            for marker in (
                "connection error",
                "connectionerror",
                "connecttimeout",
                "readtimeout",
                "network unreachable",
                "dns",
            )
        ):
            detail = "provider could not be reached"
        else:
            detail = f"request failed ({type(exc).__name__})"

    return f"UNAVAILABLE: {provider} {detail}."


def _append_source_evidence(_text: str, state: TravelState) -> str:
    query = str(state.get("user_query", ""))
    facts = _extract_request_facts(query)

    flight_result = str(state.get("flight_results") or "").strip()
    if not flight_result:
        flight_result = "UNAVAILABLE: No verified flight results are available."
    hotel_result = str(state.get("hotel_results") or "").strip()
    if not hotel_result:
        hotel_result = "UNAVAILABLE: Hotel search was not run."
    weather_result = str(
        state.get("weather_results") or "UNAVAILABLE: Weather checks were not run."
    ).strip()
    budget_breakdown = str(state.get("budget_breakdown") or "").strip()
    if not budget_breakdown:
        budget_breakdown = _build_budget_breakdown(state)

    overview = _trip_overview(facts)
    generated_itinerary = _generated_itinerary_body(_text, query)
    if generated_itinerary:
        itinerary_intro = (
            "AI-generated activity suggestions based on the request and "
            "available context; these are not live availability or bookings."
        )
    else:
        generated_itinerary = _complete_daywise_itinerary(query).split(
            "\n", 1
        )[-1]
        itinerary_intro = (
            "Fallback planning suggestions; the itinerary model did not "
            "return a complete day-wise schedule. These are not live "
            "availability or bookings."
        )
    itinerary = (
        "## Day-wise itinerary\n"
        f"{itinerary_intro}\n\n"
        f"{generated_itinerary}"
    )
    tips = (
        "## Practical travel tips\n"
        "- Check entry requirements, local advisories, and attraction opening "
        "hours before travel.\n"
        "- Keep digital and offline copies of travel documents and confirmed "
        "reservations.\n"
        "- Confirm local transport routes, payment methods, and connectivity "
        "before relying on them.\n"
        "- Leave buffer time for airport transfers; search listings are not "
        "confirmed bookings."
    )
    return "\n\n".join(
        (
            overview,
            itinerary,
            tips,
            f"## Flight status\n{flight_result}",
            f"## Hotel status\n{hotel_result}",
            f"## Weather status\n{weather_result}",
            budget_breakdown,
            f"## Price / data provenance\n{_source_provenance(state)}",
        )
    )


def _requested_day_count(query: str) -> int:
    duration = _extract_request_facts(query).get("duration", "")
    duration_match = re.fullmatch(
        r"(\d+)\s+(days?|nights?)",
        str(duration),
        re.IGNORECASE,
    )
    if not duration_match:
        return 3
    count = int(duration_match.group(1))
    if duration_match.group(2).casefold().startswith("night"):
        count += 1
    return min(max(count, 1), 14)


def _itinerary_table_data(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    day_number = 0
    for original_line in text.splitlines():
        heading = re.match(r"^\s*#{1,4}\s*Day\s+(\d+)\b", original_line, re.IGNORECASE)
        if heading:
            day_number = int(heading.group(1))
            continue
        line = original_line.strip().removeprefix("-").strip()
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != 7:
            continue
        period = parts[0].strip("* ").casefold()
        if period not in {"morning", "afternoon", "evening"}:
            continue

        def field_value(value: str, label: str) -> str:
            match = re.match(rf"{re.escape(label)}\s*:\s*(.*)$", value, re.IGNORECASE)
            return match.group(1).strip() if match else ""

        time = field_value(parts[1], "Time")
        place = field_value(parts[2], "Place")
        activity = field_value(parts[3], "Activity")
        duration = field_value(parts[4], "Duration")
        cost_text = field_value(parts[5], "Estimated cost")
        transportation = field_value(parts[6], "Transportation")
        amount_match = re.search(
            r"(?:INR\s*|₹\s*)([\d,]+(?:\.\d+)?)",
            cost_text,
            re.IGNORECASE,
        )
        amount = None
        if amount_match:
            try:
                parsed_amount = float(amount_match.group(1).replace(",", ""))
            except ValueError:
                parsed_amount = math.nan
            if math.isfinite(parsed_amount) and parsed_amount >= 0:
                amount = parsed_amount

        rows.append(
            {
                "day": day_number,
                "period": period,
                "time": time,
                "place": place,
                "activity": activity,
                "duration": duration,
                "estimated_cost_inr": amount,
                "estimated_cost_text": cost_text,
                "cost_status": (
                    "ESTIMATE — not a live price"
                    if amount is not None
                    else "UNAVAILABLE — not treated as zero"
                ),
                "transportation": transportation,
            }
        )
    return rows


def _generated_itinerary_body(text: str, query: str) -> str:
    """Keep only a complete day-wise schedule from model output."""
    text = re.sub(r"```(?:markdown|md)?\s*|\s*```", "", text).strip()
    if not text:
        return ""

    section_match = re.search(
        r"(?im)^#{1,4}\s*(?:day[- ]wise itinerary|itinerary|daily plan)\s*$",
        text,
    )
    if section_match:
        text = text[section_match.end():].strip()

    day_matches = list(
        re.finditer(r"(?im)^#{1,4}\s*day\s+(\d+)\b[^\n]*", text)
    )
    required_days = _requested_day_count(query)
    day_numbers = [int(match.group(1)) for match in day_matches]
    if day_numbers != list(range(1, required_days + 1)):
        return ""

    sections = []
    all_places: set[str] = set()
    for index, match in enumerate(day_matches):
        end = (
            day_matches[index + 1].start()
            if index + 1 < len(day_matches)
            else len(text)
        )
        section = text[match.start():end].strip()
        section = re.split(
            r"(?im)^#{1,3}\s*(?:trip overview|(?:INR\s+)?budget(?:\s+breakdown)?|"
            r"flight|hotel|weather|"
            r"price\s*/\s*data provenance|provenance|practical travel tips|"
            r"sources?)\b",
            section,
            maxsplit=1,
        )[0].strip()
        rows = _itinerary_table_data(section)
        if [row["period"] for row in rows] != [
            "morning",
            "afternoon",
            "evening",
        ]:
            return ""
        meal_by_period = {
            "morning": "breakfast",
            "afternoon": "lunch",
            "evening": "dinner",
        }
        if any(
            not re.search(
                r"\b\d{1,2}:\d{2}\s*[-–]\s*\d{1,2}:\d{2}\b",
                row["time"],
            )
            or "approximate" not in row["time"].casefold()
            or not row["place"]
            or not row["activity"]
            or not re.search(
                rf"\b{meal_by_period[row['period']]}\b",
                row["activity"],
                re.IGNORECASE,
            )
            or not row["duration"]
            or not row["transportation"]
            or not row["estimated_cost_text"]
            for row in rows
        ):
            return ""
        for row in rows:
            normalized_place = re.sub(r"[^a-z0-9]", "", row["place"].casefold())
            if normalized_place and normalized_place not in {
                "notresearched",
                "notspecified",
                "unavailable",
            }:
                if normalized_place in all_places:
                    return ""
                all_places.add(normalized_place)
        sections.append(section)

    return "\n\n".join(sections)


def _trip_overview(facts: dict[str, Any]) -> str:
    destination_value = str(facts.get("destination") or "").strip()
    destination = (
        _resolve_destination(destination_value)["city"]
        if destination_value
        else "Not specified"
    )
    details = [f"- Destination: {destination}"]
    if facts.get("origin"):
        details.append(f"- Origin: {facts['origin']}")
    if facts.get("duration"):
        details.append(f"- Duration: {facts['duration']}")
    if facts.get("travelers"):
        details.append(f"- Travelers: {facts['travelers']}")
    budget = facts.get("budget")
    if budget:
        details.append(
            f"- Budget ceiling: {budget} (spending limit, not an estimated trip cost)"
        )
    details.append(
        "- Suggested activities are flexible; adjust for actual arrival and "
        "departure times."
    )
    return "## Trip overview\n" + "\n".join(details)


def _complete_daywise_itinerary(query: str) -> str:
    facts = _extract_request_facts(query)
    day_count = _requested_day_count(query)
    destination = (
        _resolve_destination(str(facts["destination"]))["city"]
        if facts.get("destination")
        else "the destination"
    )
    lines = [
        "## Day-wise itinerary",
        (
            f"Flexible suggestions for {destination}; confirm opening hours, "
            "local conditions, prices, and entry requirements. No activities "
            "or tickets are booked."
        ),
    ]
    if destination.casefold() == "dubai":
        activities = (
            (
                "Explore Al Fahidi Historical District at an unhurried pace.",
                "Walk along Dubai Creek and explore the historic waterfront; check abra operating details locally.",
                "Choose a nearby Old Dubai dining area and keep the evening flexible.",
            ),
            (
                "Visit the Dubai Frame area; confirm current entry conditions before planning an interior visit.",
                "Explore Dubai Mall and the Burj Khalifa viewing areas; paid access and availability are not verified.",
                "See the Downtown Dubai public areas and check local fountain schedules.",
            ),
            (
                "Walk around Dubai Marina and the waterfront promenade.",
                "Spend time around JBR or choose a nearby public beach, subject to local access conditions.",
                "Keep the last evening open for packing and transfer buffers.",
            ),
        )
    else:
        activities = (
            (
                f"Explore a central district in {destination} and get oriented on foot.",
                f"Visit a public cultural site or museum in {destination}; confirm hours and entry details.",
                f"Choose a local dining area in {destination} and keep the evening flexible.",
            ),
            (
                f"Explore a historic neighborhood or public park in {destination}.",
                f"Visit a local market or another cultural attraction in {destination}.",
                f"Choose a relaxed neighborhood meal in {destination}.",
            ),
            (
                f"Walk through a scenic public area in {destination}.",
                f"Choose a remaining point of interest in {destination} to suit your interests.",
                "Keep time for rest, packing, and airport or onward-transfer buffers.",
            ),
        )
    for day_number in range(1, day_count + 1):
        morning, afternoon, evening = activities[(day_number - 1) % len(activities)]
        if day_number > len(activities):
            morning += " Choose a different area from the previous days."
            afternoon += " Select a distinct site not already visited."
        lines.extend(
            (
                f"### Day {day_number}",
                f"- Morning | Time: 08:00-11:00 (approximate local time) | "
                f"Place: Not researched | Activity: Breakfast and sightseeing suggestion: {morning} | "
                "Duration: 2-3 hours (estimate) | Estimated cost: UNAVAILABLE | "
                "Transportation: Walk or check local transit options",
                f"- Afternoon | Time: 12:00-15:00 (approximate local time) | "
                f"Place: Not researched | Activity: Lunch and sightseeing suggestion: {afternoon} | "
                "Duration: 2-3 hours (estimate) | Estimated cost: UNAVAILABLE | "
                "Transportation: Check local transit or taxi options and allow transfer time",
                f"- Evening | Time: 17:00-20:00 (approximate local time) | "
                f"Place: Not researched | Activity: Dinner and flexible evening suggestion: {evening} | "
                "Duration: 2-3 hours (estimate) | Estimated cost: UNAVAILABLE | "
                "Transportation: Choose a nearby meal and verify the return route locally",
            )
        )
    return "\n".join(lines)


def _estimate_line(text: str, label: str) -> float | None:
    for line in text.splitlines():
        normalized_line = re.sub(r"[*_`]", "", line)
        match = re.match(
            rf"^\s*[-*]?\s*{re.escape(label)}\s*[:\-]\s*(.*?)\s*$",
            normalized_line,
            re.IGNORECASE,
        )
        if not match or not re.search(r"\bestimat(?:e|ed)\b", match.group(1), re.IGNORECASE):
            continue

        estimate_text = match.group(1)
        if re.search(
            r"\b(?:sandbox|test data|provider offer)\b|"
            r"\blive\s+(?:provider\s+)?offer\b",
            estimate_text,
            re.IGNORECASE,
        ):
            continue
        amount_match = re.search(
            r"(?:\bINR\b|₹)\s*([\d,]+(?:\.\d+)?)|"
            r"([\d,]+(?:\.\d+)?)\s*(?:\bINR\b|₹)",
            estimate_text,
            re.IGNORECASE,
        )
        if not amount_match:
            continue
        amount_text = next(
            group for group in amount_match.groups() if group is not None
        )
        try:
            amount = float(amount_text.replace(",", ""))
        except ValueError:
            continue
        if math.isfinite(amount) and amount > 0:
            return amount
    return None


def _estimate_assumption(text: str, label: str) -> str | None:
    for line in text.splitlines():
        if _estimate_line(line, label) is None:
            continue
        match = re.search(
            r"\b(?:assumption|assuming|based on)\s*[:—-]\s*(.+)$",
            line,
            re.IGNORECASE,
        )
        if match:
            assumption = match.group(1).strip(" .;")[:240]
            return assumption or None
    return None


def _usable_budget_estimate(text: str, label: str) -> float | None:
    amount = _estimate_line(text, label)
    if amount is None or not _estimate_assumption(text, label):
        return None
    return amount


_BUDGET_ESTIMATE_LABELS = (
    "Flights",
    "Accommodation",
    "Food",
    "Local transport",
    "Attractions",
    "Shopping",
    "Miscellaneous",
    "Emergency allowance",
)


def _budget_estimates(
    text: str,
) -> tuple[dict[str, float | None], dict[str, str]]:
    amounts = {
        label: _usable_budget_estimate(text, label)
        for label in _BUDGET_ESTIMATE_LABELS
    }
    assumptions = {
        label: _estimate_assumption(text, label) or ""
        for label in _BUDGET_ESTIMATE_LABELS
    }
    regular_categories = _BUDGET_ESTIMATE_LABELS[:-1]
    if amounts["Emergency allowance"] is None and all(
        amounts[label] is not None for label in regular_categories
    ):
        base_total = sum(
            float(amounts[label]) for label in regular_categories
        )
        amounts["Emergency allowance"] = round(base_total * 0.10, 2)
        assumptions["Emergency allowance"] = (
            "10% contingency calculated from the seven category estimates"
        )
    return amounts, assumptions


def _budget_warning() -> str:
    return (
        "The total is a planning estimate, not a provider quote. Live flight "
        "fares and verified hotel rates or availability are unavailable."
    )


def _planning_budget_results(model_text: str) -> str:
    if not model_text.strip():
        return (
            "UNAVAILABLE: The budget model returned an empty response; "
            "planning estimates could not be generated."
        )
    return (
        "LLM planning assumptions only; not provider prices or live quotes.\n"
        + model_text.strip()
    )


def _build_budget_breakdown(state: TravelState) -> str:
    lines = ["## INR budget breakdown"]
    components: list[tuple[str, str, float | None]] = []
    estimate_components: list[tuple[str, float]] = []

    budget_text = str(state.get("budget_results", ""))
    estimate_labels = _BUDGET_ESTIMATE_LABELS
    estimates, assumptions = _budget_estimates(budget_text)
    for label in estimate_labels:
        amount = estimates[label]
        if amount is None:
            components.append(
                (
                    label,
                    "UNAVAILABLE: no defensible numeric planning estimate "
                    "was generated for this category",
                    None,
                )
            )
        else:
            estimate_components.append((label, amount))
            assumption = assumptions[label]
            provider_note = (
                "; live fare unavailable"
                if label == "Flights"
                else (
                    "; live rates and room availability unverified"
                    if label == "Accommodation"
                    else ""
                )
            )
            components.append(
                (
                    label,
                    f"ESTIMATE — not a live quote: INR {amount:,.2f} "
                    "(LLM-generated planning estimate; "
                    f"assumption: {assumption}{provider_note})",
                    amount,
                )
            )
    lines.extend(f"- **{label}:** {detail}" for label, detail, _ in components)

    included = [label for label, _ in estimate_components]
    missing_estimates = [
        label for label in estimate_labels
        if label not in included
    ]
    all_estimates_present = len(included) == len(estimate_labels)
    if all_estimates_present:
        estimated_total = sum(amount for _, amount in estimate_components)
        lines.append(
            "- **Estimated trip total (all requested categories):** "
            f"INR {estimated_total:,.2f}"
        )
        lines.append(
            "- **Included categories:** " + ", ".join(included) + "."
        )
    elif estimate_components:
        estimate_subtotal = sum(amount for _, amount in estimate_components)
        lines.append(
            f"- **Planning-estimate subtotal (not a trip total):** "
            f"INR {estimate_subtotal:,.2f}"
        )
        lines.append(
            "- **Estimated trip total:** UNAVAILABLE; missing estimates for "
            + (", ".join(missing_estimates) if missing_estimates else "one or more categories")
            + ". Missing expenses are not treated as zero."
        )
    else:
        lines.append(
            "- **Estimated trip total:** UNAVAILABLE; no category estimates "
            "were returned. Missing expenses are not treated as zero."
        )

    conversion = state.get("budget_currency_conversion")
    budget_limit = _budget_limit_inr(state.get("user_query", ""), conversion)
    if budget_limit is not None:
        lines.append(
            f"- **Maximum budget ceiling:** INR {budget_limit:,.2f}; this is "
            "a spending limit, not the trip total."
        )
        if isinstance(conversion, dict) and conversion.get("source"):
            lines.append(
                "- **Budget currency conversion:** "
                f"{conversion['source_amount']:,.2f} {conversion['source_currency']} "
                f"× {conversion['rate']:,.6f} = INR {budget_limit:,.2f}; "
                f"reference rate dated {conversion['date']} "
                f"({conversion['source']}). This is not a card or cash exchange quote."
            )
    elif re.search(
        r"\b(?:USD|EUR|GBP)\b",
        str(_extract_request_facts(state.get("user_query", "")).get("budget") or ""),
        re.IGNORECASE,
    ):
        lines.append(
            "- **Budget ceiling:** UNAVAILABLE in INR because a reference "
            "exchange rate could not be verified."
        )
    lines.append(
        f"- **Budget note:** {_budget_warning()}"
    )
    return "\n".join(lines)


def _budget_currency_conversion(query: str) -> dict[str, Any] | None:
    budget = _extract_request_facts(query).get("budget")
    if not budget:
        return None
    amount_match = re.search(
        r"(?:(INR|USD|EUR|GBP)\s*|₹\s*)([\d,]+(?:\.\d+)?)"
        r"|([\d,]+(?:\.\d+)?)\s*(INR|USD|EUR|GBP)",
        budget,
        re.IGNORECASE,
    )
    if not amount_match:
        return None
    try:
        currency = (amount_match.group(1) or amount_match.group(4) or "INR").upper()
        amount_text = amount_match.group(2) or amount_match.group(3)
        amount = float(amount_text.replace(",", ""))
    except ValueError:
        return None
    if not math.isfinite(amount) or amount < 0:
        return None
    if currency == "INR":
        return {
            "source_amount": amount,
            "source_currency": currency,
            "amount_inr": amount,
            "rate": 1.0,
            "date": date.today().isoformat(),
            "source": "input currency",
        }

    try:
        response = requests.get(
            f"https://api.frankfurter.dev/v2/rate/{currency.lower()}/inr",
            timeout=8,
        )
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        logger.info(
            "Reference exchange rate unavailable (currency=%s, error_type=%s).",
            currency,
            type(exc).__name__,
        )
        return None
    if not isinstance(payload, dict):
        return None
    rate = payload.get("rate")
    rate_date = payload.get("date")
    if (
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(float(rate))
        or rate <= 0
        or not isinstance(rate_date, str)
        or not rate_date
    ):
        return None
    return {
        "source_amount": amount,
        "source_currency": currency,
        "amount_inr": amount * float(rate),
        "rate": float(rate),
        "date": rate_date,
        "source": "Frankfurter reference rate",
    }


def _budget_limit_inr(
    query: str,
    conversion: dict[str, Any] | None = None,
) -> float | None:
    details = conversion
    if details is None:
        budget = _extract_request_facts(query).get("budget")
        if not budget:
            return None
        amount_match = re.search(
            r"(?:(INR|USD|EUR|GBP)\s*|₹\s*)([\d,]+(?:\.\d+)?)"
            r"|([\d,]+(?:\.\d+)?)\s*(INR|USD|EUR|GBP)",
            budget,
            re.IGNORECASE,
        )
        if not amount_match:
            return None
        currency = (amount_match.group(1) or amount_match.group(4) or "INR").upper()
        if currency != "INR":
            return None
        try:
            amount = float(
                (amount_match.group(2) or amount_match.group(3)).replace(",", "")
            )
        except ValueError:
            return None
        return amount if math.isfinite(amount) and amount >= 0 else None

    amount = details.get("amount_inr") if isinstance(details, dict) else None
    return (
        float(amount)
        if isinstance(amount, (int, float))
        and not isinstance(amount, bool)
        and math.isfinite(float(amount))
        and amount >= 0
        else None
    )


def _decode_tavily_result(result: Any) -> Any:
    if not isinstance(result, (str, dict, list, tuple)):
        text = getattr(result, "text", None)
        if isinstance(text, str):
            return _decode_tavily_result(text)
        structured_content = getattr(result, "structuredContent", None)
        if structured_content is not None:
            return _decode_tavily_result(structured_content)
        if hasattr(result, "model_dump"):
            dumped = result.model_dump()
            if dumped:
                return _decode_tavily_result(dumped)
    if isinstance(result, str):
        try:
            return _decode_tavily_result(json.loads(result))
        except json.JSONDecodeError:
            return result
    if isinstance(result, (list, tuple)):
        decoded = [_decode_tavily_result(item) for item in result]
        if len(decoded) == 1:
            return decoded[0]
        return decoded
    if isinstance(result, dict):
        if isinstance(result.get("text"), str):
            return _decode_tavily_result(result["text"])
        return result
    return result


def _is_generic_hotel_page(
    title: str,
    destination: str,
    source_url: str = "",
) -> bool:
    if re.search(
        r"\b(?:best|top|cheap)\s+(?:hotels?|accommodations?)\b|"
        r"\b(?:search|booking engine|booking platform|"
        r"price comparison|comparison website|how to build|directory|"
        r"guide|solutions|pricing|marketplace|search engine|hotel deals|"
        r"most-searched|hotel themes|city escapes|foodie cities|"
        r"coming soon|places to stay|ihg hotels|scraper|aggregator|"
        r"roundup|booking\.com|expedia|tripadvisor|trivago|hotels\.com|"
        r"agoda|kayak|google hotels|global luxury hotel brands|"
        r"hotel accommodation options|accommodation options|"
        r"hotel brands|hotel booking options|hotel chain|hotel management|"
        r"hotel technology|hotel tech|hotel software|hotel listing|"
        r"hotel directory|hotel roundup|travel blog|travel guide|"
        r"things to do|where to stay|tourism guide|travel advice)\b",
        title,
        re.IGNORECASE,
    ):
        return True
    if source_url:
        parsed_url = urlsplit(source_url)
        host = parsed_url.netloc.casefold().removeprefix("www.")
        blocked_hosts = {
            "booking.com",
            "expedia.com",
            "tripadvisor.com",
            "trivago.com",
            "hotels.com",
            "agoda.com",
            "kayak.com",
            "apify.com",
            "google.com",
            "bing.com",
            "yahoo.com",
            "duckduckgo.com",
            "trip.com",
            "traveloka.com",
            "hotelscombined.com",
            "makemytrip.com",
            "lonelyplanet.com",
            "timeout.com",
            "travelandleisure.com",
            "theculturetrip.com",
            "holidify.com",
        }
        if any(host == blocked or host.endswith(f".{blocked}") for blocked in blocked_hosts):
            return True
        if re.search(
            r"(?:^|[/_-])(?:search|hotel-scraper|scraper|roundup|"
            r"directory|listing|gallery|blog|guide|travel-guide|"
            r"article|news|category|tag)(?:[/_.?-]|$)",
            parsed_url.path,
            re.IGNORECASE,
        ):
            return True
        if (
            not parsed_url.path.strip("/")
            and re.search(
                r"\b(?:hotel|resort|accommodation|hospitality)\b",
                title,
                re.IGNORECASE,
            )
        ):
            return True
    if (
        destination
        and title.casefold().strip()
        in {
            f"{destination.casefold().strip()} hotel",
            f"{destination.casefold().strip()} hotels",
        }
    ):
        return True
    return bool(
        re.match(
            r"^(?:hotels?|accommodations?)\s+(?:(?:accommodation\s+)?"
            r"(?:options?|brands?|directory|listings?)|in|near|for)\b",
            title,
            re.IGNORECASE,
        )
    )


@lru_cache(maxsize=1)
def _foreign_country_patterns() -> tuple[tuple[str, re.Pattern[str]], ...]:
    return tuple(
        (
            country.alpha_2,
            re.compile(
                rf"(?:,\s*|\b(?:in|near|located in|situated in)\s+)"
                rf"(?:the\s+)?{re.escape(country.name)}\b",
                re.IGNORECASE,
            ),
        )
        for country in pycountry.countries
    )


def _hotel_search_entries(
    result: Any,
    destination: str = "",
) -> list[dict[str, str]]:
    decoded = _decode_tavily_result(result)
    if isinstance(decoded, dict):
        raw_entries = decoded.get("results", [])
        if not isinstance(raw_entries, list):
            return []
    elif isinstance(decoded, list):
        raw_entries = decoded
    else:
        return []

    entries_by_name: dict[str, dict[str, str]] = {}
    destination_details = _resolve_destination(destination)
    city = str(destination_details["city"] or destination)
    country = destination_details["country"]
    city_pattern = re.compile(
        rf"(?<![a-z0-9]){re.escape(city)}(?![a-z0-9])",
        re.IGNORECASE,
    )
    destination_country_code = destination_details["country_code"]

    def add_entry(
        name: str,
        description: str,
        source_url: str,
        source_location: str = "",
        explicit_property_name: bool = False,
    ) -> None:
        name = re.sub(r"^\s*(?:\d+[\).:-]\s*)+", "", name).strip(" -*#\t")
        normalized_name = re.sub(r"[^a-z0-9]", "", name.casefold())
        parsed_url = urlsplit(source_url)
        property_evidence = re.search(
            r"\b(?:hotel|resort|inn|suites?|hostel|guesthouse|"
            r"villa|lodge|aparthotel|serviced apartments?|"
            r"boutique property|accommodation)\b",
            " ".join((name, description)),
            re.IGNORECASE,
        )
        location_evidence = " ".join(
            (name, source_location, description, source_url)
        )
        foreign_country = any(
            country_code != destination_country_code and pattern.search(location_evidence)
            for country_code, pattern in _foreign_country_patterns()
        )
        if (
            not name
            or not normalized_name
            or not city_pattern.search(location_evidence)
            or foreign_country
            or not parsed_url.scheme
            or parsed_url.scheme.casefold() not in {"http", "https"}
            or not parsed_url.netloc
            or _is_generic_hotel_page(name, destination, source_url)
            or not (explicit_property_name or property_evidence)
        ):
            return

        entry = {
            "name": name[:160],
            "location": (
                source_location[:180]
                or ", ".join(part for part in (city, country) if part)
            ),
            "description": description[:500].strip(),
            "url": source_url[:500],
        }
        previous = entries_by_name.get(normalized_name)
        if previous is None or (
            not previous["description"] and entry["description"]
        ):
            entries_by_name[normalized_name] = entry

    for item in raw_entries:
        if not isinstance(item, dict):
            continue

        content = str(item.get("content") or item.get("description") or "").strip()
        source_url = str(item.get("url") or item.get("source_url") or "").strip()
        source_location = str(item.get("location") or item.get("address") or "").strip()
        explicit_name = (
            item.get("hotel_name")
            or item.get("property_name")
        )
        title = str(explicit_name or item.get("title") or "").strip()
        generic_title = _is_generic_hotel_page(title, destination, source_url)
        if title and not generic_title:
            add_entry(
                title,
                content,
                source_url,
                source_location,
                explicit_property_name=bool(explicit_name),
            )

        for linked_property in re.finditer(
            r"\[([^\]]{2,160})\]\((https?://[^)\s]+)\)",
            content,
            re.IGNORECASE,
        ):
            linked_name, linked_url = linked_property.groups()
            add_entry(
                linked_name,
                content,
                linked_url,
                source_location,
                explicit_property_name=False,
            )

    return list(entries_by_name.values())[:6]


def _format_current_weather(result: Any) -> str:
    decoded = _decode_tavily_result(result)
    if not isinstance(decoded, dict):
        return "UNAVAILABLE: OpenWeather returned an unexpected current-weather response."

    city = str(decoded.get("city") or "").strip()
    fields = (
        ("temperature_c", "temperature", "°C"),
        ("feels_like_c", "feels like", "°C"),
        ("humidity", "humidity", "%"),
        ("condition", "condition", ""),
        ("wind_speed", "wind speed", "m/s"),
    )
    values = []
    missing = []
    for key, label, unit in fields:
        value = decoded.get(key)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(label)
            continue
        if key == "condition":
            rendered = str(value).strip()
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            separator = "" if unit == "%" else " "
            rendered = f"{value:g}{separator}{unit}" if unit else f"{value:g}"
        else:
            missing.append(label)
            continue
        values.append(f"{label}: {rendered}")

    if not values:
        return "UNAVAILABLE: OpenWeather returned no usable current-weather fields."
    city_label = f"{city}: " if city else ""
    detail = "; ".join(values)
    if missing:
        detail += "; unavailable fields: " + ", ".join(missing)
    return f"LIVE current conditions (not a travel-date forecast): {city_label}{detail}."


def _format_weather_forecast(
    result: Any,
    departure_date: str | None,
    return_date: str | None,
) -> str:
    if not departure_date:
        return "UNAVAILABLE: Travel dates were not supplied for a date-specific forecast."

    rows = _weather_forecast_table_rows(
        result,
        departure_date,
        return_date,
    )
    available_rows = [
        row for row in rows
        if row["status"] == "Forecast available"
    ]
    if not available_rows:
        return (
            "UNAVAILABLE: OpenWeather's five-day forecast does not cover "
            "the requested travel dates."
        )
    return "; ".join(
        f"{row['date']}: "
        + (
            f"{row['temperature_min_c']:g}–{row['temperature_max_c']:g} °C"
            if row["temperature_min_c"] is not None
            and row["temperature_max_c"] is not None
            else "temperature range unavailable"
        )
        + (f", {row['condition']}" if row["condition"] else "")
        + (
            f", rain probability up to {row['rain_probability_percent']:g}%"
            if row["rain_probability_percent"] is not None
            else ""
        )
        + (f", humidity {row['humidity']}%" if row["humidity"] is not None else "")
        + f", recommendation: {row['recommendation']}"
        if row["status"] == "Forecast available"
        else f"{row['date']}: forecast unavailable"
        for row in rows
    )


def _weather_forecast_table_rows(
    result: Any,
    departure_date: str | None,
    return_date: str | None,
    location: str = "",
) -> list[dict[str, Any]]:
    if not departure_date:
        return []

    try:
        start = date.fromisoformat(departure_date)
        end = date.fromisoformat(return_date) if return_date else start
    except ValueError:
        return []
    if end < start:
        return []

    decoded = _decode_tavily_result(result)
    items = decoded.get("forecast") if isinstance(decoded, dict) else None
    if not isinstance(items, list):
        items = []
    if isinstance(decoded, dict):
        location = str(decoded.get("city") or location).strip()

    matching_by_day: dict[date, list[dict[str, Any]]] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        when = str(item.get("datetime") or "").strip()
        try:
            forecast_time = datetime.fromisoformat(when)
        except ValueError:
            continue
        forecast_day = forecast_time.date()
        temperature = item.get("temperature_c")
        if (
            not start <= forecast_day <= end
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
        ):
            continue
        temperature_min = item.get("temperature_min_c", temperature)
        temperature_max = item.get("temperature_max_c", temperature)
        if (
            not isinstance(temperature_min, (int, float))
            or not math.isfinite(float(temperature_min))
            or not isinstance(temperature_max, (int, float))
            or not math.isfinite(float(temperature_max))
        ):
            continue
        matching_by_day.setdefault(forecast_day, []).append(
            {
                "hour_distance": abs(forecast_time.hour - 12),
                "temperature_c": float(temperature),
                "temperature_min_c": float(temperature_min),
                "temperature_max_c": float(temperature_max),
                "condition": str(item.get("condition") or "").strip(),
                "humidity": item.get("humidity"),
                "rain_probability_percent": item.get("rain_probability_percent"),
            }
        )

    rows = []
    for day_offset in range((end - start).days + 1):
        forecast_day = start + timedelta(days=day_offset)
        samples = matching_by_day.get(forecast_day, [])
        if not samples:
            rows.append(
                {
                    "date": forecast_day.isoformat(),
                    "location": location,
                    "temperature_c": None,
                    "temperature_min_c": None,
                    "temperature_max_c": None,
                    "condition": "",
                    "rain_probability_percent": None,
                    "humidity": None,
                    "recommendation": "No reliable forecast sample is available for this date.",
                    "status": "Forecast unavailable",
                }
            )
            continue

        midday_sample = min(samples, key=lambda sample: sample["hour_distance"])
        humidity_values = [
            float(sample["humidity"])
            for sample in samples
            if isinstance(sample.get("humidity"), (int, float))
            and not isinstance(sample.get("humidity"), bool)
            and math.isfinite(float(sample["humidity"]))
        ]
        rain_values = [
            float(sample["rain_probability_percent"])
            for sample in samples
            if isinstance(sample.get("rain_probability_percent"), (int, float))
            and not isinstance(sample.get("rain_probability_percent"), bool)
            and math.isfinite(float(sample["rain_probability_percent"]))
            and 0 <= float(sample["rain_probability_percent"]) <= 100
        ]
        minimum = min(sample["temperature_min_c"] for sample in samples)
        maximum = max(sample["temperature_max_c"] for sample in samples)
        rain_probability = max(rain_values) if rain_values else None
        recommendation = "Recheck the forecast closer to departure."
        if rain_probability is not None and rain_probability >= 50:
            recommendation = "Carry rain protection; at least one forecast period has a high chance of precipitation."
        elif maximum >= 35:
            recommendation = "Plan outdoor activities for cooler hours and carry water."
        elif minimum <= 5:
            recommendation = "Pack warm layers for cooler conditions."

        rows.append(
            {
                "date": forecast_day.isoformat(),
                "location": location,
                "temperature_c": midday_sample["temperature_c"],
                "temperature_min_c": minimum,
                "temperature_max_c": maximum,
                "condition": midday_sample["condition"],
                "rain_probability_percent": rain_probability,
                "humidity": (
                    round(sum(humidity_values) / len(humidity_values))
                    if humidity_values
                    else None
                ),
                "recommendation": recommendation,
                "status": "Forecast available",
            }
        )
    return rows


def _source_provenance(state: TravelState) -> str:
    flight_results = str(state.get("flight_results", ""))
    hotel_results = str(state.get("hotel_results", ""))
    flight_lines = flight_results.splitlines()
    schedule_line = next(
        (
            line
            for line in flight_lines
            if line.startswith(
                (
                    "LIVE AviationStack future departure schedule",
                    "LIVE: AviationStack future departure schedule",
                )
            )
        ),
        "",
    )
    route_line = next(
        (line for line in flight_lines if line.startswith("LIVE AviationStack route reference")),
        "",
    )
    schedule_status = (
        "LIVE AviationStack future schedule"
        if schedule_line
        else ("LIVE AviationStack route reference only (no dated schedule)" if route_line else "UNAVAILABLE AviationStack future schedule")
    )
    fare_status = "UNAVAILABLE live ticket prices (no fare provider configured)"
    hotel_offer_status = (
        "UNAVAILABLE verified hotel rates/availability "
        "(Tavily search does not verify either)"
    )
    hotel_listings = (
        "LIVE Tavily search observations (not verified inventory)."
        if hotel_results.startswith("LIVE:")
        else "UNAVAILABLE Tavily hotel listings"
    )
    estimate_labels = _BUDGET_ESTIMATE_LABELS
    budget_text = str(state.get("budget_results", ""))
    estimates, assumptions = _budget_estimates(budget_text)
    estimate_sources = []
    included_estimates = []
    excluded_estimates = []
    for label in estimate_labels:
        amount = estimates[label]
        if amount is None:
            estimate_sources.append(f"{label}: UNAVAILABLE")
            excluded_estimates.append(label)
        else:
            estimate_sources.append(
                f"{label}: ESTIMATE — not a live quote, INR {amount:,.2f} "
                "(LLM-generated planning estimate; assumption: "
                f"{assumptions[label]})"
            )
            included_estimates.append(label)
    conversion = state.get("budget_currency_conversion")
    budget_limit = _budget_limit_inr(
        state.get("user_query", ""),
        conversion if isinstance(conversion, dict) else None,
    )
    budget_value = _extract_request_facts(
        state.get("user_query", "")
    ).get("budget")
    if budget_limit is not None:
        if isinstance(conversion, dict) and conversion.get("source") == "Frankfurter reference rate":
            ceiling_source = (
                f"Budget ceiling: {conversion['source_amount']:,.2f} "
                f"{conversion['source_currency']} = INR {budget_limit:,.2f} "
                f"at the {conversion['date']} Frankfurter reference rate"
            )
        else:
            ceiling_source = (
                f"Budget ceiling: INR {budget_limit:,.2f} "
                "(not a calculated trip total)"
            )
    elif budget_value and re.search(r"\b(?:USD|EUR|GBP)\b", budget_value, re.IGNORECASE):
        ceiling_source = "Budget ceiling: foreign currency reference conversion unavailable"
    else:
        ceiling_source = "Budget ceiling: not supplied"
    budget_provenance = "; ".join([ceiling_source, *estimate_sources])
    estimate_amounts = [
        float(estimates[label])
        for label in estimate_labels
        if estimates[label] is not None
    ]
    if len(estimate_amounts) == len(estimate_labels):
        subtotal_source = (
            f"Estimated trip total: INR {sum(estimate_amounts):,.2f} "
            "(all requested estimate categories)"
        )
    elif estimate_amounts:
        subtotal_source = (
            f"Planning-estimate subtotal: INR {sum(estimate_amounts):,.2f} "
            "(incomplete; not a trip total)"
        )
    else:
        subtotal_source = "Estimated trip total: unavailable"
    warning = _budget_warning()
    return (
        "PRICE / DATA PROVENANCE\n"
        f"- Flights and fares: {fare_status}; schedules: {schedule_status}.\n"
        f"- Hotel search: {hotel_listings}; "
        f"hotel rates/availability: {hotel_offer_status}\n"
        f"- Budget provenance: {budget_provenance}; {subtotal_source}.\n"
        f"- Included estimate categories: "
        f"{', '.join(included_estimates) if included_estimates else 'none'}; "
        f"excluded/unavailable: "
        f"{', '.join(excluded_estimates) if excluded_estimates else 'none'}. "
        f"Budget warning: {warning}"
    )


# =========================
# Hotel Agent - original behavior kept
# =========================
def hotel_agent(state: TravelState):
    query_facts = _extract_request_facts(state.get("user_query", ""))
    constraints = state.get("trip_constraints", {})
    destination_input = str(
        query_facts.get("destination")
        or constraints.get("destination")
        or ""
    ).strip()
    destination_details = _resolve_destination(destination_input)
    destination = str(destination_details["city"] or destination_input)
    destination_label = ", ".join(
        part
        for part in (destination, destination_details["country"])
        if part
    )
    guest_count = int(
        query_facts.get("traveler_count")
        or query_facts.get("adult_count")
        or 1
    )
    duration = query_facts.get("duration") or constraints.get("duration")
    check_in = query_facts.get("departure_date")
    check_out = query_facts.get("return_date")
    if check_in and not check_out and duration:
        duration_match = re.fullmatch(r"(\d+)\s+(days?|nights?)", str(duration), re.IGNORECASE)
        if duration_match:
            count = int(duration_match.group(1))
            nights = count if duration_match.group(2).casefold().startswith("night") else max(count - 1, 1)
            try:
                check_out = (date.fromisoformat(check_in) + timedelta(days=nights)).isoformat()
            except ValueError:
                check_out = None

    requested_budget = query_facts.get("budget")
    if not requested_budget and re.search(
        r"\b(?:budget[- ]friendly|affordable|low[- ]cost|cheap)\b",
        str(state.get("user_query", "")),
        re.IGNORECASE,
    ):
        requested_budget = "budget-friendly"
    budget_context = requested_budget or "not specified"
    query = (
        f"Find named individual hotels and resorts in {destination_label or 'the requested destination'}. "
        f"Travel dates: check-in {check_in or 'not specified'}, "
        f"check-out {check_out or 'not specified'}; guests: {guest_count}; "
        f"trip duration: {duration or 'not specified'}; budget: {budget_context}. "
        "Prioritize official individual property pages, not booking aggregators, travel guides, blogs, "
        "directories, or unrelated websites. Return properties specifically located in this destination "
        "(not another city or country), with the property name, location, a concise factual description, "
        "and source URL. "
        "Do not claim room availability or confirmed prices; web-search observations do not verify either."
    )
    entries: list[dict[str, str]] = []
    try:
        raw_results = asyncio.run(tavily_mcp_search(query))
        if _search_response_failed(raw_results) or _is_empty_search_result(raw_results):
            hotel_results = "UNAVAILABLE: Tavily did not return usable named hotel results."
        else:
            entries = _hotel_search_entries(raw_results, destination)
            if entries:
                listing_lines = [
                    f"- {entry['name']} | Location: {entry['location']} | "
                    f"Description: {entry['description'] or 'Description unavailable.'} | "
                    f"Source: {entry['url']}"
                    for entry in entries
                ]
                hotel_results = (
                    "LIVE: Tavily hotel search observations only; room availability, rates, and bookings are not verified.\n"
                    + _context_text("\n".join(listing_lines), 2200)
                )
            else:
                hotel_results = (
                    "UNAVAILABLE: Tavily results did not contain qualifying named property pages. "
                    "Try a more specific destination or hotel preference."
                )
    except Exception as exc:
        hotel_results = _integration_failure("Tavily hotel search", exc)

    hotel_results = (
        f"{hotel_results}\n"
        "Rates and room availability: UNAVAILABLE; no live hotel inventory provider is configured."
    )
    return {
        "hotel_results": hotel_results,
        "hotel_details": entries,
        "hotel_search_url": _hotel_search_url(
            destination_label,
            check_in,
            check_out,
            guest_count,
        ),
        "messages": [AIMessage(content="Hotel information processed.")],
    }


# =========================
# Weather Agent - original behavior kept
# =========================
def weather_agent(state: TravelState):
    constraints = state.get("trip_constraints", {})
    facts = _extract_request_facts(state.get("user_query", ""))
    destination = str(
        facts.get("destination") or constraints.get("destination") or ""
    ).strip()
    if not destination:
        try:
            destination = extract_destination(state["user_query"])
        except ValueError:
            return {
                "weather_results": (
                    "Weather information is unavailable until a destination "
                    "is provided."
                ),
                "messages": [
                    AIMessage(content="A destination is required for weather data.")
                ],
            }
    destination_details = _resolve_destination(destination)
    city = str(destination_details["city"] or destination)
    location_query = ", ".join(
        part
        for part in (city, destination_details["country"])
        if part
    )

    try:
        weather_data = asyncio.run(weather_mcp_search(location_query))
        current_result = _format_current_weather(weather_data)
    except Exception as exc:
        logger.warning(
            "OpenWeather current conditions failed (error_type=%s).",
            type(exc).__name__,
        )
        current_result = _integration_failure("OpenWeather current weather", exc)

    departure_date = facts.get("departure_date")
    return_date = facts.get("return_date")
    forecast_data = None
    if departure_date:
        try:
            forecast_data = asyncio.run(forecast_mcp_search(location_query))
            forecast_result = _format_weather_forecast(
                forecast_data,
                departure_date,
                return_date,
            )
        except Exception as exc:
            logger.warning(
                "OpenWeather travel-date forecast failed (error_type=%s).",
                type(exc).__name__,
            )
            forecast_result = _integration_failure(
                "OpenWeather travel-date forecast",
                exc,
            )
    else:
        forecast_result = (
            "UNAVAILABLE: Travel dates were not supplied for a date-specific forecast."
        )
    forecast_rows = _weather_forecast_table_rows(
        forecast_data,
        departure_date,
        return_date,
        city,
    )

    return {
        "weather_results": _context_text(
            f"{current_result}\nTravel-date forecast: {forecast_result}",
            1800,
        ),
        "weather_forecast": forecast_rows,
        "messages": [
            AIMessage(
                content="Weather information processed."
            )
        ],
    }


# =========================
# Budget Agent - new specialist
# =========================
def budget_agent(state: TravelState):
    conversion = _budget_currency_conversion(state.get("user_query", ""))
    budget_state = {**state, "budget_currency_conversion": conversion}
    prompt = f"""Return exactly eight lines and nothing else, labelled Flights, Accommodation, Food, Local transport, Attractions, Shopping, Miscellaneous, and Emergency allowance. Provide rounded INR planning allowances for the whole party and whole trip, not per person, when the relevant route, dates, duration, traveler count, and nights are available. Estimate round-trip flights and hotel accommodation even though no live fare or room-inventory provider is configured; clearly label them as rough, unverified planning estimates. Calculate food and local transportation for each traveler and trip day, accommodation for each room/night assumption, and flights for every traveler. Shopping and miscellaneous are separate categories. Make emergency allowance a stated contingency estimate (about 10% of the other estimated categories) and do not double-count it. If a category cannot be estimated from the available trip facts, write UNAVAILABLE and explain why; never use zero to stand in for missing costs. Use exactly this form: "Flights: ESTIMATE — not a live quote: INR 10000; assumption: rough round-trip planning allowance for the specified route and two travelers". Include a short, concrete assumption for each amount. All amounts are LLM planning assumptions only, not researched current prices, provider quotes, booking prices, or verified market rates. Never imply a flight fare, hotel rate, room availability, or booking was verified. Do not treat Tavily snippets as offers. Do not output an itinerary, weather, provider prices, booking claims, or a trip total. Do not invent exchange rates.
Original request: {_context_text(state.get('user_query', ''), 800)}
Authoritative trip facts: {_request_facts_text(state.get('user_query', ''))}
Planning constraints: {_context_text(state.get('trip_constraints', {}), 300)}"""

    response = llm.invoke(
        [
            SystemMessage(content="You are a practical travel budget analyst."),
            HumanMessage(content=_context_text(prompt, 2200)),
        ],
        max_tokens=1000,
    )

    budget_results = _planning_budget_results(
        _context_text(response.content, 1400)
    )
    return {
        "budget_results": budget_results,
        "budget_breakdown": _build_budget_breakdown(
            {
                **budget_state,
                "budget_results": budget_results,
            }
        ),
        "budget_currency_conversion": conversion,
        "messages": [AIMessage(content="Budget assessment generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Itinerary Agent - original behavior extended with selected results
# =========================
def itinerary_agent(state: TravelState):
    prompt = f"""Create a useful itinerary for human review. Return ONLY a day-wise schedule, using exactly one heading per requested day in the form "### Day 1" and exactly three activity lines per day in Morning, Afternoon, Evening order. Use this exact pipe-delimited format, with seven fields per line and no extra pipe characters: "Morning | Time: 08:00-11:00 (approximate local time) | Place: named attraction or UNAVAILABLE | Activity: concise description | Duration: 2 hours (estimate) | Estimated cost: ESTIMATE INR 500 for all travelers or UNAVAILABLE | Transportation: practical mode and transfer note". Give each activity an approximate local time range, clearly labeled as an estimate rather than a flight time. Include a breakfast suggestion with morning sightseeing, lunch with afternoon sightseeing, and dinner with an evening activity; group nearby places and avoid duplicate attractions. Prioritize stated interests, name attractions only when you can identify a plausible real place, and do not fabricate opening hours. Keep the pace realistic for travelers, duration, and budget; include transfer buffers. Use explicit flight arrival/departure times only when present in provider results. Keep arrival/departure days light when times are known; otherwise keep first/last day flexible and do not assume flight times. Mark activity costs as rough INR planning estimates for the whole party, never live prices; if not defensible, write UNAVAILABLE (not zero). Recommend transport, but don't claim routes or times were checked. Search observations do not verify bookings, prices, opening hours, or future weather. Treat current weather only as current conditions, never as a travel-date forecast. Do not include a trip overview, budget breakdown, flight/hotel/weather status, provenance section, fare, hotel rate, or booking claim in this itinerary text; those are rendered separately from source data. Do not contradict the user's request.
Request: {_context_text(state.get('user_query', ''), 700)}
Authoritative trip facts extracted from the user's request: {_request_facts_text(state.get('user_query', ''))}
Constraints: {_context_text(state.get('trip_constraints', {}), 300)}
Flights: {_context_text(state.get('flight_results', ''), 500)}
Hotels: {_context_text(state.get('hotel_results', ''), 500)}
Current weather and forecast status: {_context_text(state.get('weather_results', ''), 300)}
Planning estimates and assumptions: {_context_text(state.get('budget_results', ''), 400)}"""

    response = llm.invoke(
        [
            SystemMessage(content="You are an expert travel planner."),
            HumanMessage(content=_context_text(prompt, 3800)),
        ],
        max_tokens=2800,
    )

    draft = _append_source_evidence(
        _context_text(response.content, 4000),
        state,
    )
    approval_request = (
        "Please review the generated draft itinerary. Approve it to create the "
        "final polished plan, or provide feedback for revision."
    )

    return {
        "itinerary": draft,
        "approval_request": approval_request,
        "messages": [AIMessage(content="Draft itinerary created for human review.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Human-in-the-Loop approval
# =========================
def human_approval_agent(state: TravelState):
    # Do not wrap interrupt() in try/except. LangGraph uses it to pause execution.
    review = interrupt(
        {
            "question": "Do you approve this itinerary?",
            "draft_itinerary": state.get("itinerary", ""),
            "approval_request": state.get("approval_request", ""),
            "selected_agents": state.get("selected_agents", []),
            "supervisor_reasoning": state.get("supervisor_reasoning", ""),
            "expected_response": {
                "approved": True,
                "feedback": "Optional revision feedback",
            },
        }
    )

    approved = bool(review.get("approved", False))
    human_feedback = str(review.get("feedback", "")).strip()

    return {
        "approved": approved,
        "human_feedback": human_feedback,
        "messages": [AIMessage(content="Human approval step completed.")],
    }


# =========================
# Final Response Agent - applies the human review to the itinerary draft
# =========================
def final_agent(state: TravelState):
    approved = state.get("approved", False)
    review_instruction = (
        "The user approved this draft. Produce a distinct, polished final "
        "itinerary while preserving its approved details."
        if approved
        else (
            "The user requested a revision. Apply this feedback carefully: "
            f"{state.get('human_feedback', '') or 'Improve the draft before finalizing it.'}"
        )
    )

    grounded_sources = _context_text(
        {
            "flight_results": state.get("flight_results", ""),
            "hotel_results": state.get("hotel_results", ""),
            "weather_results": state.get("weather_results", ""),
            "budget_results": state.get("budget_results", ""),
            "budget_breakdown": state.get("budget_breakdown", ""),
        },
        2400,
    )
    final_prompt = f"""Create a polished {'final' if approved else 'revised'} itinerary from the reviewed draft. Apply the human review instructions materially; do not merely acknowledge them. Preserve all approved/requested trip details, including origin, destination, dates, traveler count, duration, budget, and stated preferences. Keep the same requested trip facts and practical activity suggestions while improving clarity and flow. Return exactly one heading per requested day ("### Day 1", etc.) and exactly three activity lines in Morning, Afternoon, Evening order. Use the same seven-field pipe-delimited format as the draft: "Morning | Time: 08:00-11:00 (approximate local time) | Place: named attraction or UNAVAILABLE | Activity: concise description | Duration: 2 hours (estimate) | Estimated cost: ESTIMATE INR 500 for all travelers or UNAVAILABLE | Transportation: practical mode and transfer note". Include breakfast with morning sightseeing, lunch with afternoon sightseeing, dinner with an evening activity, realistic transfer buffers, and no duplicate attractions. Preserve the meal, transport, duration, and cost-estimate guidance. Use only explicit flight times in provider results; never invent them. Do not output an overview, budget, flight/hotel/weather status, provenance, provider prices, or booking claims; the application renders those separately from workflow state. Never invent flight fares, hotel prices, bookings, verified opening hours, or live availability. Do not contradict the user's trip facts. Search results are observations only, current weather is not a travel-date forecast, and none of this implies bookings or availability.
User request: {_context_text(state.get('user_query', ''), 800)}
Authoritative trip facts extracted from the user's request: {_request_facts_text(state.get('user_query', ''))}
Constraints: {_context_text(state.get('trip_constraints', {}), 400)}
Human review: {_context_text(review_instruction, 600)}
Current itinerary draft: {_context_text(state.get('itinerary', ''), 2000)}
Provider and search observations (do not invent or treat as confirmed offers): {grounded_sources}"""

    response = llm.invoke(
        [
            SystemMessage(
                content="You are a professional travel itinerary planner."
            ),
            HumanMessage(content=_context_text(final_prompt, 4200)),
        ],
        max_tokens=2800,
    )

    generated_text = _context_text(response.content, 4000)
    generated_schedule = _generated_itinerary_body(
        generated_text,
        str(state.get("user_query", "")),
    )
    if not generated_schedule:
        if approved:
            raise RuntimeError(
                "The itinerary model returned an incomplete final plan. "
                "No final plan was created; please retry."
            )
        raise RuntimeError(
            "The itinerary model returned an incomplete revision. "
            "No revised draft was created; please retry with focused feedback."
        )

    if approved:
        reviewed_schedule = _generated_itinerary_body(
            str(state.get("itinerary", "")),
            str(state.get("user_query", "")),
        )
        normalize_reviewed = re.sub(r"\s+", " ", reviewed_schedule).casefold()
        normalize_generated = re.sub(r"\s+", " ", generated_schedule).casefold()
        if reviewed_schedule and normalize_reviewed == normalize_generated:
            raise RuntimeError(
                "The itinerary model returned the unchanged approved draft. "
                "No distinct final plan was created; please retry."
            )

    if not approved:
        revised_draft = _append_source_evidence(generated_schedule, state)
        return {
            "itinerary": revised_draft,
            "approval_request": (
                "Please review the revised draft. Approve it or request another revision."
            ),
            "final_response": "",
            "messages": [
                AIMessage(content="Revised itinerary prepared for human review.")
            ],
            "llm_calls": state.get("llm_calls", 0) + 1,
        }

    final_response = _append_source_evidence(generated_schedule, state)

    return {
        "final_response": final_response,
        "messages": [AIMessage(content="Final travel plan generated.")],
        "llm_calls": state.get("llm_calls", 0) + 1,
    }


# =========================
# Dynamic Supervisor Routing
# =========================
ROUTE_MAP = {
    "guardrail_blocked": "guardrail_blocked",
    "flight_agent": "flight_agent",
    "hotel_agent": "hotel_agent",
    "weather_agent": "weather_agent",
    "budget_agent": "budget_agent",
    "itinerary_agent": "itinerary_agent",
}


def _selected_agents(state: TravelState) -> list[str]:
    selected = state.get("selected_agents", [])
    return [agent for agent in AGENT_ORDER if agent in selected]


def route_from_supervisor(state: TravelState) -> str:
    if not state.get("guardrail_allowed", True):
        return "guardrail_blocked"

    selected = _selected_agents(state)
    return selected[0] if selected else "itinerary_agent"


def route_after_agent(current_agent: str):
    def route(state: TravelState) -> str:
        selected = _selected_agents(state)
        current_index = AGENT_ORDER.index(current_agent)

        for next_agent in AGENT_ORDER[current_index + 1 :]:
            if next_agent in selected:
                return next_agent

        return "itinerary_agent"

    return route


def route_after_final_agent(state: TravelState) -> str:
    return "end" if state.get("approved", False) else "human_approval"


def _safe_supervisor_node(state: TravelState) -> dict[str, Any]:
    try:
        return supervisor_agent(state)
    except Exception as exc:
        logger.error(
            "Supervisor or input guardrail failed (error_type=%s); failing closed.",
            type(exc).__name__,
        )
        reason = (
            "TripMate could not validate this request safely. No travel agents "
            "were run; please try again."
        )
        return {
            "guardrail_allowed": False,
            "guardrail_reason": reason,
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": reason,
            "final_response": reason,
            "messages": [AIMessage(content=reason)],
        }


def _safe_agent_node(agent_name: str):
    def run(state: TravelState) -> dict[str, Any]:
        try:
            return globals()[agent_name](state)
        except Exception as exc:
            logger.warning(
                "Travel agent failed (agent=%s, error_type=%s).",
                agent_name,
                type(exc).__name__,
            )
            reason = (
                f"UNAVAILABLE: {agent_name.replace('_', ' ')} failed "
                f"(error type: {type(exc).__name__})."
            )
            if agent_name == "flight_agent":
                facts = _extract_request_facts(state.get("user_query", ""))
                origin = str(facts.get("origin") or "").strip()
                destination = str(facts.get("destination") or "").strip()
                return {
                    "flight_results": reason,
                    "flight_details": [],
                    "flight_search_url": (
                        _flight_search_url(
                            origin,
                            destination,
                            facts.get("departure_date"),
                            facts.get("return_date"),
                            int(facts.get("traveler_count") or 1),
                        )
                        if origin or destination
                        else ""
                    ),
                    "messages": [AIMessage(content="Flight data is unavailable.")],
                }
            if agent_name == "hotel_agent":
                facts = _extract_request_facts(state.get("user_query", ""))
                destination = str(
                    facts.get("destination")
                    or state.get("trip_constraints", {}).get("destination")
                    or ""
                )
                return {
                    "hotel_results": reason,
                    "hotel_details": [],
                    "hotel_search_url": _hotel_search_url(
                        destination,
                        facts.get("departure_date"),
                        facts.get("return_date"),
                        int(facts.get("traveler_count") or 1),
                    ),
                    "messages": [AIMessage(content="Hotel data is unavailable.")],
                }
            if agent_name == "weather_agent":
                facts = _extract_request_facts(state.get("user_query", ""))
                return {
                    "weather_results": reason,
                    "weather_forecast": _weather_forecast_table_rows(
                        None,
                        facts.get("departure_date"),
                        facts.get("return_date"),
                        str(facts.get("destination") or ""),
                    ),
                    "messages": [AIMessage(content="Weather data is unavailable.")],
                }
            if agent_name == "budget_agent":
                return {
                    "budget_results": reason,
                    "budget_breakdown": _build_budget_breakdown(
                        {**state, "budget_results": reason}
                    ),
                    "messages": [AIMessage(content="Budget estimates are unavailable.")],
                }
            if agent_name == "itinerary_agent":
                fallback = _complete_daywise_itinerary(
                    str(state.get("user_query") or "")
                )
                approval_request = (
                    "The itinerary model was unavailable. Review these generic "
                    "fallback suggestions and adjust them before approval."
                )
                return {
                    "itinerary": fallback,
                    "approval_request": approval_request,
                    "messages": [
                        AIMessage(content="A fallback itinerary was prepared for review.")
                    ],
                }
            if agent_name == "final_agent":
                if state.get("approved", False):
                    final_response = (
                        str(state.get("itinerary") or "")
                        + "\n\nFinal formatting was unavailable; the approved "
                        "draft is retained unchanged."
                    )
                    return {
                        "final_response": final_response,
                        "messages": [
                            AIMessage(content="The approved draft was retained.")
                        ],
                    }
                return {
                    "itinerary": str(state.get("itinerary") or ""),
                    "approval_request": (
                        "Revision generation was unavailable. The current draft "
                        "is unchanged; retry with focused feedback."
                    ),
                    "final_response": "",
                    "messages": [AIMessage(content="The draft remains unchanged.")],
                }
            return {"messages": [AIMessage(content=reason)]}

    return run


# =========================
# Build Graph
# =========================
graph = StateGraph(TravelState)

graph.add_node("supervisor", _safe_supervisor_node)
graph.add_node("guardrail_blocked", guardrail_blocked_agent)
graph.add_node("flight_agent", _safe_agent_node("flight_agent"))
graph.add_node("hotel_agent", _safe_agent_node("hotel_agent"))
graph.add_node("weather_agent", _safe_agent_node("weather_agent"))
graph.add_node("budget_agent", _safe_agent_node("budget_agent"))
graph.add_node("itinerary_agent", _safe_agent_node("itinerary_agent"))
graph.add_node("human_approval", human_approval_agent)
graph.add_node("final_agent", _safe_agent_node("final_agent"))

graph.add_edge(START, "supervisor")
graph.add_conditional_edges("supervisor", route_from_supervisor, ROUTE_MAP)

graph.add_conditional_edges(
    "flight_agent", route_after_agent("flight_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "hotel_agent", route_after_agent("hotel_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "weather_agent", route_after_agent("weather_agent"), ROUTE_MAP
)
graph.add_conditional_edges(
    "budget_agent", route_after_agent("budget_agent"), ROUTE_MAP
)

graph.add_edge("itinerary_agent", "human_approval")
graph.add_edge("human_approval", "final_agent")
graph.add_conditional_edges(
    "final_agent",
    route_after_final_agent,
    {
        "human_approval": "human_approval",
        "end": END,
    },
)
graph.add_edge("guardrail_blocked", END)

# =========================
# Checkpointer - prefer PostgreSQL when available, fall back to in-memory.
# =========================
DATABASE_URL = os.getenv("DATABASE_URL")
database_pool = None
checkpointer = MemorySaver()
CHECKPOINTER_BACKEND = "memory"


def _build_postgres_pool(database_url: str | None = None) -> ConnectionPool:
    """Create a PostgreSQL connection pool for checkpoint persistence."""
    database_url = database_url or get_database_url()
    if database_url.startswith("postgres://"):
        database_url = "postgresql://" + database_url[len("postgres://"):]
    return ConnectionPool(
        database_url,
        kwargs={
            "autocommit": True,
            "row_factory": dict_row,
        },
        min_size=1,
        max_size=8,
        open=False,
        check=ConnectionPool.check_connection,
        timeout=30,
        max_idle=300,
        max_lifetime=1800,
        reconnect_timeout=300,
        name="tripmate-postgres",
    )


travel_graph = graph.compile(checkpointer=checkpointer)


def initialize_database() -> None:
    """Prefer PostgreSQL checkpoint storage but fall back to in-memory state when unavailable."""
    global DATABASE_URL, database_pool, checkpointer, travel_graph, CHECKPOINTER_BACKEND

    production = os.getenv("APP_ENV", "development").strip().lower() == "production"

    try:
        database_url = get_database_url()
    except ValueError:
        if production:
            raise RuntimeError(
                "DATABASE_URL is required when APP_ENV=production; "
                "refusing to start with non-persistent in-memory checkpoints."
            ) from None
        logger.warning(
            "DATABASE_URL is not configured; using in-memory checkpoint storage for development."
        )
        checkpointer = MemorySaver()
        CHECKPOINTER_BACKEND = "memory"
        travel_graph = graph.compile(checkpointer=checkpointer)
        return

    DATABASE_URL = database_url
    try:
        database_pool = _build_postgres_pool(database_url)
        database_pool.open(wait=True, timeout=30)
        postgres_checkpointer = PostgresSaver(database_pool)
        postgres_checkpointer.setup()
        checkpointer = postgres_checkpointer
        CHECKPOINTER_BACKEND = "postgresql"
        travel_graph = graph.compile(checkpointer=checkpointer)
        logger.info("PostgreSQL connection pool initialized for LangGraph checkpoints.")
    except Exception as exc:
        logger.error(
            "PostgreSQL checkpoint initialization failed (error_type=%s).",
            type(exc).__name__,
        )
        if database_pool is not None:
            database_pool.close()
        database_pool = None
        if production:
            raise RuntimeError(
                "PostgreSQL checkpoint initialization failed in production; "
                "refusing to start without durable conversation storage."
            ) from None
        checkpointer = MemorySaver()
        CHECKPOINTER_BACKEND = "memory"
        travel_graph = graph.compile(checkpointer=checkpointer)


def get_checkpointer_status() -> dict[str, str]:
    """Expose the active checkpoint backend for health reporting."""
    return {
        "backend": CHECKPOINTER_BACKEND,
        "persistence": "durable" if CHECKPOINTER_BACKEND == "postgresql" else "process_memory",
    }


def close_database_pool() -> None:
    """Close pooled PostgreSQL connections during application shutdown."""
    if database_pool is not None:
        database_pool.close()
        logger.info("PostgreSQL connection pool closed.")
    else:
        logger.info("No PostgreSQL pool active; in-memory checkpoint storage remains in process memory.")


class WorkflowConflictError(RuntimeError):
    """Raised when an approval request does not match a paused draft."""


def _trip_overview_data(result: dict[str, Any]) -> dict[str, Any]:
    query = str(result.get("user_query") or "")
    facts = _extract_request_facts(query)
    constraints = result.get("trip_constraints")
    if not isinstance(constraints, dict):
        constraints = {}

    destination = str(
        facts.get("destination") or constraints.get("destination") or ""
    ).strip()
    origin = str(facts.get("origin") or constraints.get("origin") or "").strip()
    destination_details = _resolve_destination(destination)
    origin_details = _resolve_destination(origin)
    traveler_count = (
        facts.get("traveler_count")
        or facts.get("adult_count")
        or constraints.get("traveler_count")
    )
    traveler_label = facts.get("travelers") or constraints.get("travelers")
    if not traveler_label and traveler_count:
        try:
            count = int(traveler_count)
        except (TypeError, ValueError):
            traveler_label = str(traveler_count)
        else:
            traveler_label = f"{count} {'adult' if count == 1 else 'adults'}"

    return {
        "destination": ", ".join(
            part
            for part in (
                destination_details["city"] or destination,
                destination_details["country"],
            )
            if part
        ),
        "origin": ", ".join(
            part
            for part in (origin_details["city"] or origin, origin_details["country"])
            if part
        ),
        "duration": facts.get("duration") or constraints.get("duration") or "",
        "travelers": str(traveler_label or ""),
        "departure_date": (
            facts.get("departure_date")
            or constraints.get("departure_date")
            or ""
        ),
        "return_date": (
            facts.get("return_date")
            or constraints.get("return_date")
            or ""
        ),
    }


def _budget_table_data(
    budget_text: str,
    user_query: str = "",
    currency_conversion: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = []
    estimated_amounts = []
    estimates, assumptions = _budget_estimates(budget_text)
    for label in _BUDGET_ESTIMATE_LABELS:
        amount = estimates[label]
        assumption = assumptions[label]
        if amount is not None:
            estimated_amounts.append(amount)
        if label == "Flights":
            price_status = (
                "Planning estimate only; live fares are unavailable."
                if amount is not None
                else "No defensible estimate returned; live fares are unavailable."
            )
        elif label == "Accommodation":
            price_status = (
                "Planning estimate only; rates and availability are unverified."
                if amount is not None
                else "No defensible estimate returned; rates and availability are unverified."
            )
        else:
            price_status = (
                "Planning estimate only; no verified provider price."
                if amount is not None
                else "No defensible estimate was returned."
            )
        items.append(
            {
                "category": label,
                "estimated_cost_inr": amount,
                "cost_status": "ESTIMATE" if amount is not None else "Not available",
                "price_status": price_status,
                "assumption": assumption or "",
            }
        )

    missing_categories = [
        item["category"]
        for item in items
        if item["estimated_cost_inr"] is None
    ]
    complete = not missing_categories
    total_status = (
        "Complete planning estimate across all categories; not a live quote. "
        "Flight fares and hotel rates/availability are not verified."
        if complete
        else (
            "Estimated trip total unavailable; missing estimates for "
            + ", ".join(missing_categories)
            + ". Missing expenses are not treated as zero."
        )
    )
    budget_ceiling = _budget_limit_inr(user_query, currency_conversion)
    budget_facts = _extract_request_facts(user_query)
    budget_currency = str(budget_facts.get("budget") or "")
    if budget_ceiling is None and re.search(
        r"\b(?:USD|EUR|GBP)\b",
        budget_currency,
        re.IGNORECASE,
    ):
        budget_ceiling_status = "Foreign-currency ceiling conversion unavailable."
    elif budget_ceiling is not None and isinstance(currency_conversion, dict):
        budget_ceiling_status = (
            f"Reference rate dated {currency_conversion.get('date', 'unavailable')}; "
            "not a live cash or card exchange quote."
        )
    elif budget_ceiling is not None:
        budget_ceiling_status = "User-specified INR spending ceiling; not the trip total."
    else:
        budget_ceiling_status = "No budget ceiling supplied."
    return {
        "items": items,
        "estimated_total_inr": (
            sum(estimated_amounts) if complete else None
        ),
        "total_status": total_status,
        "complete": complete,
        "budget_ceiling_inr": budget_ceiling,
        "budget_ceiling_status": budget_ceiling_status,
        "currency_conversion": currency_conversion,
    }


# =========================
# FastAPI-facing helpers
# =========================
def _interrupt_payload(result: dict[str, Any]) -> dict[str, Any] | None:
    interrupts = result.get("__interrupt__", [])
    if not interrupts:
        return None

    first_interrupt = interrupts[0]
    payload = getattr(first_interrupt, "value", first_interrupt)
    return payload if isinstance(payload, dict) else {"value": payload}


def _serialize_result(
    result: dict[str, Any],
    thread_id: str,
) -> dict[str, Any]:
    messages = result.get("messages", [])
    last_message = messages[-1].content if messages else ""
    final_response = result.get("final_response", "")
    interrupt_payload = _interrupt_payload(result)
    requires_approval = interrupt_payload is not None and not final_response
    draft_itinerary = (
        str(interrupt_payload.get("draft_itinerary") or "").strip()
        if interrupt_payload
        else ""
    )
    if not draft_itinerary:
        draft_itinerary = str(result.get("itinerary") or "").strip()
    if requires_approval and not draft_itinerary:
        draft_itinerary = _complete_daywise_itinerary(
            str(result.get("user_query") or "")
        )

    answer = (
        final_response
        or (
            draft_itinerary
            if interrupt_payload
            else last_message
        )
        or result.get("itinerary", "")
    )

    return {
        "thread_id": thread_id,
        "answer": answer,
        "requires_approval": requires_approval,
        "approval_request": (
            interrupt_payload.get("approval_request", "")
            if requires_approval and interrupt_payload
            else result.get("approval_request", "")
        ),
        "flight_results": result.get("flight_results", ""),
        "flight_details": result.get("flight_details", []),
        "flight_search_url": result.get("flight_search_url", ""),
        "hotel_results": result.get("hotel_results", ""),
        "hotel_details": result.get("hotel_details", []),
        "hotel_search_url": result.get("hotel_search_url", ""),
        "weather_results": result.get("weather_results", ""),
        "weather_forecast": result.get("weather_forecast", []),
        "budget_results": result.get("budget_results", ""),
        "budget_breakdown": result.get("budget_breakdown", ""),
        "budget_currency_conversion": result.get("budget_currency_conversion"),
        "budget_table": _budget_table_data(
            str(result.get("budget_results") or ""),
            str(result.get("user_query") or ""),
            (
                result.get("budget_currency_conversion")
                if isinstance(result.get("budget_currency_conversion"), dict)
                else None
            ),
        ),
        "itinerary": final_response or (
            draft_itinerary
            if requires_approval and interrupt_payload
            else result.get("itinerary", "")
        ),
        "itinerary_table": _itinerary_table_data(
            str(
                final_response
                or (
                    draft_itinerary
                    if requires_approval and interrupt_payload
                    else result.get("itinerary", "")
                )
            )
        ),
        "selected_agents": result.get("selected_agents", []),
        "trip_constraints": result.get("trip_constraints", {}),
        "trip_overview": _trip_overview_data(result),
        "supervisor_reasoning": result.get("supervisor_reasoning", ""),
        "guardrail_allowed": result.get("guardrail_allowed", True),
        "guardrail_reason": result.get("guardrail_reason", ""),
        "approved": result.get("approved"),
        "human_feedback": result.get("human_feedback", ""),
        "llm_calls": result.get("llm_calls", 0),
    }


def run_travel_agent(user_input: str, thread_id: str | None = None):
    """Start a new travel-planning run and pause at human approval."""
    if thread_id:
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = travel_graph.get_state(config)
        if snapshot.values or snapshot.next:
            raise WorkflowConflictError(
                "This thread already contains a travel plan. Start a new draft instead."
            )
    else:
        thread_id = f"user_{uuid.uuid4().hex}"

    config = {"configurable": {"thread_id": thread_id}}

    result = travel_graph.invoke(
        {
            "messages": [HumanMessage(content=user_input)],
            "user_query": user_input,
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": [],
            "trip_constraints": _empty_constraints(),
            "supervisor_reasoning": "",
            "flight_results": "",
            "flight_details": [],
            "flight_search_url": "",
            "hotel_results": "",
            "hotel_details": [],
            "weather_results": "",
            "weather_forecast": [],
            "budget_results": "",
            "budget_breakdown": "",
            "budget_currency_conversion": None,
            "itinerary": "",
            "approval_request": "",
            "approved": False,
            "human_feedback": "",
            "final_response": "",
            "llm_calls": 0,
        },
        config=config,
    )

    return _serialize_result(result, thread_id)


def resume_travel_agent(
    thread_id: str,
    approved: bool,
    feedback: str = "",
):
    """Resume the paused LangGraph thread after human review."""
    if not thread_id:
        raise ValueError("thread_id is required to resume a travel plan.")
    if not approved and not feedback.strip():
        raise ValueError("Revision feedback is required when rejecting a draft.")

    config = {"configurable": {"thread_id": thread_id}}
    snapshot = travel_graph.get_state(config)
    if "human_approval" not in snapshot.next:
        raise WorkflowConflictError(
            "This travel plan is not currently waiting for human approval."
        )

    result = travel_graph.invoke(
        Command(
            resume={
                "approved": approved,
                "feedback": feedback.strip(),
            }
        ),
        config=config,
    )

    return _serialize_result(result, thread_id)
