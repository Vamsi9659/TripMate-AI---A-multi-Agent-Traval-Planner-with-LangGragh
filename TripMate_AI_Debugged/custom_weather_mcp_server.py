import os
import math
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Any

import requests
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
import pycountry


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

mcp = FastMCP("Weather MCP Server")

OPENWEATHER_API_KEY = os.getenv(
    "OPENWEATHER_API_KEY"
)

REQUEST_TIMEOUT_SECONDS = 20


def _get_api_key() -> str:
    if not OPENWEATHER_API_KEY:
        raise RuntimeError(
            "OPENWEATHER_API_KEY is missing "
            "from the project .env file."
        )

    return OPENWEATHER_API_KEY


def _request_json(
    url: str,
    params: dict[str, Any],
) -> Any:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        failed_response = getattr(
            exc,
            "response",
            None,
        )
        status_code = getattr(failed_response, "status_code", None)
        if status_code is None:
            raise RuntimeError(
                "OpenWeather could not be reached. Check network access and retry."
            ) from None
        if status_code == 404:
            raise RuntimeError(
                "OpenWeather could not find the requested city or country (HTTP 404)."
            ) from None
        if status_code in {401, 403}:
            raise RuntimeError(
                f"OpenWeather rejected the configured API key (HTTP {status_code})."
            ) from None
        if status_code == 429:
            raise RuntimeError(
                "OpenWeather rate limit reached (HTTP 429)."
            ) from None
        raise RuntimeError(
            f"OpenWeather request failed (HTTP {status_code})."
        ) from None

    try:
        data = response.json()
    except (ValueError, requests.RequestException):
        raise RuntimeError("OpenWeather returned invalid JSON.") from None

    return data


def _geocode_city(city: str) -> dict[str, Any]:
    query = city.strip()
    if not query:
        raise ValueError("city cannot be empty")

    results = _request_json(
        "https://api.openweathermap.org/geo/1.0/direct",
        {
            "q": query,
            "limit": 5,
            "appid": _get_api_key(),
        },
    )
    if not isinstance(results, list):
        raise RuntimeError("OpenWeather returned an invalid geocoding response.")
    if not results:
        raise RuntimeError("OpenWeather could not find the requested city.")

    requested_name, _, qualifier = query.partition(",")
    exact_matches = [
        result
        for result in results
        if isinstance(result, dict)
        and str(result.get("name", "")).casefold() == requested_name.strip().casefold()
    ]
    candidates = exact_matches or [
        result for result in results if isinstance(result, dict)
    ]
    if qualifier and len(candidates) > 1:
        qualifier = qualifier.strip().casefold()
        try:
            country = pycountry.countries.lookup(qualifier)
        except LookupError:
            country = None
        candidates = [
            result
            for result in candidates
            if str(result.get("country", "")).casefold() == qualifier
            or (
                country is not None
                and str(result.get("country", "")).upper() == country.alpha_2
            )
            or str(result.get("state", "")).casefold() == qualifier
        ]

    if len(candidates) != 1:
        raise RuntimeError(
            "OpenWeather found an ambiguous city. Please include its country or state."
        )

    location = candidates[0]
    latitude = location.get("lat")
    longitude = location.get("lon")
    if (
        isinstance(latitude, bool)
        or not isinstance(latitude, (int, float))
        or not math.isfinite(float(latitude))
        or not -90 <= latitude <= 90
        or isinstance(longitude, bool)
        or not isinstance(longitude, (int, float))
        or not math.isfinite(float(longitude))
        or not -180 <= longitude <= 180
    ):
        raise RuntimeError("OpenWeather returned invalid location coordinates.")

    return {
        "name": str(location.get("name") or requested_name.strip()),
        "lat": float(latitude),
        "lon": float(longitude),
    }


def _finite_number(value: Any, label: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise RuntimeError(f"OpenWeather returned an invalid {label} value.")
    return float(value)


def _humidity(value: Any) -> float:
    humidity = _finite_number(value, "humidity")
    if not 0 <= humidity <= 100:
        raise RuntimeError("OpenWeather returned an out-of-range humidity value.")
    return humidity


def _rain_probability_percent(value: Any) -> float:
    probability = _finite_number(value, "precipitation probability")
    if not 0 <= probability <= 1:
        raise RuntimeError(
            "OpenWeather returned an out-of-range precipitation probability."
        )
    return probability * 100


@mcp.tool()
def get_current_weather(
    city: str,
) -> dict[str, Any]:
    """Return the current weather for a city."""

    city = city.strip()

    if not city:
        raise ValueError(
            "city cannot be empty"
        )

    location = _geocode_city(city)
    data = _request_json(
        "https://api.openweathermap.org/data/2.5/weather",
        {
            "lat": location["lat"],
            "lon": location["lon"],
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    try:
        current = data["main"]
        condition = data["weather"][0]["description"]
        wind_speed = _finite_number(data["wind"]["speed"], "wind speed")
        return {
            "city": data.get("name") or location["name"],
            "temperature_c": _finite_number(current["temp"], "temperature"),
            "feels_like_c": _finite_number(current["feels_like"], "feels-like temperature"),
            "humidity": _humidity(current["humidity"]),
            "condition": condition,
            "wind_speed": wind_speed,
        }
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "OpenWeather returned an incomplete current-weather response."
        ) from exc


@mcp.tool()
def get_forecast(
    city: str,
) -> dict[str, Any]:
    """
    Return available three-hour forecast entries for up to five days.
    """

    city = city.strip()

    if not city:
        raise ValueError(
            "city cannot be empty"
        )

    location = _geocode_city(city)
    data = _request_json(
        "https://api.openweathermap.org/data/2.5/forecast",
        {
            "lat": location["lat"],
            "lon": location["lon"],
            "appid": _get_api_key(),
            "units": "metric",
        },
    )

    try:
        if not isinstance(data, dict) or not isinstance(data.get("list"), list):
            raise RuntimeError("OpenWeather returned an incomplete forecast response.")
        city_data = data["city"]
        timezone_offset = _finite_number(city_data["timezone"], "city timezone")
        forecast = [
            {
                "datetime": (
                    datetime.fromtimestamp(item["dt"], timezone.utc)
                    .astimezone(timezone(timedelta(seconds=timezone_offset)))
                    .replace(tzinfo=None)
                    .isoformat(sep=" ")
                ),
                "temperature_c": _finite_number(item["main"]["temp"], "temperature"),
                "temperature_min_c": _finite_number(
                    item["main"]["temp_min"], "minimum temperature"
                ),
                "temperature_max_c": _finite_number(
                    item["main"]["temp_max"], "maximum temperature"
                ),
                "humidity": _humidity(item["main"]["humidity"]),
                "rain_probability_percent": (
                    _rain_probability_percent(item["pop"])
                    if item.get("pop") is not None
                    else None
                ),
                "condition": item["weather"][0]["description"],
            }
            for item in data["list"]
        ]
        city = str(city_data["name"])
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "OpenWeather returned an incomplete forecast response."
        ) from exc

    return {
        "city": city,
        "forecast": forecast,
    }


if __name__ == "__main__":
    # mcp_client.py launches this as a stdio subprocess.
    mcp.run(
        transport="stdio",
    )