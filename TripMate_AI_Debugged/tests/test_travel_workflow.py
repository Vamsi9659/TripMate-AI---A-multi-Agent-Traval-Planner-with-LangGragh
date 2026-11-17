import asyncio
import os
import pytest
from datetime import date as real_date, timedelta as real_timedelta
from datetime import datetime, timezone
from types import SimpleNamespace

_original_groq_key = os.environ.get("GROQ_API_KEY")
os.environ["GROQ_API_KEY"] = "unit-test-key"
import backend  # noqa: E402
import mcp_client  # noqa: E402
import custom_weather_mcp_server as weather_server  # noqa: E402

if _original_groq_key is None:
    os.environ.pop("GROQ_API_KEY", None)
else:
    os.environ["GROQ_API_KEY"] = _original_groq_key

from langgraph.checkpoint.memory import MemorySaver  # noqa: E402


TRIP_REQUEST = (
    "Plan a 3-day budget-friendly trip from Hyderabad, India to Bangkok, "
    "Thailand for 2 people from December 12 to December 14, 2026."
)


def test_example_request_dates_and_destination_resolve():
    facts = backend._extract_request_facts(TRIP_REQUEST)

    assert facts["destination"] == "Bangkok"
    assert facts["origin"] == "Hyderabad"
    assert facts["departure_date"] == "2026-12-12"
    assert facts["return_date"] == "2026-12-14"
    assert facts["traveler_count"] == 2
    assert facts["duration"] == "3 days"
    assert backend._resolve_destination("Thailand")["city"] == "Bangkok"
    assert backend._resolve_destination("Thailand")["iata"] == "BKK"
    assert backend._resolve_destination("Hyderabad")["iata"] == "HYD"
    assert backend._resolve_destination("Phuket")["iata"] == "HKT"
    assert backend._resolve_destination("Singapore")["iata"] == "SIN"


def test_party_size_includes_adults_and_children():
    facts = backend._extract_request_facts(
        "Plan a trip to Paris from Delhi for 2 adults and 1 child"
    )

    assert facts["adult_count"] == 2
    assert facts["child_count"] == 1
    assert facts["traveler_count"] == 3
    assert facts["travelers"] == "2 adults and 1 child"


def test_model_json_parser_accepts_json_with_wrapping_prose():
    parsed = backend._json_from_llm(
        'Supervisor result:\n```json\n{"selected_agents": ["flight_agent"]}\n```'
    )

    assert parsed == {"selected_agents": ["flight_agent"]}


def test_aviation_mcp_text_wrapper_and_provider_errors_are_normalized():
    rows = backend._aviation_data_rows(
        [
            {
                "type": "text",
                "text": '{"ok": true, "data": [{"arrival_iata": "BKK"}]}',
            }
        ]
    )
    error = backend._aviation_response_error(
        [
            {
                "type": "text",
                "text": (
                    '{"ok": false, "error": '
                    '"api_error (rate_limit_reached): subscription limit"}'
                ),
            }
        ]
    )

    assert rows == [{"arrival_iata": "BKK"}]
    assert error is not None and "rate_limit_reached" in error
    assert "rate limit reached" in backend._integration_failure(
        "AviationStack",
        RuntimeError(error),
    )


def test_aviationstack_schedule_adapter_preserves_http_error_status(monkeypatch, caplog):
    captured = {}
    api_key = "unit-test-aviation-key"
    requested_date = (real_date.today() + real_timedelta(days=2)).isoformat()

    class FakeResponse:
        status_code = 429

        @staticmethod
        def json():
            return {
                "error": (
                    "The upstream flight data provider is rate limiting requests."
                )
            }

    def fake_get(url, params, timeout):
        captured.update(url=url, params=params, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(mcp_client, "AVIATION_STACK_API_KEY", api_key)
    monkeypatch.setattr(mcp_client.requests, "get", fake_get)

    result = asyncio.run(
        mcp_client.aviation_mcp_call(
            "future_flights_arrival_departure_schedule",
            {
                "airport_iata_code": "HYD",
                "schedule_type": "departure",
                "date": requested_date,
                "number_of_flights": 100,
            },
        )
    )

    assert captured["url"] == mcp_client.AVIATIONSTACK_API_URL
    assert captured["params"] == {
        "access_key": api_key,
        "iataCode": "HYD",
        "type": "departure",
        "date": requested_date,
        "limit": 100,
    }
    assert result["ok"] is False
    assert result["status_code"] == 429
    assert "HTTP 429" in backend._aviation_response_error(result)
    assert "http_status=429" in caplog.text


def test_hotel_search_is_destination_filtered_and_contextual(monkeypatch):
    observed_queries = []

    async def fake_tavily_search(query):
        observed_queries.append(query)
        return {
            "results": [
                {
                    "title": "Siam Riverside Hotel Bangkok",
                    "content": "A hotel near the river in Bangkok, Thailand.",
                    "url": "https://siam.example/hotel",
                },
                {
                    "title": "Paris Grand Hotel",
                    "content": "A hotel in Paris, France.",
                    "url": "https://paris.example/hotel",
                },
                {
                    "title": "Bangkok View Hotel",
                    "content": "A property located in Paris, France; Bangkok appears in this listing title.",
                    "url": "https://wrong-country.example/hotel",
                },
            ]
        }

    monkeypatch.setattr(backend, "tavily_mcp_search", fake_tavily_search)
    state = backend.hotel_agent(
        {
            "user_query": TRIP_REQUEST,
            "trip_constraints": {"destination": "Bangkok"},
        }
    )

    assert "2026-12-12" in observed_queries[0]
    assert "2026-12-14" in observed_queries[0]
    assert "guests: 2" in observed_queries[0]
    assert "budget: budget-friendly" in observed_queries[0]
    assert "Siam Riverside Hotel Bangkok" in state["hotel_results"]
    assert "Location: Bangkok, Thailand" in state["hotel_results"]
    assert "siam.example/hotel" in state["hotel_results"]
    assert "Paris Grand Hotel" not in state["hotel_results"]
    assert "Bangkok View Hotel" not in state["hotel_results"]
    assert "availability: UNAVAILABLE" in state["hotel_results"]


def test_budget_does_not_claim_complete_total_without_fares_or_rooms():
    breakdown = backend._build_budget_breakdown(
        {
            "user_query": TRIP_REQUEST,
            "flight_results": "UNAVAILABLE live ticket prices.",
            "hotel_results": "LIVE: Tavily observations; rates unavailable.",
            "budget_results": (
                "Food: ESTIMATE - not a live quote: INR 1200; "
                "assumption: three days of local meals for two travelers\n"
                "Local transport: ESTIMATE - not a live quote: INR 1000; "
                "assumption: local transit for two travelers\n"
                "Attractions: ESTIMATE - not a live quote: INR 900; "
                "assumption: modest entry allowances\n"
                "Shopping: ESTIMATE - not a live quote: INR 200; "
                "assumption: small optional shopping allowance\n"
                "Miscellaneous: ESTIMATE - not a live quote: INR 300; "
                "assumption: small incidental allowance\n"
                "Emergency allowance: ESTIMATE - not a live quote: INR 340; "
                "assumption: contingency reserve"
            ),
        }
    )

    assert "- **Flights:** UNAVAILABLE" in breakdown
    assert "- **Accommodation:** UNAVAILABLE" in breakdown
    assert (
        "- **Planning-estimate subtotal (not a trip total):** "
        "INR 3,940.00"
    ) in breakdown
    assert "Estimated trip total:** UNAVAILABLE" in breakdown
    assert "Missing expenses are not treated as zero." in breakdown


def test_weather_table_only_uses_forecast_samples_for_travel_dates():
    rows = backend._weather_forecast_table_rows(
        {
            "city": "Bangkok",
            "forecast": [
                {
                    "datetime": "2026-12-12 12:00:00",
                    "temperature_c": 30,
                    "temperature_min_c": 23,
                    "temperature_max_c": 33,
                    "condition": "partly cloudy",
                    "humidity": 70,
                    "rain_probability_percent": 20,
                }
            ],
        },
        "2026-12-12",
        "2026-12-14",
        "Bangkok",
    )

    assert rows[0] == {
        "date": "2026-12-12",
        "location": "Bangkok",
        "temperature_c": 30.0,
        "temperature_min_c": 23.0,
        "temperature_max_c": 33.0,
        "condition": "partly cloudy",
        "rain_probability_percent": 20.0,
        "humidity": 70,
        "recommendation": "Recheck the forecast closer to departure.",
        "status": "Forecast available",
    }
    assert [row["status"] for row in rows] == [
        "Forecast available",
        "Forecast unavailable",
        "Forecast unavailable",
    ]


def test_weather_table_aggregates_daily_ranges_humidity_and_rain():
    rows = backend._weather_forecast_table_rows(
        {
            "city": "Bangkok",
            "forecast": [
                {
                    "datetime": "2026-12-12 06:00:00",
                    "temperature_c": 24,
                    "temperature_min_c": 22,
                    "temperature_max_c": 25,
                    "humidity": 60,
                    "rain_probability_percent": 20,
                    "condition": "cloudy",
                },
                {
                    "datetime": "2026-12-12 15:00:00",
                    "temperature_c": 31,
                    "temperature_min_c": 29,
                    "temperature_max_c": 35,
                    "humidity": 80,
                    "rain_probability_percent": 70,
                    "condition": "rain",
                },
            ],
        },
        "2026-12-12",
        None,
    )

    assert rows[0]["temperature_min_c"] == 22
    assert rows[0]["temperature_max_c"] == 35
    assert rows[0]["humidity"] == 70
    assert rows[0]["rain_probability_percent"] == 70
    assert "rain protection" in rows[0]["recommendation"]


def test_openweather_geocodes_and_returns_validated_forecast_fields(monkeypatch):
    calls = []
    timestamp = int(
        datetime(2026, 12, 12, tzinfo=timezone.utc).timestamp()
    )
    monkeypatch.setattr(weather_server, "OPENWEATHER_API_KEY", "unit-test-key")

    def fake_request_json(url, params):
        calls.append((url, params))
        if url.endswith("/geo/1.0/direct"):
            return [{"name": "Bangkok", "lat": 13.75, "lon": 100.5, "country": "TH"}]
        return {
            "city": {"name": "Bangkok", "timezone": 25200},
            "list": [
                {
                    "dt": timestamp,
                    "main": {
                        "temp": 29,
                        "temp_min": 24,
                        "temp_max": 32,
                        "humidity": 75,
                    },
                    "pop": 0.4,
                    "weather": [{"description": "partly cloudy"}],
                }
            ],
        }

    monkeypatch.setattr(weather_server, "_request_json", fake_request_json)
    result = weather_server.get_forecast("Bangkok, Thailand")

    assert calls[0][0].endswith("/geo/1.0/direct")
    assert calls[0][1]["q"] == "Bangkok, Thailand"
    assert calls[1][0].endswith("/data/2.5/forecast")
    assert calls[1][1]["lat"] == 13.75
    assert calls[1][1]["lon"] == 100.5
    assert result["forecast"][0] == {
        "datetime": "2026-12-12 07:00:00",
        "temperature_c": 29.0,
        "temperature_min_c": 24.0,
        "temperature_max_c": 32.0,
        "humidity": 75.0,
        "rain_probability_percent": 40.0,
        "condition": "partly cloudy",
    }


def test_openweather_rejects_ambiguous_city_and_invalid_coordinates(monkeypatch):
    monkeypatch.setattr(weather_server, "OPENWEATHER_API_KEY", "unit-test-key")
    monkeypatch.setattr(
        weather_server,
        "_request_json",
        lambda *_args, **_kwargs: [
            {"name": "Springfield", "lat": 39.8, "lon": -89.6, "country": "US"},
            {"name": "Springfield", "lat": 44.0, "lon": -70.0, "country": "US"},
        ],
    )
    with pytest.raises(RuntimeError, match="ambiguous city"):
        weather_server._geocode_city("Springfield")

    monkeypatch.setattr(
        weather_server,
        "_request_json",
        lambda *_args, **_kwargs: [
            {"name": "Nowhere", "lat": 91, "lon": 0, "country": "XX"}
        ],
    )
    with pytest.raises(RuntimeError, match="invalid location coordinates"):
        weather_server._geocode_city("Nowhere")


def test_complete_budget_estimate_includes_all_categories_and_fx_reference(monkeypatch):
    query = "Plan a trip to Paris for two travelers with a budget of USD 1000"

    class FakeResponse:
        @staticmethod
        def raise_for_status():
            return None

        @staticmethod
        def json():
            return {
                "date": "2026-11-13",
                "base": "USD",
                "quote": "INR",
                "rate": 83.5,
            }

    captured = {}

    def fake_get(url, timeout):
        captured.update(url=url, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(backend.requests, "get", fake_get)
    conversion = backend._budget_currency_conversion(query)
    assert conversion["amount_inr"] == 83500
    assert captured["url"] == "https://api.frankfurter.dev/v2/rate/usd/inr"

    estimates = "\n".join(
        (
            "Flights: ESTIMATE — not a live quote: INR 12000; assumption: all travelers round trip",
            "Accommodation: ESTIMATE — not a live quote: INR 9000; assumption: nights for both travelers",
            "Food: ESTIMATE — not a live quote: INR 1200; assumption: all travelers for all days",
            "Local transport: ESTIMATE — not a live quote: INR 1000; assumption: daily transit",
            "Attractions: ESTIMATE — not a live quote: INR 900; assumption: entry allowance",
            "Shopping: ESTIMATE — not a live quote: INR 200; assumption: optional purchases",
            "Miscellaneous: ESTIMATE — not a live quote: INR 300; assumption: incidentals",
            "Emergency allowance: ESTIMATE — not a live quote: INR 2500; assumption: contingency reserve",
        )
    )
    table = backend._budget_table_data(estimates, query, conversion)
    assert table["complete"] is True
    assert table["estimated_total_inr"] == 27100
    assert table["budget_ceiling_inr"] == 83500
    assert "2026-11-13" in table["budget_ceiling_status"]
    breakdown = backend._build_budget_breakdown(
        {
            "user_query": query,
            "budget_results": estimates,
            "budget_currency_conversion": conversion,
        }
    )
    assert "1,000.00 USD" in breakdown
    assert "INR 83,500.00" in breakdown
    assert "2026-11-13" in breakdown
    assert "Estimated trip total (all requested categories):** INR 27,100.00" in breakdown

    estimates_without_emergency = "\n".join(
        line for line in estimates.splitlines()
        if not line.startswith("Emergency allowance:")
    )
    derived_table = backend._budget_table_data(estimates_without_emergency)
    assert derived_table["complete"] is True
    assert derived_table["items"][-1]["estimated_cost_inr"] == 2460
    assert "10% contingency" in derived_table["items"][-1]["assumption"]


def test_itinerary_contract_requires_fields_and_rejects_duplicate_places():
    rows = []
    lines = []
    for day in range(1, 4):
        lines.append(f"### Day {day}")
        for index, period in enumerate(("Morning", "Afternoon", "Evening")):
            place = f"Unique Place {day}-{index}"
            meal = ("breakfast", "lunch", "dinner")[index]
            lines.append(
                f"- {period} | Time: 08:00-11:00 (approximate local time) | "
                f"Place: {place} | Activity: {meal.title()} and sightseeing | Duration: 2 hours | "
                "Estimated cost: ESTIMATE INR 500 for all travelers | Transportation: walk"
            )
    valid_schedule = "\n".join(lines)
    normalized = backend._generated_itinerary_body(valid_schedule, TRIP_REQUEST)
    rows = backend._itinerary_table_data(normalized)
    assert len(rows) == 9
    assert rows[0]["estimated_cost_inr"] == 500
    assert rows[0]["transportation"] == "walk"

    duplicate_schedule = valid_schedule.replace("Unique Place 2-0", "Unique Place 1-0")
    assert backend._generated_itinerary_body(duplicate_schedule, TRIP_REQUEST) == ""


def test_fallback_itinerary_has_structured_rows_for_every_requested_day():
    fallback = backend._complete_daywise_itinerary(
        "Plan a 5-day trip to Kyoto"
    )
    rows = backend._itinerary_table_data(fallback)

    assert len(rows) == 15
    assert {row["day"] for row in rows} == {1, 2, 3, 4, 5}
    assert all(row["time"] and row["duration"] for row in rows)
    assert all(row["estimated_cost_inr"] is None for row in rows)


def test_serialized_dashboard_tables_use_actual_agent_data():
    serialized = backend._serialize_result(
        {
            "user_query": TRIP_REQUEST,
            "trip_constraints": {
                "origin": "Hyderabad",
                "destination": "Bangkok",
                "duration": "3 days",
                "travelers": "2 travelers",
                "departure_date": "2026-12-12",
                "return_date": "2026-12-14",
            },
            "itinerary": "### Day 1\n- **Morning:** Explore the riverside.",
            "flight_details": [
                {
                    "airline": "Example Airways",
                    "flight_number": "EA123",
                    "departure": "2026-12-12T10:00:00",
                    "arrival": "2026-12-12T16:00:00",
                    "status": "Schedule listed by AviationStack",
                    "price_inr": None,
                }
            ],
            "flight_search_url": "https://www.google.com/travel/flights?q=HYD+BKK",
            "hotel_details": [
                {
                    "name": "River Hotel Bangkok",
                    "location": "Bangkok, Thailand",
                    "description": "Located by the river.",
                    "url": "https://hotel.example/property",
                }
            ],
            "weather_forecast": [
                {
                    "date": "2026-12-12",
                    "location": "Bangkok",
                    "temperature_c": 30,
                    "condition": "partly cloudy",
                    "status": "Forecast available",
                }
            ],
            "budget_results": (
                "Food: ESTIMATE — not a live quote: INR 1200; "
                "assumption: three days of local meals\n"
                "Flights: ESTIMATE — not a live quote: INR 12000; "
                "assumption: round trip allowance\n"
                "Accommodation: ESTIMATE — not a live quote: INR 9000; "
                "assumption: three nights\n"
                "Local transport: ESTIMATE — not a live quote: INR 1000; "
                "assumption: local transit\n"
                "Attractions: ESTIMATE — not a live quote: INR 900; "
                "assumption: modest entry allowances\n"
                "Shopping: ESTIMATE — not a live quote: INR 200; "
                "assumption: optional shopping\n"
                "Miscellaneous: ESTIMATE — not a live quote: INR 300; "
                "assumption: incidental allowance\n"
                "Emergency allowance: ESTIMATE — not a live quote: INR 2560; "
                "assumption: contingency reserve"
            ),
            "messages": [],
        },
        "dashboard-test",
    )

    assert serialized["trip_overview"] == {
        "destination": "Bangkok, Thailand",
        "origin": "Hyderabad, India",
        "duration": "3 days",
        "travelers": "2 travelers",
        "departure_date": "2026-12-12",
        "return_date": "2026-12-14",
    }
    assert serialized["flight_details"][0]["flight_number"] == "EA123"
    assert serialized["hotel_details"][0]["url"] == "https://hotel.example/property"
    assert serialized["weather_forecast"][0]["temperature_c"] == 30
    assert serialized["budget_table"]["estimated_total_inr"] == 27160
    assert serialized["budget_table"]["complete"] is True
    assert [
        item["estimated_cost_inr"]
        for item in serialized["budget_table"]["items"]
    ] == [12000, 9000, 1200, 1000, 900, 200, 300, 2560]


def test_serialization_recovers_draft_when_interrupt_payload_is_empty():
    serialized = backend._serialize_result(
        {
            "user_query": TRIP_REQUEST,
            "itinerary": "### Day 1\n- **Morning:** Explore the riverside.",
            "__interrupt__": [
                SimpleNamespace(
                    value={
                        "draft_itinerary": "",
                        "approval_request": "Review this draft.",
                    }
                )
            ],
            "messages": [],
        },
        "empty-interrupt-draft",
    )

    assert serialized["requires_approval"] is True
    assert serialized["itinerary"] == "### Day 1\n- **Morning:** Explore the riverside."
    assert serialized["answer"] == serialized["itinerary"]


def test_serialization_generates_fallback_when_draft_is_missing():
    serialized = backend._serialize_result(
        {
            "user_query": TRIP_REQUEST,
            "__interrupt__": [
                SimpleNamespace(
                    value={
                        "draft_itinerary": "",
                        "approval_request": "Review this draft.",
                    }
                )
            ],
            "messages": [],
        },
        "missing-interrupt-draft",
    )

    assert serialized["requires_approval"] is True
    assert "### Day 1" in serialized["itinerary"]
    assert "### Day 3" in serialized["itinerary"]
    assert serialized["answer"] == serialized["itinerary"]


def test_postgres_pool_uses_tls_and_connection_pool_settings(monkeypatch):
    captured = {}

    class FakePool:
        check_connection = staticmethod(lambda connection: connection)

        def __init__(self, conninfo, **kwargs):
            captured["conninfo"] = conninfo
            captured["kwargs"] = kwargs

    monkeypatch.setenv(
        "DATABASE_URL",
        "postgres://tripmate:unit-test-password@database.example.invalid/tripmate",
    )
    monkeypatch.setattr(backend, "ConnectionPool", FakePool)

    backend._build_postgres_pool()

    assert captured["conninfo"].startswith("postgresql://")
    assert captured["conninfo"].endswith("?sslmode=require")
    assert captured["kwargs"]["min_size"] == 1
    assert captured["kwargs"]["max_size"] == 8
    assert captured["kwargs"]["open"] is False
    assert captured["kwargs"]["kwargs"]["autocommit"] is True


def test_supervisor_runs_all_agents_for_requested_trip_plan(monkeypatch):
    calls = []

    class FakeLLM:
        def invoke(self, messages, **kwargs):
            if kwargs.get("max_tokens") is None:
                raise AssertionError("The workflow must bound model responses.")
            prompt = str(messages[-1].content)
            if "Return exactly eight lines" in prompt:
                return SimpleNamespace(
                    content=(
                        "Flights: ESTIMATE — not a live quote: INR 12000; "
                        "assumption: broad allowance for two travelers\n"
                        "Accommodation: ESTIMATE — not a live quote: INR 9000; "
                        "assumption: three nights in a budget stay\n"
                        "Food: ESTIMATE — not a live quote: INR 1200; "
                        "assumption: three days of local meals for two travelers\n"
                        "Local transport: ESTIMATE — not a live quote: INR 1000; "
                        "assumption: local transit for two travelers\n"
                        "Attractions: ESTIMATE — not a live quote: INR 900; "
                        "assumption: modest entry allowances\n"
                        "Shopping: ESTIMATE — not a live quote: INR 200; "
                        "assumption: small optional shopping allowance\n"
                        "Miscellaneous: ESTIMATE — not a live quote: INR 300; "
                        "assumption: small incidental allowance\n"
                        "Emergency allowance: ESTIMATE — not a live quote: INR 340; "
                        "assumption: contingency reserve"
                    )
                )
            if "Create a useful itinerary" in prompt:
                return SimpleNamespace(
                    content=(
                        "### Day 1\n"
                        "- Morning | Time: 08:00-11:00 (approximate local time) | Place: Riverside Walk | Activity: Breakfast and river sightseeing | Duration: 2 hours | Estimated cost: ESTIMATE INR 300 for all travelers | Transportation: walk\n"
                        "- Afternoon | Time: 12:00-15:00 (approximate local time) | Place: Bangkok Art and Culture Centre | Activity: Lunch and art visit | Duration: 3 hours | Estimated cost: ESTIMATE INR 500 for all travelers | Transportation: BTS\n"
                        "- Evening | Time: 17:00-20:00 (approximate local time) | Place: Yaowarat Road | Activity: Dinner and evening walk | Duration: 2 hours | Estimated cost: ESTIMATE INR 700 for all travelers | Transportation: MRT\n"
                        "### Day 2\n"
                        "- Morning | Time: 08:00-11:00 (approximate local time) | Place: Grand Palace | Activity: Breakfast and historic-site visit | Duration: 3 hours | Estimated cost: ESTIMATE INR 1000 for all travelers | Transportation: taxi\n"
                        "- Afternoon | Time: 12:00-15:00 (approximate local time) | Place: Chatuchak Market | Activity: Lunch and market visit | Duration: 3 hours | Estimated cost: ESTIMATE INR 800 for all travelers | Transportation: BTS\n"
                        "- Evening | Time: 17:00-20:00 (approximate local time) | Place: Lumphini Park | Activity: Dinner and park walk | Duration: 2 hours | Estimated cost: ESTIMATE INR 600 for all travelers | Transportation: MRT\n"
                        "### Day 3\n"
                        "- Morning | Time: 08:00-11:00 (approximate local time) | Place: Jim Thompson House | Activity: Breakfast and museum visit | Duration: 2 hours | Estimated cost: ESTIMATE INR 900 for all travelers | Transportation: BTS\n"
                        "- Afternoon | Time: 12:00-15:00 (approximate local time) | Place: Wat Pho | Activity: Lunch and temple visit | Duration: 3 hours | Estimated cost: ESTIMATE INR 900 for all travelers | Transportation: ferry\n"
                        "- Evening | Time: 17:00-20:00 (approximate local time) | Place: Asiatique The Riverfront | Activity: Dinner and waterfront visit | Duration: 2 hours | Estimated cost: ESTIMATE INR 1000 for all travelers | Transportation: boat"
                    )
                )
            raise AssertionError("Unexpected LLM prompt in mocked workflow.")

    def fake_supervisor_text(system_prompt, _user_prompt, _max_tokens=120):
        if not isinstance(_user_prompt, str) or _max_tokens < 1:
            raise AssertionError("Supervisor prompts must be bounded and textual.")
        if "input guardrail" in system_prompt.casefold():
            return '{"allowed": true, "reason": ""}'
        return (
            '{"selected_agents": ["itinerary_agent"], '
            '"trip_constraints": {}, "reasoning": "Plan the trip."}'
        )

    async def fake_aviation(tool_name, tool_args=None):
        calls.append(("flight", tool_name, tool_args))
        return {
            "data": [
                {
                    "flight_status": "scheduled",
                    "arrival_iata": "BKK",
                    "departure": {"iata": "HYD", "scheduled": "2026-12-12T10:00:00"},
                    "arrival": {"iata": "BKK", "scheduled": "2026-12-12T16:00:00"},
                }
            ]
        }

    async def fake_tavily(query):
        calls.append(("hotel", query))
        return {
            "results": [
                {
                    "title": "Siam Riverside Hotel Bangkok",
                    "content": "Located by the river in Bangkok, Thailand.",
                    "url": "https://siam.example/hotel",
                }
            ]
        }

    async def fake_current_weather(city):
        calls.append(("weather", city))
        return {
            "city": city,
            "temperature_c": 31,
            "feels_like_c": 34,
            "humidity": 70,
            "condition": "partly cloudy",
            "wind_speed": 2.5,
        }

    async def fake_forecast(city):
        calls.append(("forecast", city))
        return {
            "city": city,
            "forecast": [
                {
                    "datetime": f"2026-12-{day} 12:00:00",
                    "temperature_c": 30,
                    "temperature_min_c": 24,
                    "temperature_max_c": 34,
                    "humidity": 75,
                    "rain_probability_percent": 30,
                    "condition": "partly cloudy",
                }
                for day in (12, 13, 14)
            ],
        }

    class FixedDate(real_date):
        @classmethod
        def today(cls):
            return cls(2026, 11, 13)

    monkeypatch.setattr(backend, "date", FixedDate)
    monkeypatch.setattr(backend, "_llm_text", fake_supervisor_text)
    monkeypatch.setattr(backend, "llm", FakeLLM())
    monkeypatch.setattr(backend, "aviation_mcp_call", fake_aviation)
    monkeypatch.setattr(backend, "tavily_mcp_search", fake_tavily)
    monkeypatch.setattr(backend, "weather_mcp_search", fake_current_weather)
    monkeypatch.setattr(backend, "forecast_mcp_search", fake_forecast)
    monkeypatch.setattr(
        backend,
        "travel_graph",
        backend.graph.compile(checkpointer=MemorySaver()),
    )

    result = backend.run_travel_agent(
        TRIP_REQUEST,
        thread_id="mocked-tripmate-integration-test",
    )

    assert result["selected_agents"] == backend.AGENT_ORDER
    assert result["requires_approval"] is True
    assert "LIVE: AviationStack future departure schedule" in result["flight_results"]
    assert "Live ticket prices are not provided" in result["flight_results"]
    assert "BKK" in result["flight_results"]
    assert "google.com/travel/flights" in result["flight_results"]
    assert result["flight_details"][0]["status"] == "scheduled"
    assert result["flight_details"][0]["price_inr"] is None
    assert "Siam Riverside Hotel Bangkok" in result["hotel_results"]
    assert result["hotel_details"][0]["name"] == "Siam Riverside Hotel Bangkok"
    assert "google.com/search?q=" in result["hotel_search_url"]
    assert "LIVE current conditions" in result["weather_results"]
    assert "2026-12-12: 24–34 °C" in result["weather_results"]
    assert result["weather_forecast"][0]["status"] == "Forecast available"
    assert result["weather_forecast"][0]["rain_probability_percent"] == 30
    assert len(result["itinerary_table"]) == 9
    assert result["itinerary_table"][0]["duration"] == "2 hours"
    assert calls[0][2]["airport_iata_code"] == "HYD"
    assert calls[0][2]["date"] == "2026-12-12"
    assert calls[1][2]["airport_iata_code"] == "BKK"
    assert calls[1][2]["date"] == "2026-12-14"
    assert "### Day 1" in result["answer"]
    assert "### Day 3" in result["answer"]
    assert "Estimated trip total (all requested categories)" in result["budget_breakdown"]
    assert "not a provider quote" in result["budget_breakdown"]
    assert [call[0] for call in calls] == [
        "flight",
        "flight",
        "hotel",
        "weather",
        "forecast",
    ]


def test_agent_failure_is_reported_without_stopping_downstream_agents(monkeypatch):
    def fake_supervisor(_state):
        constraints = backend._extract_request_facts(TRIP_REQUEST)
        return {
            "guardrail_allowed": True,
            "guardrail_reason": "",
            "selected_agents": backend.AGENT_ORDER.copy(),
            "trip_constraints": constraints,
            "supervisor_reasoning": "Complete travel plan.",
        }

    def fail_flight(_state):
        raise RuntimeError("upstream error")

    monkeypatch.setattr(backend, "supervisor_agent", fake_supervisor)
    monkeypatch.setattr(backend, "flight_agent", fail_flight)
    monkeypatch.setattr(
        backend,
        "hotel_agent",
        lambda _state: {"hotel_results": "LIVE: hotel observations."},
    )
    monkeypatch.setattr(
        backend,
        "weather_agent",
        lambda _state: {"weather_results": "UNAVAILABLE: forecast unavailable."},
    )
    monkeypatch.setattr(
        backend,
        "budget_agent",
        lambda _state: {"budget_results": "UNAVAILABLE: estimate unavailable."},
    )
    monkeypatch.setattr(
        backend,
        "itinerary_agent",
        lambda _state: {
            "itinerary": (
                "### Day 1\n"
                "- Morning: Explore locally.\n"
                "- Afternoon: Visit a museum.\n"
                "- Evening: Have dinner nearby.\n"
                "### Day 2\n"
                "- Morning: Walk in a park.\n"
                "- Afternoon: Visit a market.\n"
                "- Evening: Keep time flexible.\n"
                "### Day 3\n"
                "- Morning: Take a short walk.\n"
                "- Afternoon: Pack and rest.\n"
                "- Evening: Leave transfer time."
            ),
            "approval_request": "Review this draft.",
        },
    )
    monkeypatch.setattr(
        backend,
        "travel_graph",
        backend.graph.compile(checkpointer=MemorySaver()),
    )

    result = backend.run_travel_agent(
        TRIP_REQUEST,
        thread_id="one-agent-failure-test",
    )

    assert result["requires_approval"] is True
    assert result["flight_results"].startswith("UNAVAILABLE:")
    assert "LIVE: hotel observations" in result["hotel_results"]
    assert result["selected_agents"] == backend.AGENT_ORDER
    assert "### Day 3" in result["answer"]


def test_supervisor_failure_fails_closed_without_raising(monkeypatch):
    def fail_supervisor(_state):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr(backend, "supervisor_agent", fail_supervisor)

    result = backend._safe_supervisor_node({"user_query": TRIP_REQUEST})

    assert result["guardrail_allowed"] is False
    assert result["selected_agents"] == []
    assert "could not validate" in result["final_response"].casefold()
