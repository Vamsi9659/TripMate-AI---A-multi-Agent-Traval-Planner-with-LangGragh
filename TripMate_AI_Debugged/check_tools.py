import asyncio
from mcp_client import client

async def main():
    tools = await client.get_tools(server_name="aviationstack")

    for tool in tools:
        if tool.name in [
            "list_routes",
            "list_airports",
            "flights_with_airline",
            "flight_arrival_departure_schedule",
            "future_flights_arrival_departure_schedule",
        ]:
            print("\nTOOL:", tool.name)
            print("ARGS:", tool.args)

asyncio.run(main())