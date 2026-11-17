# ✈️ TripMate AI – Multi-Agent Intelligent Travel Planner

An AI-powered travel planning application built using **Python, LangGraph, MCP, FastAPI, and Large Language Models (LLMs)**. TripMate AI uses multiple specialized AI agents coordinated by a Supervisor Agent to research travel information and generate personalized travel itineraries.

The application integrates external services for flight information, hotel research, weather forecasts, and travel budgeting. It also includes Human-in-the-Loop (HITL) approval, allowing users to review and revise a travel plan before final approval.

---

## 📌 1. Project Overview

Planning a trip involves researching flights, hotels, weather conditions, expenses, and activities. Managing all these tasks manually can be time-consuming.

TripMate AI simplifies this process through a multi-agent architecture.

Users provide their travel requirements, such as:
- Starting location and destination
- Number of travelers
- Travel duration
- Departure and return dates (optional)
- Budget and travel preferences

The AI agents process these requirements and collaborate to generate a structured travel plan.

## 🎯 2. Key Features

- **Multi-Agent Architecture:** Multiple specialized agents collaborate to complete travel-planning tasks.
- **Supervisor Agent:** Coordinates the workflow and selects the required agents.
- **Input Guardrails:** Validates travel requests before processing.
- **Flight Research:** Retrieves flight schedule information using AviationStack and optional flight offers through Amadeus.
- **Hotel Research:** Searches hotel information using Tavily and optional hotel offers through Amadeus.
- **Weather Information:** Retrieves current weather conditions and available forecast information using OpenWeather.
- **Budget Estimation:** Generates estimated travel expenses and budget breakdowns.
- **AI Itinerary Generation:** Creates structured day-wise travel schedules.
- **Human-in-the-Loop:** Allows users to review, approve, or request changes to generated travel plans.
- **PostgreSQL Persistence:** Uses LangGraph PostgreSQL checkpointing to maintain workflow state.
- **Interactive Web Interface:** Provides a responsive interface for submitting travel requests and viewing results.
- **Copy and PDF:** Supports copying the generated plan and printing or saving it as a PDF.

## 🏗️ 3. System Architecture

TripMate AI follows a Supervisor-based multi-agent architecture using LangGraph.

### Workflow

1. User submits a travel request.
2. Supervisor Agent analyzes the request.
3. Input Guardrails validate the request.
4. Supervisor selects the required specialized agents.
5. Flight Agent researches flight information.
6. Hotel Agent researches accommodation options.
7. Weather Agent retrieves weather information.
8. Budget Agent prepares the estimated budget.
9. Itinerary Agent generates the travel schedule.
10. Human Approval Agent pauses the workflow for user review.
11. User approves the plan or provides revision feedback.
12. Final Agent generates the approved travel plan.

### Agent Architecture

| Agent | Responsibility |
|---|---|
| Supervisor Agent | Analyzes requests and coordinates agents |
| Guardrail Agent | Handles requests blocked by input validation |
| Flight Agent | Retrieves flight-related information |
| Hotel Agent | Researches hotel information |
| Weather Agent | Retrieves weather conditions and forecasts |
| Budget Agent | Calculates estimated travel expenses |
| Itinerary Agent | Generates the day-wise itinerary |
| Human Approval Agent | Handles user approval and revision |
| Final Agent | Produces the final approved response |

The specialized agents are coordinated through conditional routing in LangGraph.

## 🛠️ 4. Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangGraph | Multi-agent workflow orchestration |
| LangChain | LLM integration and agent components |
| Groq | Large Language Model inference |
| MCP | Model Context Protocol integrations |
| FastAPI | Backend API development |
| PostgreSQL | Workflow checkpoint persistence |
| Tavily | Web and hotel research |
| AviationStack | Flight schedules and status information |
| OpenWeather | Weather information |
| Amadeus API | Optional flight and hotel offers |
| HTML5 | Frontend structure |
| CSS3 | Frontend styling |
| JavaScript | Frontend interactions |
| Docker | Containerization |

## 🔌 5. External API Integrations

### 1. Groq API
- Powers the AI agents.
- Processes travel requirements.
- Generates travel recommendations and itineraries.
- Default configured model: `openai/gpt-oss-120b`.

### 2. Tavily API
- Provides web-search capabilities through MCP.
- Supports hotel research.
- Returns search observations and source URLs.
- Does not guarantee hotel room availability or prices.

### 3. AviationStack API
- Provides flight-related information.
- Supports flight schedule and specific flight-status requests.
- Uses the AviationStack MCP server.
- Does not provide flight bookings or fare quotes.

### 4. OpenWeather API
- Retrieves current weather conditions.
- Retrieves the next five available three-hour forecast periods.
- Provides weather information through a custom MCP server.

### 5. Amadeus API (Optional)
- Flight Offers Search.
- Hotel List.
- Hotel Offers Search.
- Supports test and production environments.

**Important:** Amadeus test results are sandbox data, not confirmed live fares or hotel availability. The application does not book or reserve travel.

### 6. PostgreSQL
- Stores LangGraph workflow checkpoints.
- Maintains state between approval and revision operations.
- Supports persistent workflow execution.

## 📂 6. Project Structure

```text
TripMate_AI_Debugged/
│
├── app.py
│
├── backend.py
│
├── mcp_client.py
│
├── custom_weather_mcp_server.py
│
├── amadeus_provider.py
│
├── check_tools.py
│
├── templates/
│   └── index.html
│
├── static/
│   ├── styles.css
│   └── script.js
│
├── tests/
│   └── test_project_integrity.py
│
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
├── .dockerignore
├── Dockerfile
├── README.md
└── DEBUG_REPORT.md
```

### Main Files Explained

| File | Description |
|---|---|
| `app.py` | FastAPI application, endpoints, sessions and request handling |
| `backend.py` | Main LangGraph workflow and AI agent implementation |
| `mcp_client.py` | Connects with external MCP servers and retrieves tools |
| `custom_weather_mcp_server.py` | Custom MCP server for OpenWeather |
| `amadeus_provider.py` | Amadeus flight and hotel API integration |
| `check_tools.py` | MCP tool connection testing |
| `index.html` | Main web interface |
| `styles.css` | Frontend styling |
| `script.js` | Frontend API calls and interactions |

## ⚙️ 7. Installation and Setup

### Prerequisites

- Python 3.13+
- Git
- VS Code
- PostgreSQL database (recommended for persistent workflow state)
- Internet connection
- Required API credentials

`uv` is also required for the AviationStack MCP server.

### Step 1: Clone the Repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
```

```bash
cd TripMate_AI_Debugged
```

### Step 2: Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment on Windows:

```powershell
venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

First, check `requirements.txt`. The current uploaded version contains a malformed final dependency line. Separate it into:

```text
certifi==2026.6.17
itsdangerous>=2.2,<3
```

Then install:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create a `.env` file from the example:

```powershell
Copy-Item .env.example .env
```

Add your API credentials:

```env
# PostgreSQL Database
DATABASE_URL=postgresql://USER:PASSWORD@HOST:5432/DATABASE?sslmode=require

# Groq LLM
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=openai/gpt-oss-120b

# Tavily
TAVILY_API_KEY=your_tavily_api_key

# AviationStack
AVIATION_STACK_API_KEY=your_aviationstack_api_key

# OpenWeather
OPENWEATHER_API_KEY=your_openweather_api_key

# Amadeus (Optional)
AMADEUS_API_KEY=your_amadeus_api_key
AMADEUS_API_SECRET=your_amadeus_api_secret
AMADEUS_ENV=test

# Application Security
APP_SECRET_KEY=
APP_ENV=development
```

**Note:** 
- Groq credentials are required for AI generation.
- Tavily, AviationStack, and OpenWeather credentials are required for their respective integrations.
- Amadeus credentials are optional.
- PostgreSQL is required for production deployment.
- Never upload your `.env` file or API keys to GitHub.

### Step 5: Run the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload --host 127.0.0.1 --port 8000
```

Alternatively:

```bash
python app.py
```

### Step 6: Open the Application

Open your browser:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health check:

```text
http://127.0.0.1:8000/health
```

## 🔄 8. Application Workflow

### User Input

The user enters travel requirements through the web interface.

Example:

```text
Plan a 3-day trip to Bangkok from Hyderabad
for 2 travelers with a moderate budget.
```

### Supervisor Processing

The Supervisor Agent analyzes the request and selects the required agents.

### Research Phase

The specialized agents collect available information:

- Flight information
- Hotel research
- Weather conditions
- Estimated expenses

### Itinerary Generation

The Itinerary Agent combines the available information and generates a structured day-wise travel plan.

### Human Approval

The generated draft is presented to the user.

Users can:

- Approve the travel plan.
- Request changes by providing feedback.

If changes are requested, the workflow generates a replacement draft that requires approval again.

### Final Output

After approval, the Final Agent produces the approved travel plan.

## 🌐 9. API Endpoints

The application exposes the following main endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Loads the frontend |
| POST | `/api/travel` | Creates a travel plan |
| POST | `/api/travel/approve` | Approves or revises a travel plan |
| GET | `/health` | Checks application and checkpoint status |
| GET | `/favicon.ico` | Serves the favicon |

### Travel Request Example

`POST /api/travel`

```json
{
  "message": "Plan a 3-day trip to Bangkok from Hyderabad for 2 people",
  "departure_date": "2026-12-15",
  "return_date": "2026-12-17"
}
```

Dates are optional.

### Approval Request Example

`POST /api/travel/approve`

```json
{
  "thread_id": "your_thread_id",
  "approved": true,
  "feedback": ""
}
```

For revisions:

```json
{
  "thread_id": "your_thread_id",
  "approved": false,
  "feedback": "Reduce hotel expenses and add more sightseeing places."
}
```

## 🗄️ 10. Database and Checkpoint Management

TripMate AI uses LangGraph checkpointing to preserve workflow state.

PostgreSQL is used for persistent checkpoint storage.

The checkpoint mechanism supports:
- Maintaining workflow state.
- Resuming interrupted workflows.
- Handling Human-in-the-Loop approval.
- Processing revision feedback.
- Continuing the same workflow using its thread ID.

In development, the application can fall back to in-memory checkpointing when PostgreSQL is unavailable. In production, PostgreSQL is mandatory.

## 🧪 11. Testing

Install development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run Python tests:

```bash
python -m pytest -q
```

Check JavaScript syntax:

```bash
node --check static/script.js
```

Test MCP connections:

```bash
python check_tools.py
```

The included automated tests focus on Python source parsing and project integrity. Successful tests do not establish that external APIs, PostgreSQL, or the complete live AI workflow are working.

## 🐳 12. Docker Deployment

Build the Docker image:

```bash
docker build -t tripmate-ai .
```

Run the container:

```bash
docker run -p 8000:8000 --env-file .env tripmate-ai
```

Open:

```text
http://localhost:8000
```

For production, configure `APP_ENV=production`, a secure `APP_SECRET_KEY`, and a working PostgreSQL `DATABASE_URL`.

## 🔐 13. Security Considerations

- Store API credentials in environment variables.
- Do not commit `.env` files.
- Use a strong random `APP_SECRET_KEY` in production.
- Use PostgreSQL for persistent production checkpointing.
- The application uses signed HttpOnly browser sessions.
- Approval requests are checked against browser-session thread ownership.
- Frontend model output is inserted as text rather than untrusted HTML.

The browser-session ownership mechanism is not a complete multi-user authentication system.

## ⚠️ 14. Important Limitations

- AviationStack does not provide flight booking or fare quotations.
- Weather forecasts cover only the next five three-hour periods.
- Tavily search results are not verified hotel inventory or rates.
- Budget amounts generated by the AI are estimates, not guaranteed prices.
- Amadeus sandbox results are not live booking confirmations.
- The application does not perform actual flight or hotel reservations.
- Live API functionality requires valid credentials and successful provider connections.

## 🚀 15. Future Enhancements

Potential improvements include:

- User registration and authentication.
- Saving and managing multiple trips.
- Interactive maps and location-based recommendations.
- Improved live flight tracking.
- More comprehensive weather forecasts.
- Real-time hotel availability integration.
- Currency conversion and expense tracking.
- Downloadable PDF itinerary generation.
- Mobile-friendly application improvements.
- Trip history and personalized recommendations.

## 👨‍💻 16. Project Purpose

TripMate AI demonstrates how multiple specialized AI agents can collaborate using LangGraph and MCP to solve a real-world problem.

The project combines AI orchestration, external API integrations, backend development, workflow persistence, and Human-in-the-Loop decision-making into a single intelligent travel-planning application.

---

**Project Name:** TripMate AI – Intelligent Multi-Agent Travel Planner

**Core Technologies:** Python | LangGraph | LangChain | MCP | Groq | FastAPI | PostgreSQL

**Developer:** Add your name here

**License:** See the LICENSE file included in this repository.
