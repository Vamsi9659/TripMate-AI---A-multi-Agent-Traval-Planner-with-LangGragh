from pathlib import Path
import ast

ROOT = Path(__file__).resolve().parents[1]


def test_python_sources_parse():
    sources = [
        ROOT / "app.py",
        ROOT / "backend.py",
        ROOT / "mcp_client.py",
        ROOT / "custom_weather_mcp_server.py",
        ROOT / "check_tools.py",
    ]
    for source in sources:
        ast.parse(source.read_text(encoding="utf-8"), filename=str(source))


def test_fastapi_frontend_assets_exist():
    assert (ROOT / "templates" / "index.html").is_file()
    assert (ROOT / "static" / "script.js").is_file()
    assert (ROOT / "static" / "styles.css").is_file()


def test_guardrail_receives_full_api_message_limit():
    backend = (ROOT / "backend.py").read_text(encoding="utf-8")
    assert "Request: {_context_text(query, 4000)}" in backend


def test_production_requires_persistent_database_and_session_secret():
    backend = (ROOT / "backend.py").read_text(encoding="utf-8")
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    assert 'APP_ENV=production' in backend
    assert "APP_SECRET_KEY is required when APP_ENV=production" in app


def test_docker_ignores_local_secrets():
    dockerignore = (ROOT / ".dockerignore").read_text(encoding="utf-8")
    assert ".env" in dockerignore
    assert ".git" in dockerignore
