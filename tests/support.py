"""Shared in-process API client for endpoint tests (throwaway SQLite database).

Import this module before anything from `app`: it points the app at a temporary
database. One client and one test account are shared by every test module.
"""
import atexit
import os
import tempfile
import warnings
from pathlib import Path

_DB = Path(tempfile.gettempdir()) / f"statsathi_tests_{os.getpid()}.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_DB.as_posix()}"
os.environ.setdefault("JWT_SECRET_KEY", "unit-test-only-secret-0123456789abcdef0123456789")
# Many tests sign up and sign in from one address; the rate-limit tests switch limits back on themselves
os.environ["RATE_LIMIT_ENABLED"] = "false"
os.environ["EMAIL_BACKEND"] = "console"
warnings.filterwarnings("ignore")

from fastapi.testclient import TestClient  # noqa: E402

from app import database  # noqa: E402
from app.main import app  # noqa: E402

_client = None
_headers = None


def _api():
    global _client, _headers
    if _client is None:
        _client = TestClient(app)
        credentials = {"email": "api-tests@example.com", "password": "api-tests-password"}
        _client.post("/auth/signup", json={**credentials, "full_name": "API Tests"})
        token = _client.post("/auth/login", data={"username": credentials["email"],
                                                  "password": credentials["password"]}).json()["access_token"]
        _headers = {"Authorization": f"Bearer {token}"}
    return _client, _headers


def client_and_headers(email=None, password="api-tests-password"):
    """The shared TestClient plus auth headers for the default test user, or for another user (created on demand)."""
    client, headers = _api()
    if email is None:
        return client, headers
    client.post("/auth/signup", json={"email": email, "password": password, "full_name": "Other Test User"})
    token = client.post("/auth/login", data={"username": email, "password": password}).json()["access_token"]
    return client, {"Authorization": f"Bearer {token}"}


def post_csv(path, df, **form):
    """POST a DataFrame as an uploaded CSV plus form fields; return (status_code, json)."""
    client, headers = _api()
    response = client.post(path, headers=headers,
                           files={"file": ("data.csv", df.to_csv(index=False).encode(), "text/csv")},
                           data=form)
    return response.status_code, response.json()


def post_file(path, filename, content, media_type, **form):
    """POST raw file bytes plus form fields; return the raw response (for non-JSON endpoints)."""
    client, headers = _api()
    return client.post(path, headers=headers, files={"file": (filename, content, media_type)}, data=form)


@atexit.register
def _cleanup():
    if _client is not None:
        _client.close()
    database.engine.dispose()
    _DB.unlink(missing_ok=True)
