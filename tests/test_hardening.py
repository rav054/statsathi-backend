"""Request size limits, safe answers to unexpected errors, security headers, CORS and the health check."""
import unittest
from unittest import mock

import pandas as pd
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.testclient import TestClient

from tests.support import client_and_headers, post_file  # noqa: F401  (sets up the test database first)

from app.config import Settings, settings
from app.hardening import RequestSizeLimitMiddleware, SecurityHeadersMiddleware, UnexpectedErrorMiddleware
from app.main import app


def small_app(max_bytes=100):
    demo = FastAPI()

    @demo.post("/echo")
    async def echo(request: Request):
        return {"received": len(await request.body())}

    @demo.get("/boom")
    def boom():
        raise RuntimeError("database password is hunter2")

    demo.add_middleware(UnexpectedErrorMiddleware)
    demo.add_middleware(RequestSizeLimitMiddleware, max_bytes=max_bytes)
    demo.add_middleware(SecurityHeadersMiddleware)
    demo.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])
    return TestClient(demo)


class RequestSizeTests(unittest.TestCase):
    def test_declared_and_streamed_bodies_over_the_limit_are_refused(self):
        client = small_app(max_bytes=100)
        self.assertEqual(client.post("/echo", content=b"x" * 100).json(), {"received": 100})

        too_big = client.post("/echo", content=b"x" * 101, headers={"Origin": "http://localhost:5173"})
        self.assertEqual(too_big.status_code, 413)
        self.assertIn("at most", too_big.json()["detail"])
        self.assertEqual(too_big.headers.get("access-control-allow-origin"), "http://localhost:5173")

        streamed = client.post("/echo", content=iter([b"x" * 60, b"x" * 60]))
        self.assertEqual(streamed.status_code, 413)

    def test_analysis_uploads_over_the_table_limit_get_a_clear_message(self):
        with mock.patch.object(settings, "MAX_TABLE_UPLOAD_MB", 1):
            content = pd.DataFrame({"yield": range(200_000)}).to_csv(index=False).encode()
            self.assertGreater(len(content), 1024 * 1024)
            response = post_file("/analyze/columns", "big.csv", content, "text/csv")
        self.assertEqual(response.status_code, 413)
        self.assertIn("larger than 1 MB", response.json()["detail"])


class UnexpectedErrorTests(unittest.TestCase):
    def test_unexpected_errors_get_a_reference_and_hide_internals(self):
        client = small_app()
        with self.assertLogs("statsathi.errors", level="ERROR") as logs:
            response = client.get("/boom", headers={"Origin": "http://localhost:5173"})
        self.assertEqual(response.status_code, 500)
        body = response.json()
        self.assertNotIn("hunter2", response.text)
        self.assertIn(body["reference"], body["detail"])
        self.assertEqual(response.headers.get("access-control-allow-origin"), "http://localhost:5173")
        self.assertIn(body["reference"], "\n".join(logs.output))


class ApplicationSettingsTests(unittest.TestCase):
    def test_security_headers_and_health_check(self):
        client, _ = client_and_headers()
        response = client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "healthy", "database": "ok"})
        self.assertEqual(response.headers["x-content-type-options"], "nosniff")
        self.assertEqual(response.headers["x-frame-options"], "DENY")
        self.assertNotIn("strict-transport-security", response.headers)
        secure = client.get("/health", headers={"X-Forwarded-Proto": "https"})
        self.assertIn("max-age", secure.headers["strict-transport-security"])

    def test_health_check_reports_an_unreachable_database(self):
        client, _ = client_and_headers()
        with mock.patch("app.main.SessionLocal", side_effect=RuntimeError("down")), \
                self.assertLogs("statsathi.errors", level="ERROR"):
            response = client.get("/health")
        self.assertEqual(response.status_code, 503)

    def test_cors_origins_come_from_settings(self):
        configured = Settings(CORS_ORIGINS=" https://statsathi.org/ , http://localhost:5173")
        self.assertIn("https://statsathi.org", configured.cors_origins)
        self.assertEqual(configured.cors_origins.count("http://localhost:5173"), 1)

        client, _ = client_and_headers()
        preflight = {"Access-Control-Request-Method": "POST"}
        allowed = client.options("/auth/login", headers={"Origin": "http://localhost:5173", **preflight})
        self.assertEqual(allowed.headers.get("access-control-allow-origin"), "http://localhost:5173")
        refused = client.options("/auth/login", headers={"Origin": "https://evil.example.com", **preflight})
        self.assertIsNone(refused.headers.get("access-control-allow-origin"))


if __name__ == "__main__":
    unittest.main()
