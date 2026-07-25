"""Tests for app/server.py's API hardening: input validation returns clean
400 JSON instead of an unhandled 500, CORS is opt-in via an explicit
allowlist, request bodies are size-capped, and every error (ours or
Flask/werkzeug's own) comes back as JSON, never the default HTML page.
Requires reports/ artifacts (same precondition as tests/test_agent.py)."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent))
REPORTS = Path(__file__).parent.parent / "reports"


@unittest.skipUnless((REPORTS / "current_forecasts.csv").exists(),
                      "run scripts/run_pipeline.py first to generate reports/ artifacts")
class TestServerHardening(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from app.server import app
        app.testing = True
        cls.client = app.test_client()

    def test_health_ok(self):
        resp = self.client.get("/api/health")
        self.assertEqual(resp.status_code, 200)

    def test_valid_forecast_request_succeeds(self):
        resp = self.client.get("/api/forecast?store=1&dept=1")
        self.assertEqual(resp.status_code, 200)
        self.assertIn("forecast", resp.get_json())

    def test_non_integer_store_returns_clean_400_not_500(self):
        resp = self.client.get("/api/forecast?store=not-a-number&dept=1")
        self.assertEqual(resp.status_code, 400)
        body = resp.get_json()
        self.assertIn("error", body)
        self.assertIn("store", body["error"])

    def test_negative_store_returns_400(self):
        resp = self.client.get("/api/forecast?store=-1&dept=1")
        self.assertEqual(resp.status_code, 400)
        self.assertIn("error", resp.get_json())

    def test_non_integer_week_ahead_on_explain_returns_400(self):
        resp = self.client.get("/api/explain?store=1&dept=1&week_ahead=banana")
        self.assertEqual(resp.status_code, 400)

    def test_unknown_route_returns_json_404_not_html(self):
        resp = self.client.get("/api/does-not-exist")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.content_type, "application/json")
        self.assertIn("error", resp.get_json())

    def test_wrong_method_returns_json_405(self):
        resp = self.client.post("/api/forecast")
        self.assertEqual(resp.status_code, 405)
        self.assertEqual(resp.content_type, "application/json")

    def test_ask_missing_question_returns_400(self):
        resp = self.client.post("/api/ask", json={})
        self.assertEqual(resp.status_code, 400)

    def test_ask_non_string_question_returns_400(self):
        resp = self.client.post("/api/ask", json={"question": 12345})
        self.assertEqual(resp.status_code, 400)

    def test_ask_blank_question_returns_400(self):
        resp = self.client.post("/api/ask", json={"question": "   "})
        self.assertEqual(resp.status_code, 400)

    def test_ask_too_long_question_returns_400(self):
        resp = self.client.post("/api/ask", json={"question": "x" * 3000})
        self.assertEqual(resp.status_code, 400)

    def test_ask_malformed_body_returns_400_not_500(self):
        resp = self.client.post("/api/ask", data="not json", content_type="application/json")
        self.assertEqual(resp.status_code, 400)

    def test_ask_valid_question_uses_mock_backend(self):
        with patch.dict("os.environ", {"LLM_BACKEND": "mock"}, clear=False):
            resp = self.client.post("/api/ask", json={"question": "what's the forecast for store 1 dept 1?"})
        self.assertEqual(resp.status_code, 200)
        body = resp.get_json()
        self.assertEqual(body["backend"], "mock")
        self.assertIn("final_answer", body)

    def test_body_over_max_content_length_returns_413(self):
        huge_payload = json.dumps({"question": "x" * 20_000})
        resp = self.client.post("/api/ask", data=huge_payload, content_type="application/json")
        self.assertEqual(resp.status_code, 413)

    def test_cors_header_absent_when_origin_not_allowlisted(self):
        resp = self.client.get("/api/health", headers={"Origin": "https://evil.example.com"})
        self.assertNotIn("Access-Control-Allow-Origin", resp.headers)

    def test_cors_header_present_for_allowlisted_origin(self):
        import app.server as server_mod
        with patch.object(server_mod, "ALLOWED_ORIGINS", {"https://trusted.example.com"}):
            resp = self.client.get("/api/health", headers={"Origin": "https://trusted.example.com"})
        self.assertEqual(resp.headers.get("Access-Control-Allow-Origin"), "https://trusted.example.com")

    def test_unhandled_exception_returns_json_500_not_debug_page(self):
        def _boom(**kwargs):
            raise RuntimeError("boom")

        with patch.dict("app.server.TOOLS", {"get_forecast": _boom}):
            resp = self.client.get("/api/forecast?store=1&dept=1")
        self.assertEqual(resp.status_code, 500)
        self.assertEqual(resp.content_type, "application/json")
        self.assertIn("error", resp.get_json())


if __name__ == "__main__":
    unittest.main()
