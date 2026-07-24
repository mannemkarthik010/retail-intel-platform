"""Tests for the API Gateway <-> WSGI Lambda adapter (infra/lambda_api/handler.py).

None of this touches AWS -- these tests call `lambda_handler` directly with
hand-built API Gateway HTTP API v2.0 events (the same shape API Gateway
would actually send) and check that it drives the real `app.server.app`
Flask object correctly, i.e. that deploying behind Lambda would produce the
same responses `python app/server.py` does locally.
"""
import importlib
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

REPORTS = ROOT / "reports"


def _v2_event(method="GET", path="/api/health", query_string="", body=None,
               headers=None, is_base64=False):
    """Build a minimal but realistic API Gateway HTTP API v2.0 event."""
    return {
        "version": "2.0",
        "rawPath": path,
        "rawQueryString": query_string,
        "headers": headers or {"host": "abc123.execute-api.us-east-1.amazonaws.com"},
        "requestContext": {"http": {"method": method, "sourceIp": "203.0.113.5"}},
        "body": body,
        "isBase64Encoded": is_base64,
    }


class TestEventToEnviron(unittest.TestCase):
    def setUp(self):
        from infra.lambda_api import handler
        self.handler = handler

    def test_basic_get_maps_method_path_and_query(self):
        event = _v2_event(method="GET", path="/api/forecast", query_string="store=3&dept=5")
        environ = self.handler._event_to_environ(event)
        self.assertEqual(environ["REQUEST_METHOD"], "GET")
        self.assertEqual(environ["PATH_INFO"], "/api/forecast")
        self.assertEqual(environ["QUERY_STRING"], "store=3&dept=5")
        self.assertEqual(environ["SERVER_NAME"], "abc123.execute-api.us-east-1.amazonaws.com")

    def test_json_post_body_is_utf8_decoded_into_wsgi_input(self):
        payload = json.dumps({"question": "what's the forecast for store 1 dept 1?"})
        event = _v2_event(method="POST", path="/api/ask", body=payload,
                           headers={"host": "x", "content-type": "application/json"})
        environ = self.handler._event_to_environ(event)
        self.assertEqual(environ["REQUEST_METHOD"], "POST")
        self.assertEqual(environ["CONTENT_TYPE"], "application/json")
        self.assertEqual(environ["wsgi.input"].read(), payload.encode("utf-8"))
        self.assertEqual(environ["CONTENT_LENGTH"], str(len(payload.encode("utf-8"))))

    def test_base64_encoded_body_is_decoded(self):
        import base64
        raw = b'{"question": "hi"}'
        event = _v2_event(method="POST", path="/api/ask", body=base64.b64encode(raw).decode(),
                           is_base64=True)
        environ = self.handler._event_to_environ(event)
        self.assertEqual(environ["wsgi.input"].read(), raw)

    def test_stage_prefix_is_stripped_when_configured(self):
        event = _v2_event(method="GET", path="/prod/api/health")
        with patch.dict("os.environ", {"API_GATEWAY_STAGE_PREFIX": "/prod"}, clear=False):
            importlib.reload(self.handler)
            try:
                environ = self.handler._event_to_environ(event)
                self.assertEqual(environ["PATH_INFO"], "/api/health")
            finally:
                importlib.reload(self.handler)  # restore STAGE_PREFIX="" for other tests

    def test_headers_become_http_prefixed_environ_vars(self):
        event = _v2_event(headers={"host": "x", "x-custom-header": "abc"})
        environ = self.handler._event_to_environ(event)
        self.assertEqual(environ["HTTP_X_CUSTOM_HEADER"], "abc")
        # content-type/content-length must NOT get the HTTP_ prefix (WSGI/PEP 3333).
        self.assertNotIn("HTTP_CONTENT_TYPE", environ)


class TestLambdaHandlerEndToEnd(unittest.TestCase):
    """Drives the real Flask app through the adapter -- no mocking of
    app.server itself, so a route-definition bug there would show up here."""

    def test_health_check_roundtrip(self):
        from infra.lambda_api.handler import lambda_handler
        response = lambda_handler(_v2_event(method="GET", path="/api/health"))
        self.assertEqual(response["statusCode"], 200)
        self.assertEqual(json.loads(response["body"]), {"status": "ok"})
        self.assertFalse(response["isBase64Encoded"])

    def test_unknown_route_returns_404(self):
        from infra.lambda_api.handler import lambda_handler
        response = lambda_handler(_v2_event(method="GET", path="/api/does-not-exist"))
        self.assertEqual(response["statusCode"], 404)

    @unittest.skipUnless((REPORTS / "current_forecasts.csv").exists(),
                         "run scripts/run_pipeline.py first to generate reports/ artifacts")
    def test_forecast_query_params_pass_through(self):
        from infra.lambda_api.handler import lambda_handler
        response = lambda_handler(_v2_event(method="GET", path="/api/forecast",
                                             query_string="store=1&dept=1"))
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertIn("forecast", body)

    @unittest.skipUnless((REPORTS / "current_forecasts.csv").exists(),
                         "run scripts/run_pipeline.py first to generate reports/ artifacts")
    def test_post_ask_with_json_body_uses_mock_backend(self):
        from infra.lambda_api.handler import lambda_handler
        # Force the deterministic mock so this test makes no real network
        # calls regardless of which API keys happen to be set in the
        # environment this test runs in.
        with patch.dict("os.environ", {"LLM_BACKEND": "mock"}, clear=False):
            payload = json.dumps({"question": "what's the forecast for store 1 dept 1?"})
            event = _v2_event(method="POST", path="/api/ask", body=payload,
                               headers={"host": "x", "content-type": "application/json"})
            response = lambda_handler(event)
        self.assertEqual(response["statusCode"], 200)
        body = json.loads(response["body"])
        self.assertEqual(body["backend"], "mock")
        self.assertIn("final_answer", body)

    def test_missing_question_returns_400(self):
        from infra.lambda_api.handler import lambda_handler
        event = _v2_event(method="POST", path="/api/ask", body=json.dumps({}),
                           headers={"host": "x", "content-type": "application/json"})
        response = lambda_handler(event)
        self.assertEqual(response["statusCode"], 400)


class TestSecretHydration(unittest.TestCase):
    """_hydrate_secret_env_var() resolves a *_SECRET_ARN env var into the
    real env var via a mocked Secrets Manager client -- no real AWS call,
    same sys.modules-mocking technique as src/agent.py's BedrockLLM tests."""

    def test_populates_env_var_from_secrets_manager_when_arn_present(self):
        from infra.lambda_api import handler
        fake_boto3 = MagicMock()
        fake_client = MagicMock()
        fake_boto3.client.return_value = fake_client
        fake_client.get_secret_value.return_value = {"SecretString": "sk-real-key"}

        env = {"ANTHROPIC_API_KEY_SECRET_ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:fake"}
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with patch.dict("os.environ", env, clear=True):
                handler._hydrate_secret_env_var("ANTHROPIC_API_KEY")
                self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), "sk-real-key")
        fake_client.get_secret_value.assert_called_once_with(
            SecretId="arn:aws:secretsmanager:us-east-1:123456789012:secret:fake")

    def test_does_not_overwrite_an_already_set_env_var(self):
        from infra.lambda_api import handler
        env = {"ANTHROPIC_API_KEY": "sk-already-here",
               "ANTHROPIC_API_KEY_SECRET_ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:fake"}
        with patch.dict("os.environ", env, clear=True):
            handler._hydrate_secret_env_var("ANTHROPIC_API_KEY")
            self.assertEqual(os.environ.get("ANTHROPIC_API_KEY"), "sk-already-here")

    def test_noop_when_no_secret_arn_configured(self):
        from infra.lambda_api import handler
        with patch.dict("os.environ", {}, clear=True):
            handler._hydrate_secret_env_var("OPENAI_API_KEY")
            self.assertIsNone(os.environ.get("OPENAI_API_KEY"))

    def test_leaves_env_var_unset_when_secret_still_holds_placeholder(self):
        from infra.lambda_api import handler
        fake_boto3 = MagicMock()
        fake_client = MagicMock()
        fake_boto3.client.return_value = fake_client
        fake_client.get_secret_value.return_value = {"SecretString": "REPLACE_ME_MANUALLY"}

        env = {"OPENAI_API_KEY_SECRET_ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:fake"}
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with patch.dict("os.environ", env, clear=True):
                handler._hydrate_secret_env_var("OPENAI_API_KEY")
                self.assertIsNone(os.environ.get("OPENAI_API_KEY"))

    def test_swallows_errors_instead_of_raising(self):
        from infra.lambda_api import handler
        fake_boto3 = MagicMock()
        fake_boto3.client.side_effect = RuntimeError("no network access in this sandbox")

        env = {"OPENAI_API_KEY_SECRET_ARN": "arn:aws:secretsmanager:us-east-1:123456789012:secret:fake"}
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with patch.dict("os.environ", env, clear=True):
                handler._hydrate_secret_env_var("OPENAI_API_KEY")  # must not raise
                self.assertIsNone(os.environ.get("OPENAI_API_KEY"))


if __name__ == "__main__":
    unittest.main()
