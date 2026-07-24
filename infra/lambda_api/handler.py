"""AWS Lambda entrypoint that serves the existing Flask app (app/server.py)
completely unmodified, via a small API Gateway HTTP API (payload format
v2.0) <-> WSGI adapter written by hand here.

Why hand-written instead of a library like `aws-wsgi`/`serverless-wsgi`:
this adapter is ~60 lines and entirely mechanical (translate one dict shape
into another), so writing it out is more auditable than adding a dependency
for something this small -- consistent with this project's existing
"no vendor SDK where a plain implementation is small and clear enough"
choices for the Anthropic/OpenAI HTTP calls (see docs/ARCHITECTURE.md).
Bedrock's boto3 dependency (src/agent.py::BedrockLLM) is the deliberate
exception to that pattern, for the reasons documented there.

This module contains ZERO route logic. It imports the exact `app` object
`python app/server.py` runs locally and only translates between API
Gateway's event/response shape and WSGI's environ/start_response contract.
Every route (`/api/forecast`, `/api/ask`, the dashboard HTML, etc.) is
defined exactly once, in app/server.py, and behaves identically whether
Flask's dev server or this Lambda adapter is serving it.

Deploy note: infra/terraform/apigateway.tf wires this Lambda behind the
API Gateway HTTP API's `$default` stage specifically so `rawPath` arrives
WITHOUT a stage prefix (e.g. "/api/health", not "/prod/api/health") --
this adapter assumes that. If you deploy behind a named stage instead, set
the API_GATEWAY_STAGE_PREFIX environment variable (e.g. "/prod") and it
will be stripped from the path before dispatch.

Secrets Manager note: infra/terraform/secretsmanager.tf creates (empty,
placeholder) secrets for ANTHROPIC_API_KEY/OPENAI_API_KEY, and
infra/terraform/lambda.tf sets this Lambda's *_SECRET_ARN environment
variables to point at them (never the raw key value -- that stays out of
Lambda's own environment-variable configuration, which Terraform state and
the Lambda console would otherwise expose in plaintext). This module
resolves those ARNs into the real ANTHROPIC_API_KEY/OPENAI_API_KEY
environment variables at cold start via a Secrets Manager lookup, so
src/agent.py::get_llm_backend() sees them exactly as if they'd been set
directly. See _hydrate_secret_env_var()'s docstring for the fallback
behavior when no real key has been populated yet.

NOT validated against a real API Gateway invocation, a real Lambda
runtime, or a real Secrets Manager lookup in this build sandbox -- see
infra/README.md's disclosure section. What IS verified
(tests/test_lambda_adapter.py) is that this adapter, called directly with
a synthetic API Gateway v2.0 event, correctly drives the real
`app.server.app` Flask object and gets back the same JSON the Flask dev
server would produce for the same request, and that the secret-hydration
logic resolves (or safely no-ops on) a mocked Secrets Manager client.
"""
import base64
import io
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _hydrate_secret_env_var(env_var_name: str) -> None:
    """If `env_var_name` isn't already set (e.g. by a local test or a
    manual override) but a matching `<env_var_name>_SECRET_ARN` is, fetch
    its current value from Secrets Manager and populate the env var.
    Best-effort and deliberately broad in what it catches: a missing
    boto3, a permissions error, or the secret still holding the
    Terraform-created "REPLACE_ME_MANUALLY" placeholder should all degrade
    to "no real key available" rather than crash Lambda cold start --
    src/agent.py::get_llm_backend() already falls back to the
    deterministic MockLLM whenever no real API key ends up set, which is
    exactly the safe default this is meant to preserve."""
    if os.environ.get(env_var_name):
        return
    secret_arn = os.environ.get(f"{env_var_name}_SECRET_ARN")
    if not secret_arn:
        return
    try:
        import boto3
        client = boto3.client("secretsmanager")
        value = client.get_secret_value(SecretId=secret_arn)["SecretString"]
        if value and value != "REPLACE_ME_MANUALLY":
            os.environ[env_var_name] = value
    except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring above
        print(f"Could not hydrate {env_var_name} from Secrets Manager ({secret_arn}): {e}")


_hydrate_secret_env_var("ANTHROPIC_API_KEY")
_hydrate_secret_env_var("OPENAI_API_KEY")

from app.server import app  # noqa: E402  -- the exact same Flask app object used locally

STAGE_PREFIX = os.environ.get("API_GATEWAY_STAGE_PREFIX", "").rstrip("/")

# WSGI reserves these two header names as their own environ keys (no
# HTTP_ prefix) -- see PEP 3333. Every other header gets HTTP_<NAME>.
_WSGI_RESERVED_HEADER_ENVIRON = {"content-type": "CONTENT_TYPE", "content-length": "CONTENT_LENGTH"}


def _event_to_environ(event: dict) -> dict:
    """Build a WSGI environ dict from an API Gateway HTTP API v2.0 event.
    Shape reference: AWS docs, "Working with AWS Lambda proxy integrations
    for HTTP APIs" (payload format version 2.0)."""
    request_context = event.get("requestContext", {}) or {}
    http_ctx = request_context.get("http", {}) or {}
    method = http_ctx.get("method", "GET")

    raw_path = event.get("rawPath") or "/"
    if STAGE_PREFIX and raw_path.startswith(STAGE_PREFIX):
        raw_path = raw_path[len(STAGE_PREFIX):] or "/"

    headers = {k.lower(): v for k, v in (event.get("headers") or {}).items()}

    # rawQueryString is the ORIGINAL, unsplit query string. Payload v2 also
    # provides queryStringParameters with repeated keys comma-joined, which
    # would corrupt any repeated param containing a literal comma value --
    # rawQueryString round-trips exactly what the client sent, so prefer it.
    query_string = event.get("rawQueryString", "") or ""

    body_raw = event.get("body") or ""
    if event.get("isBase64Encoded"):
        body_bytes = base64.b64decode(body_raw)
    else:
        body_bytes = body_raw.encode("utf-8")

    host_header = headers.get("host", "lambda")
    server_name, _, server_port = host_header.partition(":")

    environ = {
        "REQUEST_METHOD": method,
        "SCRIPT_NAME": "",
        "PATH_INFO": raw_path,
        "QUERY_STRING": query_string,
        "SERVER_NAME": server_name or "lambda",
        "SERVER_PORT": server_port or "443",
        "SERVER_PROTOCOL": "HTTP/1.1",
        "REMOTE_ADDR": http_ctx.get("sourceIp", "0.0.0.0"),
        "CONTENT_TYPE": headers.get("content-type", ""),
        "CONTENT_LENGTH": str(len(body_bytes)),
        "wsgi.version": (1, 0),
        "wsgi.url_scheme": headers.get("x-forwarded-proto", "https"),
        "wsgi.input": io.BytesIO(body_bytes),
        "wsgi.errors": sys.stderr,
        "wsgi.multithread": False,
        "wsgi.multiprocess": False,
        "wsgi.run_once": True,
    }
    for key, value in headers.items():
        if key in _WSGI_RESERVED_HEADER_ENVIRON:
            continue
        environ["HTTP_" + key.upper().replace("-", "_")] = value
    return environ


def lambda_handler(event: dict, context=None) -> dict:
    """The Lambda entrypoint (Terraform sets this as the handler:
    infra/lambda_api/handler.lambda_handler, see infra/terraform/lambda.tf).
    Takes an API Gateway HTTP API v2.0 event, drives the Flask app exactly
    as its own WSGI server would, and returns a v2.0-format response."""
    environ = _event_to_environ(event)

    response_state = {}

    def start_response(status, response_headers, exc_info=None):
        response_state["status"] = status
        response_state["headers"] = response_headers

    body_chunks = app.wsgi_app(environ, start_response)
    try:
        body = b"".join(body_chunks).decode("utf-8")
    finally:
        if hasattr(body_chunks, "close"):
            body_chunks.close()  # release Werkzeug's response context per-request

    status_code = int(response_state["status"].split(" ", 1)[0])
    response_headers = {k: v for k, v in response_state["headers"]}

    return {
        "statusCode": status_code,
        "headers": response_headers,
        "body": body,
        "isBase64Encoded": False,
    }
