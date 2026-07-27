"""Vercel entrypoint: wraps the existing Flask app (app/server.py) unmodified,
the same "adapt around the app, don't rewrite it" pattern infra/lambda_api/
uses for the AWS Lambda deployment. Vercel's Python runtime looks for a
module-level WSGI `app` object in api/*.py files -- this just re-exports it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.server import app  # noqa: E402,F401
