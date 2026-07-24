"""Inference entrypoint for the optional SageMaker Serverless Inference
endpoint (infra/terraform/sagemaker.tf), served via AWS's prebuilt
scikit-learn inference container (see infra/README.md for the exact image
URI variable). Implements the four functions that container's contract
expects: model_fn, input_fn, predict_fn, output_fn.

Why an endpoint exists at all, when every other agent tool
(get_forecast/get_anomalies/explain_change/top_movers) only ever reads a
precomputed reports/*.csv artifact: two tools --
`simulate_scenario`/`src/scenario.py` and
`explain_forecast_drivers`/`src/explain.py`'s occlusion attribution -- are
the ONE part of this system that recomputes a prediction on demand for a
hypothetical/counterfactual input that was never scored offline by
definition (see docs/ARCHITECTURE.md's "The agent layer" section). That is
a genuine real-time inference need, which is exactly what SageMaker
Serverless Inference is for, and it's why this endpoint is scoped
narrowly to the point GBM model those two tools need rather than
standing up an endpoint for the whole pipeline.

Deliberate, disclosed tradeoff: at this project's actual traffic scale (a
handful of forecasting-analyst queries per day, not a high-throughput
production service), bundling the ~1MB point-GBM joblib artifact directly
into the API Lambda's container image would be simpler and cheaper than a
separate SageMaker endpoint -- no cold-start-to-a-second-service, no extra
piece of infrastructure to operate. A real Serverless Inference endpoint
is still built and wired here specifically because Condor's job
description names SageMaker explicitly as part of their stack, and the
architecture is worth showing regardless of whether this exact traffic
volume strictly needs it. infra/README.md says this plainly rather than
implying the endpoint was the only reasonable choice.

Model artifact contract (see infra/scripts/package_model.sh): the
model.tar.gz this expects at SM_MODEL_DIR (SageMaker's env var,
conventionally /opt/ml/model) contains gbm_point.joblib and
feature_medians.joblib -- the exact two files
scripts/run_pipeline.py already persists via joblib for
src/scenario.py/src/explain.py's local (non-endpoint) use. This script
doesn't retrain or change that artifact at all, only serves it.

NOT deployed against, or invoked through, a real SageMaker Serverless
Inference endpoint in this build sandbox -- see infra/README.md's
disclosure section. This follows the documented scikit-learn inference
container contract (https://github.com/aws/sagemaker-scikit-learn-container)
but that's a "written to spec" claim, not an "observed serving real
traffic" claim.
"""
import json
import os
from pathlib import Path

import joblib
import numpy as np


def model_fn(model_dir):
    """Called once per container/worker startup. Loads the exact same two
    joblib artifacts src/scenario.py and src/explain.py load locally, so
    the endpoint's predictions match a local run bit-for-bit given the
    same feature row."""
    model_dir = Path(model_dir)
    point_model = joblib.load(model_dir / "gbm_point.joblib")
    feature_medians = joblib.load(model_dir / "feature_medians.joblib")
    return {"point_model": point_model, "feature_medians": feature_medians}


def input_fn(request_body, request_content_type):
    """Accepts a JSON body: {"feature_columns": [...], "rows": [[...], ...]}
    -- a plain list-of-lists rather than a pickled DataFrame, so the
    request stays inspectable/auditable in CloudWatch logs and doesn't
    require the caller to have pandas installed."""
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type} (expected application/json)")
    payload = json.loads(request_body)
    feature_columns = payload["feature_columns"]
    rows = payload["rows"]
    return {"feature_columns": feature_columns, "rows": np.array(rows, dtype=float)}


def predict_fn(input_data, model):
    """Point-forecast prediction only -- this endpoint intentionally does
    NOT also serve the q10/q90 quantile models; simulate_scenario's
    baseline-vs-scenario delta and explain_forecast_drivers' occlusion
    attribution both only ever need the point model (see src/scenario.py,
    src/explain.py). Widening this endpoint to the quantile models too
    would be a natural next step, not done here to keep this one endpoint
    scoped to what the two calling tools actually use today."""
    point_model = model["point_model"]
    predictions = point_model.predict(input_data["rows"])
    return predictions.tolist()


def output_fn(prediction, response_content_type):
    if response_content_type != "application/json":
        raise ValueError(f"Unsupported accept type: {response_content_type} (expected application/json)")
    return json.dumps({"predictions": prediction})
