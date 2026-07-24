# Container image for the API Lambda (infra/lambda_api/handler.py).
#
# Why a container image instead of a zip archive: Lambda zip deployments
# are capped at 250MB unzipped, and this app's dependency tree (pandas,
# numpy, scikit-learn, scipy -- needed because app/server.py's DataStore
# unpickles joblib-persisted sklearn models, and src/agent.py's
# simulate_scenario/explain_forecast_drivers tools call back into
# scikit-learn directly) comfortably exceeds that on its own. Lambda
# container images support up to 10GB, which is the standard, documented
# AWS pattern for exactly this situation. The trigger Lambda
# (infra/trigger_lambda/) has no such dependency and stays a plain zip --
# see infra/terraform/lambda.tf.
#
# Build/push (see infra/README.md for the full walkthrough):
#   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com
#   docker build -f infra/lambda_api/Dockerfile -t <account>.dkr.ecr.<region>.amazonaws.com/retail-intel-api:latest .
#   docker push <account>.dkr.ecr.<region>.amazonaws.com/retail-intel-api:latest
#
# NOT built or pushed in this build sandbox -- no Docker daemon is running
# here (see infra/README.md's disclosure section) -- so this Dockerfile
# follows the standard AWS base-image contract but hasn't been observed to
# actually build.
FROM public.ecr.aws/lambda/python:3.11

WORKDIR ${LAMBDA_TASK_ROOT}

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bring in the whole repo (src/, app/, data/, scripts/, infra/) so the
# Lambda serves the exact same app/server.py Flask app the Dockerfile at
# the repo root runs locally.
COPY src/ ./src/
COPY app/ ./app/
COPY data/ ./data/
COPY scripts/ ./scripts/
COPY infra/ ./infra/

# Bake the pipeline artifacts in at build time, same as the repo-root
# Dockerfile -- so the Lambda's reports/ is a specific, versioned snapshot
# rather than something computed on cold start.
RUN python data/generate_data.py \
    && python scripts/run_pipeline.py \
    && python scripts/run_monitoring_sim.py

CMD ["infra.lambda_api.handler.lambda_handler"]
