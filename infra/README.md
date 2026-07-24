# AWS deployment

This directory takes the Retail Demand Intelligence Platform (the rest of
this repo) and deploys it onto AWS using the services Condor's job
description names explicitly (SageMaker, Bedrock, Lambda), while keeping
every route/tool/model exactly what `docs/ARCHITECTURE.md` already
describes -- nothing here reimplements pipeline logic, it only adds cloud
plumbing around the same `src/`, `app/`, `scripts/` code the rest of this
repo runs locally.

## Why this exists

`docs/JOB_MAPPING.md` originally listed "AWS deployment" under **what this
project deliberately does NOT cover** -- this directory closes that gap
with real infrastructure-as-code and integration code, not just a
narrative description of how it *would* work.

## Architecture

```
                      EventBridge (nightly cron, default 03:00 UTC)
                                    |
                                    v
                      trigger Lambda (infra/trigger_lambda/)
                         boto3 sagemaker:CreateProcessingJob
                                    |
                                    v
              SageMaker Processing Job (infra/sagemaker/)
      container: infra/sagemaker/Dockerfile + processing_entrypoint.py
         |                         |                          |
         v                         v                          v
  data/generate_data.py   scripts/run_pipeline.py     scripts/run_monitoring_sim.py
  (or real data via the    (backtest, anomaly scan,    (fleet + per-series drift)
   raw-data input channel) forecast, intervals,               |
                            explainability)                   v
         |                         |              CloudWatch custom metrics
         +------------+------------+              (FleetMeanWAPEPercent,
                       v                            SeriesFlaggedForRetraining)
              S3 (processing-output/)                         |
              reports/*.csv, data/*.csv                       v
                       |                          CloudWatch alarm -> SNS
                       |  (baked into the API Lambda's image at build time --
                       |   see infra/lambda_api/Dockerfile; NOT read live from S3)
                       v
     API Gateway (HTTP API, $default stage)
                       |
                       v
     API Lambda (infra/lambda_api/handler.py)
     container image wrapping app/server.py's Flask app unmodified
     via a hand-written WSGI adapter (no aws-wsgi dependency)
                       |
                       v
     src/agent.py -- MockLLM (default) / AnthropicLLM / OpenAILLM / BedrockLLM
     (Bedrock Converse API tool-calling loop, same TOOL_SPECS registry as
      the other two real backends)
                       |
          (not yet wired -- see "What's NOT wired up" below)
                       v
     SageMaker Serverless Inference endpoint (optional, off by default)
     infra/sagemaker/inference/inference.py -- point GBM model only,
     for simulate_scenario / explain_forecast_drivers
```

Everything above the API Lambda box is the **offline/batch** side (same
split as `docs/ARCHITECTURE.md`'s system diagram); everything from the API
Lambda down is the **online/serving** side. That split doesn't change by
moving to AWS -- it's *why* the two can scale and fail independently on
AWS the same way they do locally.

## What's in this directory

| Path | What it is |
|---|---|
| `lambda_api/handler.py` | API Gateway (HTTP API, payload v2.0) <-> WSGI adapter wrapping `app/server.py`'s Flask app unmodified. Also resolves the two optional LLM API keys from Secrets Manager at cold start. |
| `lambda_api/Dockerfile` | Container image for the API Lambda (zip packaging can't fit pandas/scikit-learn under Lambda's 250MB limit). |
| `sagemaker/processing_entrypoint.py` | S3-aware wrapper around `data/generate_data.py` + `scripts/run_pipeline.py` + `scripts/run_monitoring_sim.py` for a SageMaker Processing Job. |
| `sagemaker/Dockerfile` | Container image for the Processing Job. |
| `sagemaker/inference/inference.py` | `model_fn`/`input_fn`/`predict_fn`/`output_fn` for the optional Serverless Inference endpoint (point GBM model only). |
| `trigger_lambda/handler.py` | Nightly EventBridge target; calls `sagemaker:CreateProcessingJob` with a fresh, timestamp-suffixed job name (Processing Jobs are transient/API-invoked, not a persistent Terraform resource). |
| `terraform/*.tf` | All the AWS infrastructure: S3, ECR, IAM, the two Lambdas, API Gateway, EventBridge, the optional SageMaker Serverless endpoint, CloudWatch, Secrets Manager. |
| `scripts/package_model.sh` | Packages `reports/models/gbm_point.joblib` + `feature_medians.joblib` + `inference.py` into the `model.tar.gz` the Serverless endpoint expects, and uploads it to S3. |
| `scripts/sync_code.sh` | Syncs `src/`, `scripts/`, `data/generate_data.py`, `infra/` to the S3 prefix the Processing Job reads as its "code" input. |

## Deploying this

You'll need: an AWS account, the AWS CLI configured with credentials,
Docker, and Terraform >= 1.5 installed locally -- **none of which are
available in the sandbox this repo was built in** (see "What's honestly
NOT verified" below), so these steps have not been run end-to-end by
anyone but you.

1. **Bootstrap the ECR repos and other non-image resources first.**
   `api_lambda_image_uri`/`sagemaker_processing_image_uri` are required
   variables, but the repos they point at don't exist until Terraform
   creates them -- so either apply once with placeholder URIs just to
   create the ECR repos (`terraform apply -target aws_ecr_repository.api
   -target aws_ecr_repository.sagemaker_processing`), or accept that the
   first full `apply` will fail at the Lambda/Processing-Job-image-pull
   step until you've pushed real images (see step 3 for why images can't
   be pushed until the repos exist).

2. **Build and push the two images:**
   ```bash
   aws ecr get-login-password --region <region> | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

   docker build -f infra/lambda_api/Dockerfile -t <account>.dkr.ecr.<region>.amazonaws.com/retail-intel-<env>-api:latest .
   docker push <account>.dkr.ecr.<region>.amazonaws.com/retail-intel-<env>-api:latest

   docker build -f infra/sagemaker/Dockerfile -t <account>.dkr.ecr.<region>.amazonaws.com/retail-intel-<env>-sagemaker-processing:latest .
   docker push <account>.dkr.ecr.<region>.amazonaws.com/retail-intel-<env>-sagemaker-processing:latest
   ```

3. **Copy `infra/terraform/terraform.tfvars.example` to
   `infra/terraform/terraform.tfvars`** and fill in the two image URIs
   from step 2 (plus any optional variables -- see the example file's
   comments).

4. **`terraform init && terraform plan && terraform apply`** from
   `infra/terraform/`.

5. **Sync the pipeline code to S3** so the nightly Processing Job has
   something to run:
   ```bash
   export CODE_S3_URI=$(terraform output -raw code_s3_uri)
   ../scripts/sync_code.sh
   ```

6. **(Optional) enable the Serverless Inference endpoint.** Run
   `python scripts/run_pipeline.py` locally first (or wait for the first
   nightly Processing Job) to produce `reports/models/gbm_point.joblib` +
   `feature_medians.joblib`, then:
   ```bash
   export MODEL_ARTIFACT_S3_URI=$(terraform output -raw model_artifact_s3_uri_hint)
   ./package_model.sh
   ```
   then set `enable_sagemaker_serverless_endpoint = true`,
   `model_data_s3_uri`, and `sagemaker_sklearn_inference_image_uri` in
   `terraform.tfvars` and re-apply.

7. **Trigger the first nightly run manually** rather than waiting for the
   schedule: `aws lambda invoke --function-name $(terraform output -raw
   trigger_lambda_name) /tmp/out.json`, then check
   `terraform output -raw cloudwatch_dashboard_url`.

8. Visit `terraform output -raw api_endpoint_url` for the dashboard, or
   `<that>/api/health` to smoke-test the API Lambda.

### Populating a real LLM API key

Don't put a real key in `terraform.tfvars` (even though it's marked
`sensitive`, it still lands in Terraform state in plaintext -- state
should be treated as sensitive too, but a separate file makes it too easy
to accidentally commit). Instead, after the first apply, run:
```bash
aws secretsmanager put-secret-value \
  --secret-id $(terraform output -raw anthropic_api_key_secret_arn) \
  --secret-string "sk-ant-..."
```
The next cold start of the API Lambda picks it up automatically (see
`lambda_api/handler.py::_hydrate_secret_env_var`).

## Cost estimate (rough, us-east-1, one operator's traffic)

This is a back-of-envelope estimate for evaluating whether this is worth
turning on, not a quote -- actual AWS pricing changes over time and this
wasn't run against the AWS Pricing API.

| Component | Rough monthly cost |
|---|---|
| API Lambda (container image, low request volume, a few hundred requests/day) | ~$0-1 (mostly within the always-free tier) |
| API Gateway HTTP API (same volume) | ~$0 (well under 1M requests/month) |
| Trigger Lambda (1 invocation/night) | ~$0 |
| SageMaker Processing Job (ml.m5.large, ~5-10 min/night) | ~$1-3/month |
| S3 (a few hundred MB of reports/data snapshots, 90-day expiry) | <$1/month |
| ECR (two small-ish images, <10 tags each after lifecycle policy) | ~$1/month |
| CloudWatch (logs + 2 custom metrics + 1 dashboard + 3 alarms) | ~$1-3/month |
| Secrets Manager (2 secrets) | ~$0.80/month |
| SageMaker Serverless Inference endpoint (optional, if enabled, low traffic) | ~$0-2/month (billed per-inference plus a per-GB-second compute charge; no charge when idle, unlike a real-time endpoint) |
| **Total** | **roughly $5-15/month** at this project's actual traffic scale |

The single biggest cost lever if this were scaled to production traffic
would be the API Lambda's memory/duration (currently 1024MB/30s, tuned for
a demo, not load-tested) and, if enabled, the Serverless endpoint's
concurrency ceiling.

## What's honestly NOT verified

This whole `infra/` addition was written in the same sandboxed cloud
container the rest of this project was built in (see
`docs/ARCHITECTURE.md`'s "Substitutions made because of this build
environment" section for the earlier examples of this same disclosure
pattern -- Docker, CI, gunicorn). Direct testing in this environment
confirmed:

- No `aws` CLI, no `terraform`/`tofu` binary, no `cdk` -- none installable
  (`apt-get install terraform-config-inspect`, a lightweight HCL
  syntax-checker, also failed with a 403 from `security.ubuntu.com`).
- `boto3` and `moto` (AWS's official SDK and its mocking library) are not
  installed and not pip-installable here (`ERROR: Could not find a version
  that satisfies the requirement boto3`).
- Direct requests to real AWS API endpoints (`sts.amazonaws.com`,
  `bedrock-runtime.us-east-1.amazonaws.com`) return `403 Forbidden` at the
  sandbox's network boundary -- not a real AWS auth rejection, just this
  environment's egress restriction.
- The `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` environment variables
  present in this sandbox are placeholder/proxy-injected values, not real,
  usable AWS credentials.
- No Docker daemon is running here (`docker info` fails to connect to the
  socket) -- consistent with the same limitation already disclosed for
  this repo's root `Dockerfile`/CI workflow.

Given that, here's the honest split between what's real and what's
"written to spec, not run":

**What IS verified** (by an actual, passing test run in this sandbox --
`python -m unittest discover -s tests`, 95 tests):
- `src/agent.py::BedrockLLM`'s dispatch logic, tool-config conversion, and
  full tool-calling loop (mocked `boto3`, no real Bedrock call).
- `infra/lambda_api/handler.py`'s API Gateway <-> WSGI event translation
  and secret-hydration logic, exercised against the real
  `app.server.app` Flask object (so a real route bug would surface here).
- `infra/sagemaker/processing_entrypoint.py`'s input-staging,
  read-only-workspace fallback, output-sync, and CloudWatch-metric-publish
  logic (temp directories standing in for `/opt/ml/processing/*`, mocked
  `boto3`/`cloudwatch`).
- `infra/trigger_lambda/handler.py`'s `CreateProcessingJob` request
  construction (mocked `boto3`, no real SageMaker call).
- `infra/sagemaker/inference/inference.py`'s four contract functions
  against a fake model object and real joblib files.

**What is NOT verified** (written to the documented AWS contract/API
shape, but never actually run against real AWS infrastructure from this
sandbox):
- That `terraform init`/`plan`/`apply` actually succeeds against a real
  AWS account -- there's no `terraform validate` pass behind this, only
  careful manual authoring and brace/paren-balance checks.
- That the two Dockerfiles actually build (no Docker daemon here).
- That a real API Gateway request reaches the Lambda and gets the shape
  of event this adapter assumes (built from AWS's documented HTTP API
  v2.0 payload format, not observed from a live invocation).
- That a real SageMaker Processing Job accepts the `CreateProcessingJob`
  request shape `trigger_lambda/handler.py` builds, or that the
  read-only-input-channel assumption `_ensure_writable_code_root()` is
  based on holds exactly as documented for every instance type.
- That the Serverless Inference endpoint, if enabled, actually loads and
  serves the model artifact correctly end-to-end.
- Whether the IAM policies here are minimal-but-sufficient (some, like
  `ecr:GetAuthorizationToken` and `cloudwatch:PutMetricData`, are
  necessarily `resources = ["*"]` since those actions don't support
  resource-level scoping -- documented inline in `iam.tf` -- but a real
  IAM Access Analyzer pass hasn't been run against this).

The honest framing that holds up in an interview, extending
`docs/JOB_MAPPING.md`'s existing one: *"I designed and wrote the full AWS
deployment -- Bedrock as a third pluggable LLM backend behind the same
tool registry, a hand-written Lambda/API-Gateway adapter for the existing
Flask app, a SageMaker Processing Job for the nightly batch pipeline with
a real CloudWatch metric wired to a real alarm, and a Serverless Inference
endpoint scoped to the two tools that actually need live recomputation --
and everything with real test coverage against mocked AWS calls. What I
haven't done is run `terraform apply` against a real AWS account, because
this was built in a sandboxed environment with no AWS network access at
all. That's a real gap, and I know exactly where the risk in closing it
would be: the Dockerfiles actually building, the Processing Job's
container-entrypoint/input-channel assumptions holding up, and the IAM
policies being exactly right on the first try."*
