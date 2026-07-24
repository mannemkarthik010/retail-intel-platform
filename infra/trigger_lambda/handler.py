"""Nightly-schedule Lambda that kicks off the batch pipeline as a SageMaker
Processing Job.

Why this is a Lambda at all, instead of Terraform managing the Processing
Job directly: Processing Jobs are transient, API-invoked resources -- each
nightly run is its own `CreateProcessingJob` call with a unique job name,
not a persistent piece of infrastructure a Terraform resource can represent
1:1 (there's no "the" processing job to converge to, only "the next one").
So Terraform manages everything ABOUT how the job runs (the container
image in ECR, the IAM role it assumes, the S3 buckets it reads/writes) and
this tiny Lambda -- invoked on a schedule by
infra/terraform/eventbridge.tf's EventBridge rule -- calls
`sagemaker:CreateProcessingJob` each time with a fresh, timestamp-suffixed
job name.

Deliberately just boto3 (already present in every Lambda Python runtime,
so this needs zero extra dependencies / no container image, unlike the API
Lambda) -- see infra/terraform/lambda.tf for how this is packaged as a
plain zip via Terraform's `archive_file` data source.

NOT invoked against a real EventBridge/Lambda/SageMaker setup in this build
sandbox -- see infra/README.md's disclosure section. What IS verified
(tests/test_trigger_lambda.py) is that this handler builds a correct
CreateProcessingJob call from its configured environment variables, using
the same `unittest.mock.patch.dict('sys.modules', {'boto3': MagicMock()})`
technique src/agent.py's BedrockLLM tests use, so it's checked without
boto3 needing to be installed or any real AWS call happening.
"""
import os


def _job_name(now_iso: str) -> str:
    """SageMaker Processing Job names must be <= 63 chars, alphanumeric +
    hyphens only. `now_iso` is passed in (rather than computed here with
    datetime.utcnow()) purely so this function stays trivially unit
    testable with a fixed, predictable value."""
    prefix = os.environ.get("JOB_NAME_PREFIX", "retail-intel-pipeline")
    safe_ts = now_iso.replace(":", "-").replace(".", "-")
    return f"{prefix}-{safe_ts}"[:63].rstrip("-")


def _build_create_processing_job_kwargs(now_iso: str) -> dict:
    """Assembles the CreateProcessingJob request from environment variables
    Terraform sets on this Lambda (see infra/terraform/lambda.tf) --
    keeping all the AWS-account-specific values (role ARN, image URI,
    bucket names) out of code and in Terraform-managed config."""
    image_uri = os.environ["SAGEMAKER_IMAGE_URI"]
    role_arn = os.environ["SAGEMAKER_EXECUTION_ROLE_ARN"]
    code_s3_uri = os.environ["CODE_S3_URI"]
    output_s3_uri = os.environ["OUTPUT_S3_URI"]
    instance_type = os.environ.get("SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")
    instance_count = int(os.environ.get("SAGEMAKER_INSTANCE_COUNT", "1"))
    volume_size_gb = int(os.environ.get("SAGEMAKER_VOLUME_SIZE_GB", "10"))
    max_runtime_seconds = int(os.environ.get("SAGEMAKER_MAX_RUNTIME_SECONDS", str(60 * 60)))

    # Passed through to the Processing Job container's own environment --
    # infra/sagemaker/processing_entrypoint.py::_publish_monitoring_metrics
    # reads CLOUDWATCH_METRIC_NAMESPACE to know where to publish (and
    # no-ops if it's absent, e.g. a manual local dry run).
    container_env = {}
    if os.environ.get("CLOUDWATCH_METRIC_NAMESPACE"):
        container_env["CLOUDWATCH_METRIC_NAMESPACE"] = os.environ["CLOUDWATCH_METRIC_NAMESPACE"]

    return {
        "ProcessingJobName": _job_name(now_iso),
        "RoleArn": role_arn,
        "AppSpecification": {
            "ImageUri": image_uri,
            "ContainerEntrypoint": ["python3", "/opt/ml/processing/input/code/infra/sagemaker/processing_entrypoint.py"],
        },
        **({"Environment": container_env} if container_env else {}),
        "ProcessingInputs": [{
            "InputName": "code",
            "S3Input": {
                "S3Uri": code_s3_uri,
                "LocalPath": "/opt/ml/processing/input/code",
                "S3DataType": "S3Prefix",
                "S3InputMode": "File",
                "S3DataDistributionType": "FullyReplicated",
            },
        }],
        "ProcessingOutputConfig": {
            "Outputs": [{
                "OutputName": "reports-and-data",
                "S3Output": {
                    "S3Uri": output_s3_uri,
                    "LocalPath": "/opt/ml/processing/output",
                    "S3UploadMode": "EndOfJob",
                },
            }],
        },
        "ProcessingResources": {
            "ClusterConfig": {
                "InstanceType": instance_type,
                "InstanceCount": instance_count,
                "VolumeSizeInGB": volume_size_gb,
            },
        },
        "StoppingCondition": {"MaxRuntimeInSeconds": max_runtime_seconds},
    }


def lambda_handler(event, context=None):
    """The Lambda entrypoint (Terraform sets this as the handler:
    infra/trigger_lambda/handler.lambda_handler, see
    infra/terraform/lambda.tf). `event` is whatever EventBridge passes on
    the scheduled invocation (unused -- this Lambda has exactly one job to
    do, kick off tonight's Processing Job)."""
    import boto3  # local import: keeps this module importable (and its
                   # pure request-building logic testable) without boto3
                   # installed, mirroring src/agent.py::BedrockLLM.
    from datetime import datetime, timezone

    now_iso = datetime.now(timezone.utc).isoformat()
    kwargs = _build_create_processing_job_kwargs(now_iso)

    client = boto3.client("sagemaker", region_name=os.environ.get("AWS_REGION", "us-east-1"))
    response = client.create_processing_job(**kwargs)
    return {
        "processingJobName": kwargs["ProcessingJobName"],
        "processingJobArn": response.get("ProcessingJobArn"),
    }
