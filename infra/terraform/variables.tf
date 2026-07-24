variable "project_name" {
  description = "Short name used to prefix every resource this stack creates."
  type        = string
  default     = "retail-intel"
}

variable "environment" {
  description = "Deployment environment name (e.g. dev, staging, prod) -- part of the resource-name prefix."
  type        = string
  default     = "dev"
}

variable "aws_region" {
  description = <<-EOT
    AWS region to deploy into. Bedrock model availability varies by region --
    check https://docs.aws.amazon.com/bedrock/latest/userguide/models-regions.html
    before changing this if bedrock_model_id is set.
  EOT
  type        = string
  default     = "us-east-1"
}

# --------------------------------------------------------------------------
# API Lambda (infra/lambda_api/) -- container image, see infra/lambda_api/Dockerfile
# --------------------------------------------------------------------------
variable "api_lambda_image_uri" {
  description = <<-EOT
    Full ECR image URI (including tag/digest) for the API Lambda, built from
    infra/lambda_api/Dockerfile and pushed to the ECR repo this stack creates
    (see aws_ecr_repository.api in ecr.tf). Required -- there is no sensible
    default, since the ECR repo this stack creates is empty until you build
    and push an image into it. See infra/README.md's deployment steps.
  EOT
  type        = string
}

variable "lambda_memory_size" {
  description = "Memory (MB) allocated to the API Lambda. Also proportionally scales its CPU share."
  type        = number
  default     = 1024
}

variable "lambda_timeout_seconds" {
  description = <<-EOT
    Timeout for the API Lambda. simulate_scenario/explain_forecast_drivers
    recompute a prediction live (see docs/ARCHITECTURE.md), so this is a
    little higher than a typical thin API's default.
  EOT
  type        = number
  default     = 30
}

# --------------------------------------------------------------------------
# LLM backend selection (src/agent.py) -- see docs/ARCHITECTURE.md's "The agent layer"
# --------------------------------------------------------------------------
variable "llm_backend" {
  description = <<-EOT
    Optional explicit override for src/agent.py::get_llm_backend()'s
    LLM_BACKEND env var (anthropic|openai|bedrock|mock). Leave empty to use
    the default precedence (Anthropic key present -> OpenAI key present ->
    Bedrock model ID present -> mock).
  EOT
  type        = string
  default     = ""
}

variable "bedrock_model_id" {
  description = <<-EOT
    Bedrock model ID for src/agent.py::BedrockLLM (e.g.
    anthropic.claude-3-5-sonnet-20241022-v2:0). Leave empty to not enable
    the Bedrock backend at all.
  EOT
  type        = string
  default     = ""
}

variable "anthropic_api_key" {
  description = <<-EOT
    Optional Anthropic API key, stored in Secrets Manager (never in Lambda's
    own plaintext environment-variable configuration -- see
    infra/lambda_api/handler.py's secret-hydration logic). Leave empty to
    skip creating a real secret value (a placeholder is stored instead,
    which you can update later via `aws secretsmanager put-secret-value`).
  EOT
  type        = string
  default     = ""
  sensitive   = true
}

variable "openai_api_key" {
  description = "Optional OpenAI API key -- same handling as anthropic_api_key above."
  type        = string
  default     = ""
  sensitive   = true
}

# --------------------------------------------------------------------------
# SageMaker Processing (nightly batch pipeline) -- infra/sagemaker/, infra/trigger_lambda/
# --------------------------------------------------------------------------
variable "sagemaker_processing_image_uri" {
  description = <<-EOT
    Full ECR image URI for the SageMaker Processing container, built from
    infra/sagemaker/Dockerfile and pushed to the ECR repo this stack creates
    (see aws_ecr_repository.sagemaker_processing in ecr.tf). Required for
    the same reason as api_lambda_image_uri above.
  EOT
  type        = string
}

variable "batch_schedule_expression" {
  description = "EventBridge schedule expression for the nightly batch pipeline trigger. Default: 03:00 UTC every day."
  type        = string
  default     = "cron(0 3 * * ? *)"
}

variable "sagemaker_instance_type" {
  description = <<-EOT
    Instance type for the nightly Processing Job. ml.m5.large comfortably
    covers this project's 240-series backtest + anomaly scan + monitoring
    sim; a real multi-thousand-series fleet would need a bigger instance or
    the recursive-partitioning approach noted in docs/EVAL_REPORT.md.
  EOT
  type        = string
  default     = "ml.m5.large"
}

variable "sagemaker_instance_count" {
  type    = number
  default = 1
}

variable "sagemaker_volume_size_gb" {
  type    = number
  default = 10
}

variable "sagemaker_max_runtime_seconds" {
  description = "Hard ceiling on the Processing Job's runtime, after which SageMaker force-stops it."
  type        = number
  default     = 3600
}

# --------------------------------------------------------------------------
# SageMaker Serverless Inference endpoint (optional) -- see
# infra/sagemaker/inference/inference.py's docstring for why this exists
# and what it's deliberately scoped to.
# --------------------------------------------------------------------------
variable "enable_sagemaker_serverless_endpoint" {
  description = <<-EOT
    Whether to create the optional SageMaker Serverless Inference endpoint
    for the point GBM model (used by simulate_scenario/explain_forecast_drivers).
    Off by default since it requires model_data_s3_uri to already point at a
    real packaged model artifact (see infra/scripts/package_model.sh) --
    turning this on before that exists will fail to apply.
  EOT
  type        = bool
  default     = false
}

variable "sagemaker_sklearn_inference_image_uri" {
  description = <<-EOT
    AWS's prebuilt scikit-learn inference container image URI for this
    region (the exact URI is account/region-specific -- see
    https://github.com/aws/deep-learning-containers/blob/master/available_images.md
    and the sagemaker-scikit-learn-container repo's released tags).
    Required if enable_sagemaker_serverless_endpoint is true.
  EOT
  type        = string
  default     = ""
}

variable "model_data_s3_uri" {
  description = <<-EOT
    S3 URI of the model.tar.gz produced by infra/scripts/package_model.sh
    (gbm_point.joblib + feature_medians.joblib + code/inference.py).
    Required if enable_sagemaker_serverless_endpoint is true.
  EOT
  type        = string
  default     = ""
}

variable "serverless_memory_size_mb" {
  description = "Memory (MB) for the Serverless Inference endpoint. Must be one of SageMaker's allowed values (1024, 2048, 3072, 4096, 5120, 6144)."
  type        = number
  default     = 2048
}

variable "serverless_max_concurrency" {
  type    = number
  default = 5
}

# --------------------------------------------------------------------------
# Observability
# --------------------------------------------------------------------------
variable "log_retention_days" {
  type    = number
  default = 14
}

variable "alarm_email" {
  description = <<-EOT
    Optional email address to subscribe to the pipeline-failure/model-drift
    SNS topic. Leave empty to create the topic with no subscribers (you can
    add one later in the console, or via a separate
    aws_sns_topic_subscription resource).
  EOT
  type        = string
  default     = ""
}
