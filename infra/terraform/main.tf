# Shared locals/data sources. The provider block lives in versions.tf.

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

locals {
  name_prefix = "${var.project_name}-${var.environment}"

  # LLM backend selection env vars shared between the API Lambda
  # (infra/lambda_api/handler.py -> app.server -> src/agent.py) and
  # documented in docs/ARCHITECTURE.md's "The agent layer" section.
  llm_env_vars = merge(
    var.llm_backend != "" ? { LLM_BACKEND = var.llm_backend } : {},
    var.bedrock_model_id != "" ? { BEDROCK_MODEL_ID = var.bedrock_model_id } : {},
  )
}
