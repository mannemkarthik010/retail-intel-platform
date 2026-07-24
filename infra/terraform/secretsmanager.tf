# Two secrets, one per optional LLM provider key. Terraform always creates
# the secret itself and a placeholder value ("REPLACE_ME_MANUALLY") when no
# real key is supplied via the anthropic_api_key/openai_api_key variables --
# this means `terraform apply` never fails for lack of a key (both LLM
# backends are optional; src/agent.py::get_llm_backend() falls back to the
# deterministic MockLLM if neither ends up set), and a real key can be
# supplied later via `terraform apply -var anthropic_api_key=...` or
# directly via `aws secretsmanager put-secret-value` without any other
# infrastructure change.
#
# infra/lambda_api/handler.py::_hydrate_secret_env_var() reads these at
# Lambda cold start via the *_SECRET_ARN environment variables lambda.tf
# sets -- the actual key value is never written into the Lambda function's
# own environment-variable configuration (which is visible in plaintext in
# the Lambda console and in Terraform state either way, so this mainly
# avoids ALSO duplicating it into Lambda's config -- see that module's
# docstring for the full reasoning).

resource "aws_secretsmanager_secret" "anthropic_api_key" {
  name        = "${local.name_prefix}-anthropic-api-key"
  description = "Anthropic API key for src/agent.py::AnthropicLLM. Placeholder until set."
}

resource "aws_secretsmanager_secret_version" "anthropic_api_key" {
  secret_id     = aws_secretsmanager_secret.anthropic_api_key.id
  secret_string = var.anthropic_api_key != "" ? var.anthropic_api_key : "REPLACE_ME_MANUALLY"

  lifecycle {
    # Don't let a later `terraform apply` (run without -var
    # anthropic_api_key=... set again) stomp a real value someone put here
    # manually via the AWS CLI/console after the initial apply.
    ignore_changes = [secret_string]
  }
}

resource "aws_secretsmanager_secret" "openai_api_key" {
  name        = "${local.name_prefix}-openai-api-key"
  description = "OpenAI API key for src/agent.py::OpenAILLM. Placeholder until set."
}

resource "aws_secretsmanager_secret_version" "openai_api_key" {
  secret_id     = aws_secretsmanager_secret.openai_api_key.id
  secret_string = var.openai_api_key != "" ? var.openai_api_key : "REPLACE_ME_MANUALLY"

  lifecycle {
    ignore_changes = [secret_string]
  }
}
