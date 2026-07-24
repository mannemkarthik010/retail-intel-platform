output "api_endpoint_url" {
  description = "Base URL of the deployed API + dashboard. Same routes as `python app/server.py` locally (e.g. <this>/api/health, <this>/, <this>/api/ask)."
  value       = aws_apigatewayv2_api.http_api.api_endpoint
}

output "api_ecr_repository_url" {
  description = "Push the image built from infra/lambda_api/Dockerfile here."
  value       = aws_ecr_repository.api.repository_url
}

output "sagemaker_processing_ecr_repository_url" {
  description = "Push the image built from infra/sagemaker/Dockerfile here."
  value       = aws_ecr_repository.sagemaker_processing.repository_url
}

output "artifacts_bucket_name" {
  value = aws_s3_bucket.artifacts.bucket
}

output "code_s3_uri" {
  description = "Sync the repo here with infra/scripts/sync_code.sh before the first nightly run (or any code change)."
  value       = "s3://${aws_s3_bucket.artifacts.bucket}/code/"
}

output "model_artifact_s3_uri_hint" {
  description = "Where infra/scripts/package_model.sh expects to upload model.tar.gz (only relevant if enable_sagemaker_serverless_endpoint is true) -- pass this as MODEL_ARTIFACT_S3_URI."
  value       = "s3://${aws_s3_bucket.artifacts.bucket}/models/model.tar.gz"
}

output "trigger_lambda_name" {
  value = aws_lambda_function.trigger.function_name
}

output "trigger_lambda_arn" {
  value = aws_lambda_function.trigger.arn
}

output "api_lambda_name" {
  value = aws_lambda_function.api.function_name
}

output "nightly_schedule_rule_name" {
  value = aws_cloudwatch_event_rule.nightly_batch_pipeline.name
}

output "sagemaker_execution_role_arn" {
  value = aws_iam_role.sagemaker_exec.arn
}

output "anthropic_api_key_secret_arn" {
  value = aws_secretsmanager_secret.anthropic_api_key.arn
}

output "openai_api_key_secret_arn" {
  value = aws_secretsmanager_secret.openai_api_key.arn
}

output "cloudwatch_dashboard_url" {
  value = "https://${var.aws_region}.console.aws.amazon.com/cloudwatch/home?region=${var.aws_region}#dashboards:name=${aws_cloudwatch_dashboard.main.dashboard_name}"
}

output "serverless_inference_endpoint_name" {
  description = "Only set when enable_sagemaker_serverless_endpoint = true."
  value       = var.enable_sagemaker_serverless_endpoint ? aws_sagemaker_endpoint.point_gbm[0].name : null
}
