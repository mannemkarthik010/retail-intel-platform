# Two Lambda functions with deliberately different packaging (see each
# Dockerfile's docstring and infra/trigger_lambda/handler.py's docstring
# for the full reasoning):
#   - api: container image (pandas/scikit-learn/flask exceed the 250MB zip
#     limit)
#   - trigger: plain zip (boto3 only, already in every Lambda Python
#     runtime -- no image needed)

# --------------------------------------------------------------------------
# API Lambda (infra/lambda_api/handler.py) -- serves app/server.py's Flask
# app behind API Gateway.
# --------------------------------------------------------------------------
resource "aws_lambda_function" "api" {
  function_name = "${local.name_prefix}-api"
  role          = aws_iam_role.lambda_api_exec.arn
  package_type  = "Image"
  image_uri     = var.api_lambda_image_uri
  timeout       = var.lambda_timeout_seconds
  memory_size   = var.lambda_memory_size

  environment {
    variables = merge(local.llm_env_vars, {
      ANTHROPIC_API_KEY_SECRET_ARN = aws_secretsmanager_secret.anthropic_api_key.arn
      OPENAI_API_KEY_SECRET_ARN    = aws_secretsmanager_secret.openai_api_key.arn
    })
  }

  depends_on = [aws_iam_role_policy_attachment.lambda_api_basic_logs, aws_cloudwatch_log_group.api_lambda]
}

resource "aws_lambda_permission" "api_gateway_invoke" {
  statement_id  = "AllowAPIGatewayInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.api.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_apigatewayv2_api.http_api.execution_arn}/*/*"
}

# --------------------------------------------------------------------------
# Trigger Lambda (infra/trigger_lambda/handler.py) -- fired nightly by
# EventBridge (see eventbridge.tf), calls sagemaker:CreateProcessingJob.
# --------------------------------------------------------------------------
data "archive_file" "trigger_lambda_zip" {
  type        = "zip"
  source_dir  = "${path.module}/../trigger_lambda"
  output_path = "${path.module}/build/trigger_lambda.zip"
  excludes    = ["__pycache__"]
}

resource "aws_lambda_function" "trigger" {
  function_name    = "${local.name_prefix}-trigger"
  role             = aws_iam_role.trigger_lambda_exec.arn
  handler          = "handler.lambda_handler"
  runtime          = "python3.11"
  filename         = data.archive_file.trigger_lambda_zip.output_path
  source_code_hash = data.archive_file.trigger_lambda_zip.output_base64sha256
  timeout          = 30
  memory_size      = 256

  environment {
    variables = {
      SAGEMAKER_IMAGE_URI             = var.sagemaker_processing_image_uri
      SAGEMAKER_EXECUTION_ROLE_ARN    = aws_iam_role.sagemaker_exec.arn
      CODE_S3_URI                     = "s3://${aws_s3_bucket.artifacts.bucket}/code/"
      OUTPUT_S3_URI                   = "s3://${aws_s3_bucket.artifacts.bucket}/processing-output/"
      SAGEMAKER_INSTANCE_TYPE         = var.sagemaker_instance_type
      SAGEMAKER_INSTANCE_COUNT        = tostring(var.sagemaker_instance_count)
      SAGEMAKER_VOLUME_SIZE_GB        = tostring(var.sagemaker_volume_size_gb)
      SAGEMAKER_MAX_RUNTIME_SECONDS   = tostring(var.sagemaker_max_runtime_seconds)
      JOB_NAME_PREFIX                 = "${local.name_prefix}-pipeline"
      CLOUDWATCH_METRIC_NAMESPACE     = "RetailIntel/${local.name_prefix}"
    }
  }

  depends_on = [aws_iam_role_policy_attachment.trigger_lambda_basic_logs, aws_cloudwatch_log_group.trigger_lambda]
}
