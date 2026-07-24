# Nightly schedule -> trigger Lambda -> sagemaker:CreateProcessingJob.
# See infra/trigger_lambda/handler.py's module docstring for why this is a
# Lambda-mediated trigger rather than Terraform managing a Processing Job
# resource directly (Processing Jobs are transient/API-invoked, not a
# persistent resource Terraform can converge to).
resource "aws_cloudwatch_event_rule" "nightly_batch_pipeline" {
  name                = "${local.name_prefix}-nightly-batch-pipeline"
  description         = "Triggers the nightly SageMaker Processing Job that runs the offline pipeline (backtest, anomaly scan, forecast, monitoring sim)."
  schedule_expression = var.batch_schedule_expression
}

resource "aws_cloudwatch_event_target" "trigger_lambda" {
  rule = aws_cloudwatch_event_rule.nightly_batch_pipeline.name
  arn  = aws_lambda_function.trigger.arn
}

resource "aws_lambda_permission" "eventbridge_invoke_trigger" {
  statement_id  = "AllowEventBridgeInvoke"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.trigger.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.nightly_batch_pipeline.arn
}
