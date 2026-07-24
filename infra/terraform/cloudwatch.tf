# Log groups (declared explicitly, with a retention policy, rather than
# left to Lambda's default of "auto-created, never expires" -- both
# Lambda functions depend on their log group so Terraform creates it
# before the function's first invocation could otherwise auto-create one
# with the default, un-managed retention setting).
resource "aws_cloudwatch_log_group" "api_lambda" {
  name              = "/aws/lambda/${local.name_prefix}-api"
  retention_in_days = var.log_retention_days
}

resource "aws_cloudwatch_log_group" "trigger_lambda" {
  name              = "/aws/lambda/${local.name_prefix}-trigger"
  retention_in_days = var.log_retention_days
}

# SNS topic for pipeline-failure/model-drift alarms. No subscription is
# created if alarm_email is left empty -- the topic still exists so
# alarms have somewhere to notify, and you can subscribe to it later
# without any other change.
resource "aws_sns_topic" "alerts" {
  name = "${local.name_prefix}-alerts"
}

resource "aws_sns_topic_subscription" "alerts_email" {
  count     = var.alarm_email != "" ? 1 : 0
  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alarm_email
}

# --------------------------------------------------------------------------
# Alarms
# --------------------------------------------------------------------------

# The trigger Lambda failing means tonight's batch pipeline never ran at
# all -- reports/*.csv artifacts go stale and the API keeps serving
# yesterday's (or older) forecasts without anyone noticing, since nothing
# else would surface that silently.
resource "aws_cloudwatch_metric_alarm" "trigger_lambda_errors" {
  alarm_name          = "${local.name_prefix}-trigger-lambda-errors"
  alarm_description   = "The nightly batch-pipeline trigger Lambda (infra/trigger_lambda/handler.py) failed to invoke sagemaker:CreateProcessingJob."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.trigger.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 0
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_metric_alarm" "api_lambda_errors" {
  alarm_name          = "${local.name_prefix}-api-lambda-errors"
  alarm_description   = "The API Lambda (infra/lambda_api/handler.py) is returning errors to API Gateway."
  namespace           = "AWS/Lambda"
  metric_name         = "Errors"
  dimensions          = { FunctionName = aws_lambda_function.api.function_name }
  statistic           = "Sum"
  period              = 300
  evaluation_periods   = 2
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

# Watches the metric infra/sagemaker/processing_entrypoint.py::
# _publish_monitoring_metrics publishes after every nightly run -- this is
# the fleet-level counterpart to docs/EVAL_REPORT.md §3's per-series
# monitoring table; a real jump here means the deployed models are
# collectively drifting, not just one series.
resource "aws_cloudwatch_metric_alarm" "fleet_model_drift" {
  alarm_name          = "${local.name_prefix}-fleet-model-drift"
  alarm_description   = "Fleet-mean WAPE (published nightly by the batch pipeline) has jumped meaningfully above its recent baseline -- see docs/EVAL_REPORT.md §3 for what this metric means and infra/sagemaker/processing_entrypoint.py::_publish_monitoring_metrics for how it's published."
  namespace           = "RetailIntel/${local.name_prefix}"
  metric_name         = "FleetMeanWAPEPercent"
  statistic           = "Average"
  period              = 86400 # one nightly run per day
  evaluation_periods  = 1
  # docs/EVAL_REPORT.md's actual backtest fleet mean WAPE is ~8%; alarming
  # above 15% is a deliberately loose first threshold meant to be tuned
  # after a few weeks of real nightly values, not treated as a
  # scientifically derived cutoff.
  threshold           = 15
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching" # the metric only exists once the first nightly run has completed
  alarm_actions       = [aws_sns_topic.alerts.arn]
}

resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${local.name_prefix}-overview"
  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "API Lambda: invocations / errors / duration"
          view    = "timeSeries"
          region  = var.aws_region
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", aws_lambda_function.api.function_name],
            ["AWS/Lambda", "Errors", "FunctionName", aws_lambda_function.api.function_name],
            ["AWS/Lambda", "Duration", "FunctionName", aws_lambda_function.api.function_name],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title   = "API Gateway: requests / 4xx / 5xx"
          view    = "timeSeries"
          region  = var.aws_region
          metrics = [
            ["AWS/ApiGateway", "Count", "ApiId", aws_apigatewayv2_api.http_api.id],
            ["AWS/ApiGateway", "4xx", "ApiId", aws_apigatewayv2_api.http_api.id],
            ["AWS/ApiGateway", "5xx", "ApiId", aws_apigatewayv2_api.http_api.id],
          ]
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "Nightly batch pipeline: trigger Lambda invocations / errors"
          view    = "timeSeries"
          region  = var.aws_region
          metrics = [
            ["AWS/Lambda", "Invocations", "FunctionName", aws_lambda_function.trigger.function_name],
            ["AWS/Lambda", "Errors", "FunctionName", aws_lambda_function.trigger.function_name],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 6
        width  = 12
        height = 6
        properties = {
          title   = "Fleet mean WAPE % (nightly) and series flagged for retraining"
          view    = "timeSeries"
          region  = var.aws_region
          metrics = [
            ["RetailIntel/${local.name_prefix}", "FleetMeanWAPEPercent"],
            ["RetailIntel/${local.name_prefix}", "SeriesFlaggedForRetraining"],
          ]
        }
      },
    ]
  })
}
