# Optional SageMaker Serverless Inference endpoint for the point GBM model
# (infra/sagemaker/inference/inference.py). Off by default
# (enable_sagemaker_serverless_endpoint = false) since it requires a real
# packaged model artifact already sitting at model_data_s3_uri (see
# infra/scripts/package_model.sh) -- there's nothing sensible to deploy
# before that exists. See inference.py's module docstring for the full
# reasoning on why this endpoint exists and what it's deliberately scoped
# to (just the point model, just for simulate_scenario/
# explain_forecast_drivers -- not yet wired into the agent's actual tool
# calls, which is disclosed rather than implied-done in infra/README.md).
#
# The nightly Processing Job (infra/sagemaker/, trigger_lambda/) is
# unrelated infrastructure and is NOT gated by this flag -- it always
# exists once the stack is applied. Only this real-time endpoint is optional.

resource "aws_sagemaker_model" "point_gbm" {
  count              = var.enable_sagemaker_serverless_endpoint ? 1 : 0
  name               = "${local.name_prefix}-point-gbm"
  execution_role_arn = aws_iam_role.sagemaker_exec.arn

  primary_container {
    image          = var.sagemaker_sklearn_inference_image_uri
    model_data_url = var.model_data_s3_uri
    environment = {
      SAGEMAKER_PROGRAM             = "inference.py"
      SAGEMAKER_SUBMIT_DIRECTORY    = "/opt/ml/model/code"
      SAGEMAKER_CONTAINER_LOG_LEVEL = "20" # logging.INFO
    }
  }
}

resource "aws_sagemaker_endpoint_configuration" "point_gbm" {
  count = var.enable_sagemaker_serverless_endpoint ? 1 : 0
  name  = "${local.name_prefix}-point-gbm"

  production_variants {
    variant_name = "AllTraffic"
    model_name   = aws_sagemaker_model.point_gbm[0].name
    serverless_config {
      max_concurrency   = var.serverless_max_concurrency
      memory_size_in_mb = var.serverless_memory_size_mb
    }
  }
}

resource "aws_sagemaker_endpoint" "point_gbm" {
  count                = var.enable_sagemaker_serverless_endpoint ? 1 : 0
  name                 = "${local.name_prefix}-point-gbm"
  endpoint_config_name = aws_sagemaker_endpoint_configuration.point_gbm[0].name
}
