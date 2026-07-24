# Three execution roles, one per component that needs its own trust
# relationship: the API Lambda, the trigger Lambda, and SageMaker itself
# (shared by the Processing Job and, if enabled, the Serverless Inference
# endpoint -- both are "SageMaker acting on your behalf," so one role
# covers both rather than duplicating an near-identical policy twice).

# --------------------------------------------------------------------------
# API Lambda (infra/lambda_api/handler.py)
# --------------------------------------------------------------------------
data "aws_iam_policy_document" "lambda_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["lambda.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "lambda_api_exec" {
  name               = "${local.name_prefix}-lambda-api-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "lambda_api_basic_logs" {
  role       = aws_iam_role.lambda_api_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "lambda_api_inline" {
  # Read the two optional LLM API key secrets (see
  # infra/lambda_api/handler.py::_hydrate_secret_env_var and
  # secretsmanager.tf).
  statement {
    effect = "Allow"
    actions = [
      "secretsmanager:GetSecretValue",
    ]
    resources = [
      aws_secretsmanager_secret.anthropic_api_key.arn,
      aws_secretsmanager_secret.openai_api_key.arn,
    ]
  }

  # Bedrock Converse API calls (src/agent.py::BedrockLLM) -- only relevant
  # when bedrock_model_id is set, but harmless to always grant (Bedrock
  # model invocation is opt-in per-request in the app code, not something
  # this permission alone triggers).
  statement {
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:Converse",
      "bedrock:ConverseStream",
    ]
    resources = ["arn:aws:bedrock:*::foundation-model/*"]
  }

  # Only relevant if the Serverless Inference endpoint is enabled -- the
  # agent tools don't call it yet (see infra/README.md's disclosure
  # section), but the permission is granted now since that's the natural
  # next integration step and least-privilege still allows scoping it to
  # exactly this one endpoint.
  dynamic "statement" {
    for_each = var.enable_sagemaker_serverless_endpoint ? [1] : []
    content {
      effect    = "Allow"
      actions   = ["sagemaker:InvokeEndpoint"]
      resources = [aws_sagemaker_endpoint.point_gbm[0].arn]
    }
  }
}

resource "aws_iam_role_policy" "lambda_api_inline" {
  name   = "${local.name_prefix}-lambda-api-inline"
  role   = aws_iam_role.lambda_api_exec.id
  policy = data.aws_iam_policy_document.lambda_api_inline.json
}

# --------------------------------------------------------------------------
# Trigger Lambda (infra/trigger_lambda/handler.py)
# --------------------------------------------------------------------------
resource "aws_iam_role" "trigger_lambda_exec" {
  name               = "${local.name_prefix}-trigger-lambda-exec"
  assume_role_policy = data.aws_iam_policy_document.lambda_assume_role.json
}

resource "aws_iam_role_policy_attachment" "trigger_lambda_basic_logs" {
  role       = aws_iam_role.trigger_lambda_exec.name
  policy_arn = "arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole"
}

data "aws_iam_policy_document" "trigger_lambda_inline" {
  statement {
    effect = "Allow"
    actions = [
      "sagemaker:CreateProcessingJob",
      "sagemaker:DescribeProcessingJob",
    ]
    # Processing Job ARNs are only known after creation (the job name is
    # timestamp-suffixed at invocation time -- see
    # infra/trigger_lambda/handler.py::_job_name), so this can't be scoped
    # tighter than "any processing job in this account/region" without
    # fighting Terraform's static-ARN model for a dynamically-named resource.
    resources = ["*"]
  }

  # Required so this Lambda can hand the SageMaker execution role to the
  # Processing Job it creates -- IAM requires whoever calls
  # CreateProcessingJob to also be allowed to pass the role the job runs as.
  statement {
    effect    = "Allow"
    actions   = ["iam:PassRole"]
    resources = [aws_iam_role.sagemaker_exec.arn]
    condition {
      test     = "StringEquals"
      variable = "iam:PassedToService"
      values   = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role_policy" "trigger_lambda_inline" {
  name   = "${local.name_prefix}-trigger-lambda-inline"
  role   = aws_iam_role.trigger_lambda_exec.id
  policy = data.aws_iam_policy_document.trigger_lambda_inline.json
}

# --------------------------------------------------------------------------
# SageMaker (Processing Job + optional Serverless Inference endpoint)
# --------------------------------------------------------------------------
data "aws_iam_policy_document" "sagemaker_assume_role" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]
    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker_exec" {
  name               = "${local.name_prefix}-sagemaker-exec"
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume_role.json
}

data "aws_iam_policy_document" "sagemaker_exec_inline" {
  statement {
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:ListBucket",
    ]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]
  }

  statement {
    effect = "Allow"
    actions = [
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
    ]
    resources = ["arn:aws:logs:*:${data.aws_caller_identity.current.account_id}:log-group:/aws/sagemaker/*"]
  }

  # Publishes the two monitoring metrics
  # infra/sagemaker/processing_entrypoint.py::_publish_monitoring_metrics
  # sends -- see cloudwatch.tf's model-drift alarm, which is what actually
  # consumes them.
  statement {
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] # PutMetricData does not support resource-level scoping
  }

  # Pulls this stack's own Processing container image, and (if the
  # Serverless endpoint is enabled) AWS's prebuilt scikit-learn inference
  # image -- ECR's authorization/layer-pull actions don't support
  # per-repository resource scoping for GetAuthorizationToken, so that one
  # action is necessarily "*"; the others are scoped to repos this
  # project's own images live in plus "*" for the AWS-managed inference
  # image (whose exact ARN varies by region/account -- see
  # sagemaker_sklearn_inference_image_uri's variable description).
  statement {
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }
  statement {
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_exec_inline" {
  name   = "${local.name_prefix}-sagemaker-exec-inline"
  role   = aws_iam_role.sagemaker_exec.id
  policy = data.aws_iam_policy_document.sagemaker_exec_inline.json
}
