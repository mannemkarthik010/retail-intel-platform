# Two ECR repos: one for the API Lambda's container image
# (infra/lambda_api/Dockerfile), one for the SageMaker Processing
# container's image (infra/sagemaker/Dockerfile). Both start empty --
# Terraform manages the repos, but building/pushing the actual images is a
# manual step (see infra/README.md) since no Docker daemon is available in
# the sandbox this whole project was built in to do that here.

resource "aws_ecr_repository" "api" {
  name                 = "${local.name_prefix}-api"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_repository" "sagemaker_processing" {
  name                 = "${local.name_prefix}-sagemaker-processing"
  image_tag_mutability = "MUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

# Keep only the most recent 10 images per repo so ECR storage costs don't
# grow unbounded across iterative deploys.
resource "aws_ecr_lifecycle_policy" "api" {
  repository = aws_ecr_repository.api.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}

resource "aws_ecr_lifecycle_policy" "sagemaker_processing" {
  repository = aws_ecr_repository.sagemaker_processing.name
  policy = jsonencode({
    rules = [{
      rulePriority = 1
      description  = "keep last 10 images"
      selection = {
        tagStatus   = "any"
        countType   = "imageCountMoreThan"
        countNumber = 10
      }
      action = { type = "expire" }
    }]
  })
}
