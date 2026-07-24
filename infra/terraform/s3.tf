# One bucket, three logical prefixes -- code/ (the repo tree synced by
# infra/scripts/sync_code.sh, read by the SageMaker Processing Job as its
# "code" input), processing-output/ (reports/+data/ written back by
# infra/sagemaker/processing_entrypoint.py::_sync_outputs), and models/
# (the packaged model.tar.gz from infra/scripts/package_model.sh, read by
# the optional Serverless Inference endpoint). One bucket keeps IAM
# policies simple (one ARN pattern to grant against) since none of these
# three uses need to be isolated from each other for this project's
# threat model.
resource "aws_s3_bucket" "artifacts" {
  bucket = "${local.name_prefix}-artifacts-${random_string.suffix.result}"
}

resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

# Lifecycle: processing-output/ accumulates one snapshot per nightly run
# (reports/*.csv, data/*.csv, model artifacts) -- old snapshots are cheap
# but unbounded growth isn't free either, so trim anything older than 90
# days. code/ and models/ are deliberately excluded -- both are "current
# state," not a history you'd want auto-expired.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id
  rule {
    id     = "expire-old-processing-output"
    status = "Enabled"
    filter {
      prefix = "processing-output/"
    }
    expiration {
      days = 90
    }
  }
}
