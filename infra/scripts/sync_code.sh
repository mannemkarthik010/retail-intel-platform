#!/usr/bin/env bash
# Syncs this repo's code (everything the SageMaker Processing container
# needs at runtime: src/, scripts/, data/generate_data.py, infra/) to the
# S3 prefix the trigger Lambda's CreateProcessingJob call references as its
# "code" ProcessingInput (CODE_S3_URI -- see infra/trigger_lambda/handler.py
# and infra/terraform/lambda.tf). Re-run this any time the pipeline code
# changes; no image rebuild is needed for a code-only change (see
# infra/sagemaker/Dockerfile's docstring for why code is synced separately
# from the container image).
#
# NOT run in this build sandbox (no AWS credentials/network access here --
# see infra/README.md's disclosure section).
set -euo pipefail

if [[ -z "${CODE_S3_URI:-}" ]]; then
  echo "Set CODE_S3_URI to the S3 prefix Terraform's code_s3_uri output shows, e.g.:" >&2
  echo "  export CODE_S3_URI=s3://retail-intel-artifacts-<suffix>/code/" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

aws s3 sync "${ROOT_DIR}/src" "${CODE_S3_URI}src" --delete
aws s3 sync "${ROOT_DIR}/scripts" "${CODE_S3_URI}scripts" --delete
aws s3 sync "${ROOT_DIR}/data" "${CODE_S3_URI}data" --delete --exclude "*.csv" --exclude "*.json"
aws s3 sync "${ROOT_DIR}/infra" "${CODE_S3_URI}infra" --delete --exclude "terraform/*" --exclude "scripts/*"

echo "Synced code to ${CODE_S3_URI}"
