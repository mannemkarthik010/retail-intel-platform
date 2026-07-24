#!/usr/bin/env bash
# Packages the point-GBM model artifact + inference script into the
# model.tar.gz layout AWS's scikit-learn inference container expects, then
# uploads it to S3 -- run this once after `python scripts/run_pipeline.py`
# has produced reports/models/*.joblib, and again any time the model is
# retrained, before `terraform apply` (or a `terraform taint` +
# re-apply of the SageMaker model resource) picks up a new model_data_s3_uri.
#
# Expected layout inside model.tar.gz (SKLearn container contract):
#   model.tar.gz
#     gbm_point.joblib
#     feature_medians.joblib
#     code/
#       inference.py          <- model_fn/input_fn/predict_fn/output_fn
#
# NOT run in this build sandbox (no AWS credentials/network access here --
# see infra/README.md's disclosure section); this is meant to be run by
# you, locally, with your own AWS credentials, against your own S3 bucket.
set -euo pipefail

if [[ -z "${MODEL_ARTIFACT_S3_URI:-}" ]]; then
  echo "Set MODEL_ARTIFACT_S3_URI to where model.tar.gz should be uploaded, e.g.:" >&2
  echo "  export MODEL_ARTIFACT_S3_URI=s3://retail-intel-artifacts-<suffix>/models/model.tar.gz" >&2
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPORTS_MODELS_DIR="${ROOT_DIR}/reports/models"
INFERENCE_DIR="${ROOT_DIR}/infra/sagemaker/inference"

for f in gbm_point.joblib feature_medians.joblib; do
  if [[ ! -f "${REPORTS_MODELS_DIR}/${f}" ]]; then
    echo "Missing ${REPORTS_MODELS_DIR}/${f} -- run 'python scripts/run_pipeline.py' first." >&2
    exit 1
  fi
done

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

cp "${REPORTS_MODELS_DIR}/gbm_point.joblib" "${WORK_DIR}/"
cp "${REPORTS_MODELS_DIR}/feature_medians.joblib" "${WORK_DIR}/"
mkdir -p "${WORK_DIR}/code"
cp "${INFERENCE_DIR}/inference.py" "${WORK_DIR}/code/"

TARBALL="${WORK_DIR}/model.tar.gz"
tar -czf "${TARBALL}" -C "${WORK_DIR}" gbm_point.joblib feature_medians.joblib code

echo "Uploading ${TARBALL} -> ${MODEL_ARTIFACT_S3_URI}"
aws s3 cp "${TARBALL}" "${MODEL_ARTIFACT_S3_URI}"

echo "Done. Pass this S3 URI as the model_data_s3_uri Terraform variable."
