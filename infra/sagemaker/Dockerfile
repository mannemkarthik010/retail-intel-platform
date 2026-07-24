# Container image for the nightly SageMaker Processing Job
# (infra/sagemaker/processing_entrypoint.py).
#
# This is a custom container rather than AWS's prebuilt scikit-learn
# processing image so the exact same requirements.txt this whole repo uses
# is what runs in production -- one dependency manifest, not two drifting
# copies. The tradeoff (documented, not hidden): you own the rebuild/push
# cycle for this image instead of getting security patches for free from
# AWS's managed image. At this project's scale that's the right trade;
# a team running many different processing containers would likely prefer
# the prebuilt image plus a requirements.txt install step instead.
#
# Build/push (see infra/README.md for the full walkthrough):
#   docker build -f infra/sagemaker/Dockerfile -t <account>.dkr.ecr.<region>.amazonaws.com/retail-intel-sagemaker:latest .
#   docker push <account>.dkr.ecr.<region>.amazonaws.com/retail-intel-sagemaker:latest
#
# NOT built or pushed in this build sandbox -- no Docker daemon is running
# here (see infra/README.md's disclosure section).
FROM python:3.11-slim

WORKDIR /opt/program

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# The actual repo code is supplied at RUN time via a SageMaker
# ProcessingInput (see infra/trigger_lambda/handler.py's
# ContainerEntrypoint, which points at
# /opt/ml/processing/input/code/infra/sagemaker/processing_entrypoint.py)
# rather than baked into this image -- that way a code change doesn't
# require rebuilding/repushing this image, only re-syncing the S3 "code"
# prefix (infra/scripts/package_model.sh's sibling step, documented in
# infra/README.md). This image only needs to provide the Python
# environment/dependencies.
ENTRYPOINT ["python3"]
