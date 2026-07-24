"""Tests for infra/trigger_lambda/handler.py -- the nightly EventBridge ->
Lambda -> sagemaker:CreateProcessingJob trigger. Same sys.modules-mocking
technique as src/agent.py's BedrockLLM tests (see tests/test_agent.py) so
this runs without boto3 installed and without any real AWS call."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from infra.trigger_lambda.handler import _job_name, _build_create_processing_job_kwargs  # noqa: E402


class TestJobName(unittest.TestCase):
    def test_sanitizes_and_truncates(self):
        name = _job_name("2026-07-24T03:00:00.123456+00:00")
        self.assertNotIn(":", name)
        self.assertNotIn(".", name)
        self.assertLessEqual(len(name), 63)
        self.assertTrue(name.startswith("retail-intel-pipeline-"))

    def test_respects_prefix_override(self):
        with patch.dict("os.environ", {"JOB_NAME_PREFIX": "custom-job"}, clear=True):
            name = _job_name("2026-07-24T03:00:00+00:00")
        self.assertTrue(name.startswith("custom-job-"))


class TestBuildCreateProcessingJobKwargs(unittest.TestCase):
    REQUIRED_ENV = {
        "SAGEMAKER_IMAGE_URI": "123456789012.dkr.ecr.us-east-1.amazonaws.com/retail-intel-sagemaker:latest",
        "SAGEMAKER_EXECUTION_ROLE_ARN": "arn:aws:iam::123456789012:role/retail-intel-sagemaker-exec",
        "CODE_S3_URI": "s3://retail-intel-artifacts/code/",
        "OUTPUT_S3_URI": "s3://retail-intel-artifacts/processing-output/",
    }

    def test_builds_expected_request_shape_from_env(self):
        with patch.dict("os.environ", self.REQUIRED_ENV, clear=True):
            kwargs = _build_create_processing_job_kwargs("2026-07-24T03:00:00+00:00")

        self.assertEqual(kwargs["RoleArn"], self.REQUIRED_ENV["SAGEMAKER_EXECUTION_ROLE_ARN"])
        self.assertEqual(kwargs["AppSpecification"]["ImageUri"], self.REQUIRED_ENV["SAGEMAKER_IMAGE_URI"])
        self.assertEqual(kwargs["AppSpecification"]["ContainerEntrypoint"],
                          ["python3", "/opt/ml/processing/input/code/infra/sagemaker/processing_entrypoint.py"])
        self.assertEqual(kwargs["ProcessingInputs"][0]["S3Input"]["S3Uri"], self.REQUIRED_ENV["CODE_S3_URI"])
        self.assertEqual(kwargs["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
                          self.REQUIRED_ENV["OUTPUT_S3_URI"])
        # sensible defaults when the optional tuning knobs aren't set
        self.assertEqual(kwargs["ProcessingResources"]["ClusterConfig"]["InstanceType"], "ml.m5.large")
        self.assertEqual(kwargs["ProcessingResources"]["ClusterConfig"]["InstanceCount"], 1)
        self.assertEqual(kwargs["StoppingCondition"]["MaxRuntimeInSeconds"], 3600)

    def test_optional_overrides_are_respected(self):
        env = dict(self.REQUIRED_ENV, SAGEMAKER_INSTANCE_TYPE="ml.m5.xlarge",
                   SAGEMAKER_INSTANCE_COUNT="2", SAGEMAKER_VOLUME_SIZE_GB="30",
                   SAGEMAKER_MAX_RUNTIME_SECONDS="7200")
        with patch.dict("os.environ", env, clear=True):
            kwargs = _build_create_processing_job_kwargs("2026-07-24T03:00:00+00:00")
        self.assertEqual(kwargs["ProcessingResources"]["ClusterConfig"]["InstanceType"], "ml.m5.xlarge")
        self.assertEqual(kwargs["ProcessingResources"]["ClusterConfig"]["InstanceCount"], 2)
        self.assertEqual(kwargs["ProcessingResources"]["ClusterConfig"]["VolumeSizeInGB"], 30)
        self.assertEqual(kwargs["StoppingCondition"]["MaxRuntimeInSeconds"], 7200)

    def test_missing_required_env_var_raises_keyerror(self):
        with patch.dict("os.environ", {}, clear=True):
            with self.assertRaises(KeyError):
                _build_create_processing_job_kwargs("2026-07-24T03:00:00+00:00")

    def test_passes_cloudwatch_namespace_through_to_container_environment(self):
        env = dict(self.REQUIRED_ENV, CLOUDWATCH_METRIC_NAMESPACE="RetailIntel/dev")
        with patch.dict("os.environ", env, clear=True):
            kwargs = _build_create_processing_job_kwargs("2026-07-24T03:00:00+00:00")
        self.assertEqual(kwargs["Environment"], {"CLOUDWATCH_METRIC_NAMESPACE": "RetailIntel/dev"})

    def test_omits_environment_key_entirely_when_namespace_not_set(self):
        with patch.dict("os.environ", self.REQUIRED_ENV, clear=True):
            kwargs = _build_create_processing_job_kwargs("2026-07-24T03:00:00+00:00")
        self.assertNotIn("Environment", kwargs)


class TestLambdaHandler(unittest.TestCase):
    def test_calls_create_processing_job_once_and_returns_job_name(self):
        from infra.trigger_lambda import handler as trigger_handler

        fake_boto3 = MagicMock()
        fake_client = MagicMock()
        fake_boto3.client.return_value = fake_client
        fake_client.create_processing_job.return_value = {
            "ProcessingJobArn": "arn:aws:sagemaker:us-east-1:123456789012:processing-job/fake"
        }

        env = TestBuildCreateProcessingJobKwargs.REQUIRED_ENV
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with patch.dict("os.environ", env, clear=True):
                result = trigger_handler.lambda_handler({}, None)

        fake_client.create_processing_job.assert_called_once()
        self.assertTrue(result["processingJobName"].startswith("retail-intel-pipeline-"))
        self.assertEqual(result["processingJobArn"],
                          "arn:aws:sagemaker:us-east-1:123456789012:processing-job/fake")


if __name__ == "__main__":
    unittest.main()
