"""Tests for infra/sagemaker/processing_entrypoint.py's input-staging and
output-sync logic. None of this touches real SageMaker/S3 -- module-level
SM_INPUT_BASE/SM_OUTPUT_BASE/DATA_DIR/REPORTS_DIR are patched to temp
directories standing in for /opt/ml/processing/{input,output}, and main()'s
tests mock out the (slow, unrelated-to-this-module) pipeline stages so this
file stays fast and focused on the adapter logic itself."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from infra.sagemaker import processing_entrypoint as entry  # noqa: E402


class TestStageRealDataIfPresent(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.input_base = self.tmp / "input"
        self.data_dir = self.tmp / "data"
        self.input_base.mkdir()
        self.data_dir.mkdir()
        self._patchers = [
            patch.object(entry, "SM_INPUT_BASE", self.input_base),
            patch.object(entry, "DATA_DIR", self.data_dir),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_returns_false_and_copies_nothing_when_no_raw_data_channel(self):
        result = entry._stage_real_data_if_present()
        self.assertFalse(result)
        self.assertEqual(list(self.data_dir.iterdir()), [])

    def test_copies_csvs_from_raw_data_channel_into_data_dir(self):
        raw_channel = self.input_base / "raw-data"
        raw_channel.mkdir()
        (raw_channel / "retail_sales.csv").write_text("Store,Dept,Date,Weekly_Sales\n1,1,2020-01-01,100\n")
        (raw_channel / "store_meta.csv").write_text("Store,Type\n1,A\n")
        (raw_channel / "readme.txt").write_text("not a csv, should be ignored")

        result = entry._stage_real_data_if_present()

        self.assertTrue(result)
        self.assertTrue((self.data_dir / "retail_sales.csv").exists())
        self.assertTrue((self.data_dir / "store_meta.csv").exists())
        self.assertFalse((self.data_dir / "readme.txt").exists())
        self.assertIn("100", (self.data_dir / "retail_sales.csv").read_text())


class TestSyncOutputs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.output_base = self.tmp / "output"
        self.output_base.mkdir()
        self.data_dir = self.tmp / "data"
        self.reports_dir = self.tmp / "reports"
        self.data_dir.mkdir()
        self.reports_dir.mkdir()
        (self.reports_dir / "current_forecasts.csv").write_text("Store,Dept,forecast\n1,1,100\n")
        (self.reports_dir / "models").mkdir()
        (self.reports_dir / "models" / "gbm_point.joblib").write_bytes(b"fake-joblib-bytes")
        (self.data_dir / "retail_sales.csv").write_text("Store,Dept\n1,1\n")
        (self.data_dir / "ground_truth_events.json").write_text(json.dumps({"n_rows": 1}))

        self._patchers = [
            patch.object(entry, "SM_OUTPUT_BASE", self.output_base),
            patch.object(entry, "DATA_DIR", self.data_dir),
            patch.object(entry, "REPORTS_DIR", self.reports_dir),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_copies_full_reports_tree_and_data_files_into_output_base(self):
        entry._sync_outputs()

        reports_out = self.output_base / "reports"
        data_out = self.output_base / "data"
        self.assertTrue((reports_out / "current_forecasts.csv").exists())
        self.assertTrue((reports_out / "models" / "gbm_point.joblib").exists())
        self.assertTrue((data_out / "retail_sales.csv").exists())
        self.assertTrue((data_out / "ground_truth_events.json").exists())

    def test_is_idempotent_when_reports_output_already_exists(self):
        entry._sync_outputs()
        # Change the source after the first sync -- the second sync should
        # overwrite the stale copy in the output dir, not error out because
        # the destination already exists (shutil.copytree fails on that by
        # default, which is exactly the bug this test guards against).
        (self.reports_dir / "current_forecasts.csv").write_text("Store,Dept,forecast\n1,1,999\n")
        entry._sync_outputs()
        content = (self.output_base / "reports" / "current_forecasts.csv").read_text()
        self.assertIn("999", content)


class TestPublishMonitoringMetrics(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.reports_dir = self.tmp / "reports"
        self.reports_dir.mkdir()
        (self.reports_dir / "monitoring_fleet_log.csv").write_text(
            "cutoff,wape,prior_avg,fleet_drift_flag\n"
            "2023-01-22,0.0841,,False\n"
            "2023-03-19,0.0656,0.0841,False\n"
        )
        (self.reports_dir / "monitoring_series_alerts.csv").write_text(
            "series,model,latest_wape,prior_avg_wape,delta\n"
            "S13_D08,seasonal_naive,0.227,0.058,0.169\n"
            "S19_D07,global_gbm,0.191,0.041,0.150\n"
        )
        self._patcher = patch.object(entry, "REPORTS_DIR", self.reports_dir)
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_skips_publish_when_namespace_not_configured(self):
        with patch.dict("os.environ", {}, clear=True):
            entry._publish_monitoring_metrics()  # must not raise

    def test_publishes_latest_fleet_wape_and_flagged_count_when_namespace_set(self):
        fake_boto3 = MagicMock()
        fake_client = MagicMock()
        fake_boto3.client.return_value = fake_client

        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with patch.dict("os.environ", {"CLOUDWATCH_METRIC_NAMESPACE": "RetailIntel/test-env"}, clear=True):
                entry._publish_monitoring_metrics()

        fake_boto3.client.assert_called_once_with("cloudwatch", region_name="us-east-1")
        call_kwargs = fake_client.put_metric_data.call_args.kwargs
        self.assertEqual(call_kwargs["Namespace"], "RetailIntel/test-env")
        metrics = {m["MetricName"]: m["Value"] for m in call_kwargs["MetricData"]}
        self.assertAlmostEqual(metrics["FleetMeanWAPEPercent"], 6.56, places=2)  # latest cutoff's row
        self.assertEqual(metrics["SeriesFlaggedForRetraining"], 2.0)

    def test_swallows_errors_instead_of_raising(self):
        fake_boto3 = MagicMock()
        fake_boto3.client.side_effect = RuntimeError("no network access in this sandbox")
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with patch.dict("os.environ", {"CLOUDWATCH_METRIC_NAMESPACE": "RetailIntel/test-env"}, clear=True):
                entry._publish_monitoring_metrics()  # must not raise

    def test_handles_missing_fleet_log_gracefully(self):
        (self.reports_dir / "monitoring_fleet_log.csv").unlink()
        with patch.dict("os.environ", {"CLOUDWATCH_METRIC_NAMESPACE": "RetailIntel/test-env"}, clear=True):
            entry._publish_monitoring_metrics()  # must not raise


class TestEnsureWritableCodeRoot(unittest.TestCase):
    """Real Processing Jobs mount the "code" input channel read-only (see
    the function's docstring); this exercises that branch explicitly by
    patching entry.ROOT to a fixture dir and os.access to report it as
    non-writable, without needing an actual read-only mount in this
    sandbox."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.fake_readonly_root = self.tmp / "readonly_root"
        (self.fake_readonly_root / "src").mkdir(parents=True)
        (self.fake_readonly_root / "src" / "marker.py").write_text("MARKER = 1\n")
        self.workspace = self.tmp / "workspace"

        self._patchers = [
            patch.object(entry, "ROOT", self.fake_readonly_root),
        ]
        for p in self._patchers:
            p.start()
        self._orig_sys_path = list(sys.path)

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        sys.path[:] = self._orig_sys_path
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_copies_repo_to_writable_workspace_when_root_is_readonly(self):
        with patch("os.access", return_value=False), \
             patch.dict("os.environ", {"SM_PROCESSING_WORKSPACE": str(self.workspace)}, clear=False):
            result = entry._ensure_writable_code_root()

        self.assertEqual(result, self.workspace)
        self.assertTrue((self.workspace / "src" / "marker.py").exists())
        self.assertTrue((self.workspace / "reports").exists())
        self.assertEqual(str(self.workspace), sys.path[0])

    def test_returns_root_unchanged_when_already_writable(self):
        # os.access is NOT patched here -- self.fake_readonly_root is a
        # normal temp dir, genuinely writable, so this should take the
        # fast path and copy nothing.
        result = entry._ensure_writable_code_root()
        self.assertEqual(result, self.fake_readonly_root)
        self.assertFalse(self.workspace.exists())


class TestMainOrchestration(unittest.TestCase):
    """main() itself just sequences generate()/run_pipeline_main()/
    run_monitoring_sim_main()/_sync_outputs() -- these are mocked out here
    since exercising the real pipeline is what tests/test_forecasting.py
    etc. already do; this only checks main() calls the right things in the
    right order given different starting states."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.data_dir = self.tmp / "data"
        self.output_base = self.tmp / "output"  # deliberately not created -> "local dry run"
        self.data_dir.mkdir()
        self._patchers = [
            patch.object(entry, "DATA_DIR", self.data_dir),
            patch.object(entry, "SM_OUTPUT_BASE", self.output_base),
            patch.object(entry, "_stage_real_data_if_present", return_value=False),
        ]
        for p in self._patchers:
            p.start()

    def tearDown(self):
        for p in self._patchers:
            p.stop()
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run_main_with_mocked_pipeline(self, generate_mock, pipeline_mock, monitoring_mock):
        # Patches entry's own module-level names (rather than swapping
        # sys.modules entries for data.generate_data/scripts.run_pipeline/
        # scripts.run_monitoring_sim) -- those three are real modules that
        # transitively import pandas/numpy, and replacing-then-restoring
        # them mid-process via patch.dict(sys.modules, ...) corrupts
        # numpy's C-extension state for every import afterward in this
        # process. patch.object on plain function references has none of
        # that risk.
        with patch.object(entry, "generate", generate_mock), \
             patch.object(entry, "run_pipeline_main", pipeline_mock), \
             patch.object(entry, "run_monitoring_sim_main", monitoring_mock), \
             patch.object(entry, "_publish_monitoring_metrics", MagicMock()):
            entry.main()

    def test_generates_data_when_no_dataset_present(self):
        generate_mock, pipeline_mock, monitoring_mock = MagicMock(), MagicMock(), MagicMock()
        self._run_main_with_mocked_pipeline(generate_mock, pipeline_mock, monitoring_mock)
        generate_mock.assert_called_once()
        pipeline_mock.assert_called_once()
        monitoring_mock.assert_called_once()

    def test_skips_generation_when_dataset_already_present(self):
        (self.data_dir / "retail_sales.csv").write_text("Store,Dept\n1,1\n")
        generate_mock, pipeline_mock, monitoring_mock = MagicMock(), MagicMock(), MagicMock()
        self._run_main_with_mocked_pipeline(generate_mock, pipeline_mock, monitoring_mock)
        generate_mock.assert_not_called()
        pipeline_mock.assert_called_once()
        monitoring_mock.assert_called_once()

    def test_does_not_attempt_output_sync_when_no_sagemaker_output_dir(self):
        # self.output_base was never created -> main() should not try to
        # mkdir or write under it (that's the "local dry run" branch).
        generate_mock, pipeline_mock, monitoring_mock = MagicMock(), MagicMock(), MagicMock()
        self._run_main_with_mocked_pipeline(generate_mock, pipeline_mock, monitoring_mock)
        self.assertFalse(self.output_base.exists())


if __name__ == "__main__":
    unittest.main()
