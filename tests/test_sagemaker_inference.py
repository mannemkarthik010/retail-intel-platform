"""Tests for infra/sagemaker/inference/inference.py's model_fn/input_fn/
predict_fn/output_fn contract functions. model_fn is exercised against real
joblib files written to a temp dir (no SageMaker container needed -- this
is exactly what AWS's scikit-learn inference container does when it starts
a worker); predict_fn uses a tiny fake model object instead of the real
~240-series GBM so this test doesn't depend on reports/ existing."""
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from infra.sagemaker.inference import inference  # noqa: E402


class _FakeModel:
    """Stands in for the real HistGradientBoostingRegressor -- returns the
    row sum, just so predict_fn's plumbing is checkable deterministically."""
    def predict(self, rows):
        return np.array(rows).sum(axis=1)


class TestModelFn(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        joblib.dump(_FakeModel(), self.tmp / "gbm_point.joblib")
        joblib.dump({"lag_1": 100.0}, self.tmp / "feature_medians.joblib")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_loads_both_artifacts(self):
        model = inference.model_fn(str(self.tmp))
        self.assertIn("point_model", model)
        self.assertIn("feature_medians", model)
        self.assertEqual(model["feature_medians"], {"lag_1": 100.0})


class TestInputOutputFn(unittest.TestCase):
    def test_input_fn_parses_json_body(self):
        body = json.dumps({"feature_columns": ["lag_1", "rollmean_4"], "rows": [[10.0, 20.0], [1.0, 2.0]]})
        parsed = inference.input_fn(body, "application/json")
        self.assertEqual(parsed["feature_columns"], ["lag_1", "rollmean_4"])
        np.testing.assert_array_equal(parsed["rows"], np.array([[10.0, 20.0], [1.0, 2.0]]))

    def test_input_fn_rejects_non_json_content_type(self):
        with self.assertRaises(ValueError):
            inference.input_fn("not json", "text/plain")

    def test_output_fn_serializes_predictions_as_json(self):
        result = inference.output_fn([1.0, 2.5], "application/json")
        self.assertEqual(json.loads(result), {"predictions": [1.0, 2.5]})

    def test_output_fn_rejects_non_json_accept_type(self):
        with self.assertRaises(ValueError):
            inference.output_fn([1.0], "text/plain")


class TestPredictFn(unittest.TestCase):
    def test_predicts_using_point_model_only(self):
        model = {"point_model": _FakeModel(), "feature_medians": {}}
        input_data = {"feature_columns": ["a", "b"], "rows": np.array([[1.0, 2.0], [3.0, 4.0]])}
        predictions = inference.predict_fn(input_data, model)
        self.assertEqual(predictions, [3.0, 7.0])


if __name__ == "__main__":
    unittest.main()
