"""Tests for the agent layer: tool functions + MockLLM intent routing +
the audit trail. Requires reports/ artifacts to exist (run
scripts/run_pipeline.py first) since DataStore reads them from disk --
this mirrors production, where the online agent never re-scores live."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPORTS = Path(__file__).parent.parent / "reports"


@unittest.skipUnless((REPORTS / "current_forecasts.csv").exists(),
                      "run scripts/run_pipeline.py first to generate reports/ artifacts")
class TestAgentTools(unittest.TestCase):
    def test_get_forecast_returns_horizon_length_series(self):
        from src.agent import tool_get_forecast
        result = tool_get_forecast(store=1, dept=1)
        self.assertIn("forecast", result)
        self.assertEqual(len(result["forecast"]), 8)
        self.assertIn("model_used", result)

    def test_get_forecast_unknown_series_errors_cleanly(self):
        from src.agent import tool_get_forecast
        result = tool_get_forecast(store=999, dept=999)
        self.assertIn("error", result)

    def test_get_anomalies_shape(self):
        from src.agent import tool_get_anomalies
        result = tool_get_anomalies(store=4, dept=8)
        self.assertIn("anomalies", result)
        self.assertIsInstance(result["count"], int)

    def test_explain_change_has_reasons(self):
        from src.agent import tool_explain_change
        result = tool_explain_change(store=1, dept=1)
        self.assertIn("reasons", result)
        self.assertGreaterEqual(len(result["reasons"]), 1)

    def test_top_movers_direction(self):
        from src.agent import tool_top_movers
        down = tool_top_movers(direction="down", n=3)
        self.assertEqual(len(down["movers"]), 3)
        pct_changes = [m["pct_change_yoy"] for m in down["movers"]]
        self.assertEqual(pct_changes, sorted(pct_changes))  # ascending = most negative first


@unittest.skipUnless((REPORTS / "current_forecasts.csv").exists(),
                      "run scripts/run_pipeline.py first to generate reports/ artifacts")
class TestMockLLMRouting(unittest.TestCase):
    def test_routes_why_question_to_explain_change(self):
        from src.agent import ask
        trace = ask("Why did store 2 dept 3 change?")
        self.assertEqual(trace.steps[0]["tool"], "explain_change")
        self.assertEqual(trace.steps[0]["args"], {"store": 2, "dept": 3})

    def test_routes_anomaly_question_to_get_anomalies(self):
        from src.agent import ask
        trace = ask("Are there any anomalies for store 5 department 2?")
        self.assertEqual(trace.steps[0]["tool"], "get_anomalies")
        self.assertEqual(trace.steps[0]["args"], {"store": 5, "dept": 2})

    def test_routes_mover_question_to_top_movers(self):
        from src.agent import ask
        trace = ask("What are the biggest declines this year?")
        self.assertEqual(trace.steps[0]["tool"], "top_movers")

    def test_default_routes_to_forecast(self):
        from src.agent import ask
        trace = ask("store 6 dept 4")
        self.assertEqual(trace.steps[0]["tool"], "get_forecast")

    def test_every_answer_has_a_logged_trace_step(self):
        from src.agent import ask
        trace = ask("Why did store 1 dept 1 change?")
        self.assertGreaterEqual(len(trace.steps), 1)
        self.assertTrue(trace.final_answer)


if __name__ == "__main__":
    unittest.main()
