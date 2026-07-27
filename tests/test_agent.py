"""Tests for the agent layer: tool functions + MockLLM intent routing +
the audit trail. Requires reports/ artifacts to exist (run
scripts/run_pipeline.py first) since DataStore reads them from disk --
this mirrors production, where the online agent never re-scores live."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

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

    def test_get_forecast_includes_prediction_interval(self):
        from src.agent import tool_get_forecast
        result = tool_get_forecast(store=1, dept=1)
        first = result["forecast"][0]
        self.assertIn("low", first)
        self.assertIn("high", first)
        self.assertLessEqual(first["low"], first["value"])
        self.assertGreaterEqual(first["high"], first["value"])


@unittest.skipUnless((REPORTS / "models" / "gbm_point.joblib").exists(),
                      "run scripts/run_pipeline.py first to persist reports/models/*.joblib")
class TestNewAgentTools(unittest.TestCase):
    """explain_forecast_drivers and simulate_scenario are only meaningful
    for series whose selected model is global_gbm -- both branches
    (available series, and the clean error on a non-GBM series) are
    exercised here."""

    @classmethod
    def setUpClass(cls):
        import json
        sel = json.loads((REPORTS / "selected_model.json").read_text())
        cls.gbm_series = [k for k, v in sel.items() if v == "global_gbm"]
        cls.non_gbm_series = [k for k, v in sel.items() if v != "global_gbm"]

    def test_explain_forecast_drivers_on_gbm_series(self):
        if not self.gbm_series:
            self.skipTest("no global_gbm series in this run's selected_model.json")
        from src.agent import tool_explain_forecast_drivers
        store, dept = int(self.gbm_series[0][1:3]), int(self.gbm_series[0][-2:])
        result = tool_explain_forecast_drivers(store=store, dept=dept, week_ahead=1)
        self.assertNotIn("error", result)
        self.assertIn("top_drivers", result)
        self.assertGreater(len(result["top_drivers"]), 0)
        self.assertIn("method", result)  # explicitly documents occlusion-not-SHAP -- see src/explain.py

    def test_explain_forecast_drivers_errors_cleanly_on_non_gbm_series(self):
        if not self.non_gbm_series:
            self.skipTest("no non-global_gbm series in this run's selected_model.json")
        from src.agent import tool_explain_forecast_drivers
        store, dept = int(self.non_gbm_series[0][1:3]), int(self.non_gbm_series[0][-2:])
        result = tool_explain_forecast_drivers(store=store, dept=dept)
        self.assertIn("error", result)

    def test_simulate_scenario_on_gbm_series(self):
        if not self.gbm_series:
            self.skipTest("no global_gbm series in this run's selected_model.json")
        from src.agent import tool_simulate_scenario
        store, dept = int(self.gbm_series[0][1:3]), int(self.gbm_series[0][-2:])
        result = tool_simulate_scenario(store=store, dept=dept, markdown_active=True, weeks=3)
        self.assertNotIn("error", result)
        self.assertEqual(len(result["scenario_forecast"]), 3)
        self.assertEqual(len(result["delta_vs_baseline"]), 3)

    def test_simulate_scenario_errors_cleanly_on_non_gbm_series(self):
        if not self.non_gbm_series:
            self.skipTest("no non-global_gbm series in this run's selected_model.json")
        from src.agent import tool_simulate_scenario
        store, dept = int(self.non_gbm_series[0][1:3]), int(self.non_gbm_series[0][-2:])
        result = tool_simulate_scenario(store=store, dept=dept, markdown_active=True)
        self.assertIn("error", result)


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

    def test_routes_what_if_question_to_simulate_scenario(self):
        from src.agent import ask
        trace = ask("What if store 1 dept 2 ran a markdown promotion for 3 weeks?")
        self.assertEqual(trace.steps[0]["tool"], "simulate_scenario")
        self.assertEqual(trace.steps[0]["args"]["markdown_active"], True)
        self.assertIsNone(trace.steps[0]["args"]["is_holiday"])  # not mentioned -> preserve real calendar
        self.assertEqual(trace.steps[0]["args"]["weeks"], 3)

    def test_what_if_holiday_wording_sets_explicit_override(self):
        from src.agent import ask
        trace = ask("What if store 1 dept 2 were a holiday week?")
        self.assertEqual(trace.steps[0]["tool"], "simulate_scenario")
        self.assertEqual(trace.steps[0]["args"]["is_holiday"], True)

    def test_routes_driver_question_to_explain_forecast_drivers(self):
        from src.agent import ask
        trace = ask("What features are driving the forecast for store 1 dept 2?")
        self.assertEqual(trace.steps[0]["tool"], "explain_forecast_drivers")
        self.assertEqual(trace.steps[0]["args"]["store"], 1)
        self.assertEqual(trace.steps[0]["args"]["dept"], 2)

    def test_every_answer_has_a_logged_trace_step(self):
        from src.agent import ask
        trace = ask("Why did store 1 dept 1 change?")
        self.assertGreaterEqual(len(trace.steps), 1)
        self.assertTrue(trace.final_answer)


class TestBackendSelection(unittest.TestCase):
    """No network calls here -- just the env-var dispatch logic in
    get_llm_backend(), and the OpenAI/Bedrock tool-spec shape conversions.
    Real calls to any provider are exercised manually (see README) with
    live credentials, not in the test suite. Bedrock specifically needs
    boto3 to even import, so those tests patch sys.modules with a fake
    boto3 module -- this exercises the dispatch/loop logic without boto3
    actually being installed or any real AWS call happening."""

    def test_defaults_to_mock_with_no_keys(self):
        from src.agent import get_llm_backend, MockLLM
        with patch.dict("os.environ", {}, clear=True):
            self.assertIsInstance(get_llm_backend(), MockLLM)

    def test_prefers_anthropic_when_only_that_key_present(self):
        from src.agent import get_llm_backend, AnthropicLLM
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-fake"}, clear=True):
            self.assertIsInstance(get_llm_backend(), AnthropicLLM)

    def test_uses_openai_when_only_that_key_present(self):
        from src.agent import get_llm_backend, OpenAILLM
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test-fake"}, clear=True):
            backend = get_llm_backend()
            self.assertIsInstance(backend, OpenAILLM)
            self.assertEqual(backend.model, "gpt-4o-mini")

    def test_anthropic_takes_precedence_when_both_keys_present(self):
        from src.agent import get_llm_backend, AnthropicLLM
        env = {"ANTHROPIC_API_KEY": "sk-test-fake", "OPENAI_API_KEY": "sk-test-fake"}
        with patch.dict("os.environ", env, clear=True):
            self.assertIsInstance(get_llm_backend(), AnthropicLLM)

    def test_llm_backend_env_var_forces_openai_even_with_anthropic_key_present(self):
        from src.agent import get_llm_backend, OpenAILLM
        env = {"ANTHROPIC_API_KEY": "sk-test-fake", "OPENAI_API_KEY": "sk-test-fake", "LLM_BACKEND": "openai"}
        with patch.dict("os.environ", env, clear=True):
            self.assertIsInstance(get_llm_backend(), OpenAILLM)

    def test_openai_model_override_via_env_var(self):
        from src.agent import get_llm_backend, OpenAILLM
        env = {"OPENAI_API_KEY": "sk-test-fake", "OPENAI_MODEL": "gpt-4o"}
        with patch.dict("os.environ", env, clear=True):
            backend = get_llm_backend()
            self.assertIsInstance(backend, OpenAILLM)
            self.assertEqual(backend.model, "gpt-4o")

    def test_openai_tool_specs_match_shared_tool_registry(self):
        from src.agent import _to_openai_tool_specs, TOOLS
        specs = _to_openai_tool_specs()
        names = {s["function"]["name"] for s in specs}
        self.assertEqual(names, set(TOOLS.keys()))
        for s in specs:
            self.assertEqual(s["type"], "function")
            self.assertIn("parameters", s["function"])
            self.assertEqual(s["function"]["parameters"]["type"], "object")

    def test_prefers_bedrock_when_only_that_signal_present(self):
        from src.agent import get_llm_backend, BedrockLLM
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            with patch.dict("os.environ", {"BEDROCK_MODEL_ID": "anthropic.claude-3-fake"}, clear=True):
                self.assertIsInstance(get_llm_backend(), BedrockLLM)

    def test_generic_aws_credentials_alone_do_not_trigger_bedrock(self):
        """Deliberate design choice (see get_llm_backend()'s docstring):
        AWS creds are often present for unrelated reasons (S3 access,
        deployment tooling), so their mere presence shouldn't silently
        switch the agent's brain to Bedrock -- only an explicit
        BEDROCK_MODEL_ID (or LLM_BACKEND=bedrock) should."""
        from src.agent import get_llm_backend, MockLLM
        env = {"AWS_ACCESS_KEY_ID": "fake", "AWS_SECRET_ACCESS_KEY": "fake", "AWS_REGION": "us-east-1"}
        with patch.dict("os.environ", env, clear=True):
            self.assertIsInstance(get_llm_backend(), MockLLM)

    def test_llm_backend_env_var_forces_bedrock_even_with_anthropic_key_present(self):
        from src.agent import get_llm_backend, BedrockLLM
        env = {"ANTHROPIC_API_KEY": "sk-test-fake", "BEDROCK_MODEL_ID": "anthropic.claude-3-fake",
               "LLM_BACKEND": "bedrock"}
        with patch.dict("sys.modules", {"boto3": MagicMock()}):
            with patch.dict("os.environ", env, clear=True):
                self.assertIsInstance(get_llm_backend(), BedrockLLM)

    def test_bedrock_region_defaults_and_override(self):
        from src.agent import BedrockLLM
        fake_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with patch.dict("os.environ", {"BEDROCK_MODEL_ID": "anthropic.claude-3-fake"}, clear=True):
                backend = BedrockLLM()
                self.assertEqual(backend.region, "us-east-1")
                fake_boto3.client.assert_called_once_with("bedrock-runtime", region_name="us-east-1")

            fake_boto3.reset_mock()
            with patch.dict("os.environ", {"BEDROCK_MODEL_ID": "anthropic.claude-3-fake",
                                            "AWS_REGION": "eu-west-1"}, clear=True):
                backend = BedrockLLM()
                self.assertEqual(backend.region, "eu-west-1")

    def test_bedrock_tool_config_matches_shared_tool_registry(self):
        from src.agent import _to_bedrock_tool_config, TOOLS
        config = _to_bedrock_tool_config()
        names = {t["toolSpec"]["name"] for t in config["tools"]}
        self.assertEqual(names, set(TOOLS.keys()))
        for t in config["tools"]:
            spec = t["toolSpec"]
            self.assertIn("description", spec)
            self.assertIn("json", spec["inputSchema"])
            self.assertEqual(spec["inputSchema"]["json"]["type"], "object")

    def test_bedrock_answer_loop_executes_tool_call_then_returns_final_text(self):
        """Simulates one Converse API round-trip: the model calls a tool,
        gets a result, then answers in text on the next turn. The boto3
        client itself is a MagicMock (no real AWS call), and the tool
        registry is patched so this doesn't depend on reports/ existing."""
        from src.agent import BedrockLLM, AgentTrace, TOOLS

        fake_boto3 = MagicMock()
        fake_client = MagicMock()
        fake_boto3.client.return_value = fake_client
        tool_use_turn = {"output": {"message": {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": "t1", "name": "get_forecast", "input": {"store": 1, "dept": 1}}}
        ]}}}
        final_turn = {"output": {"message": {"role": "assistant", "content": [
            {"text": "Here is the forecast."}
        ]}}}
        fake_client.converse.side_effect = [tool_use_turn, final_turn]

        fake_tool = MagicMock(return_value={"forecast": [], "model_used": "mock", "backtest_wape": None})
        with patch.dict("sys.modules", {"boto3": fake_boto3}):
            with patch.dict("os.environ", {"BEDROCK_MODEL_ID": "anthropic.claude-3-fake"}, clear=True):
                with patch.dict(TOOLS, {"get_forecast": fake_tool}):
                    backend = BedrockLLM()
                    trace = AgentTrace(question="what's the forecast for store 1 dept 1?")
                    answer = backend.answer("what's the forecast for store 1 dept 1?", trace)

        self.assertEqual(answer, "Here is the forecast.")
        self.assertEqual(len(trace.steps), 1)
        self.assertEqual(trace.steps[0]["tool"], "get_forecast")
        fake_tool.assert_called_once_with(store=1, dept=1)
        self.assertEqual(fake_client.converse.call_count, 2)


class TestRetryBackoffAndCostTracking(unittest.TestCase):
    """Real LLM backends (unlike MockLLM) make network calls that can
    transiently fail -- rate limits, momentary 5xx, dropped connections --
    and cost real money per token. This covers the retry/backoff helper and
    the cost-estimation/usage-accumulation logic in isolation (no real
    network calls), plus one integration test showing AnthropicLLM actually
    wires retries and usage into the trace end to end."""

    def setUp(self):
        # every test in this class disables the real time.sleep so retry
        # backoff doesn't actually slow the suite down
        patcher = patch("src.agent.time.sleep", return_value=None)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_retry_succeeds_on_first_try_reports_zero_retries(self):
        from src.agent import _retry_with_backoff
        result, retries = _retry_with_backoff(lambda: "ok")
        self.assertEqual(result, "ok")
        self.assertEqual(retries, 0)

    def test_retry_recovers_after_transient_failures(self):
        from src.agent import _retry_with_backoff
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("transient")
            return "ok"

        result, retries = _retry_with_backoff(flaky, retryable=lambda e: True)
        self.assertEqual(result, "ok")
        self.assertEqual(retries, 2)
        self.assertEqual(calls["n"], 3)

    def test_retry_raises_immediately_when_not_retryable(self):
        from src.agent import _retry_with_backoff
        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise ValueError("permanent")

        with self.assertRaises(ValueError):
            _retry_with_backoff(always_fails, retryable=lambda e: False)
        self.assertEqual(calls["n"], 1)  # no retry attempted at all

    def test_retry_exhausts_max_attempts_then_raises(self):
        from src.agent import _retry_with_backoff
        calls = {"n": 0}

        def always_fails():
            calls["n"] += 1
            raise ConnectionError("still down")

        with self.assertRaises(ConnectionError):
            _retry_with_backoff(always_fails, max_retries=3, retryable=lambda e: True)
        self.assertEqual(calls["n"], 3)

    def test_requests_error_classifier_flags_rate_limit_and_5xx_as_retryable(self):
        import requests
        from src.agent import _is_retryable_requests_error

        for status in (429, 500, 502, 503):
            resp = MagicMock(status_code=status)
            err = requests.exceptions.HTTPError(response=resp)
            self.assertTrue(_is_retryable_requests_error(err), f"status {status} should be retryable")

        resp = MagicMock(status_code=401)
        err = requests.exceptions.HTTPError(response=resp)
        self.assertFalse(_is_retryable_requests_error(err), "auth errors should not be retried")

        self.assertTrue(_is_retryable_requests_error(requests.exceptions.ConnectionError()))
        self.assertFalse(_is_retryable_requests_error(ValueError("unrelated")))

    def test_bedrock_error_classifier_flags_throttling_not_validation(self):
        from src.agent import _is_retryable_bedrock_error

        throttling = Exception()
        throttling.response = {"Error": {"Code": "ThrottlingException"}}
        self.assertTrue(_is_retryable_bedrock_error(throttling))

        validation = Exception()
        validation.response = {"Error": {"Code": "ValidationException"}}
        self.assertFalse(_is_retryable_bedrock_error(validation))

        self.assertFalse(_is_retryable_bedrock_error(ValueError("no response attr")))

    def test_cost_estimate_known_model(self):
        from src.agent import _estimate_cost_usd
        cost = _estimate_cost_usd("anthropic", "claude-sonnet-4-5", input_tokens=1_000_000, output_tokens=1_000_000)
        self.assertAlmostEqual(cost, 3.00 + 15.00)

    def test_cost_estimate_unknown_model_non_bedrock_returns_none(self):
        from src.agent import _estimate_cost_usd
        self.assertIsNone(_estimate_cost_usd("openai", "some-future-model", 1000, 1000))

    def test_cost_estimate_bedrock_unknown_model_id_uses_default_pricing(self):
        from src.agent import _estimate_cost_usd, BEDROCK_DEFAULT_PRICING
        cost = _estimate_cost_usd("bedrock", "anthropic.claude-3-5-sonnet-fake", 1_000_000, 1_000_000)
        self.assertAlmostEqual(cost, BEDROCK_DEFAULT_PRICING["input"] + BEDROCK_DEFAULT_PRICING["output"])

    def test_trace_add_usage_accumulates_across_multiple_turns(self):
        from src.agent import AgentTrace
        trace = AgentTrace(question="q", backend="anthropic")
        trace.add_usage(100, 50, model="claude-sonnet-4-5")
        trace.add_usage(30, 20, model="claude-sonnet-4-5")
        self.assertEqual(trace.token_usage, {"input_tokens": 130, "output_tokens": 70})
        self.assertIsNotNone(trace.estimated_cost_usd)
        self.assertGreater(trace.estimated_cost_usd, 0)

    def test_persist_writes_usage_log_only_for_real_backends(self):
        import tempfile
        from src import agent as agent_mod

        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(agent_mod, "REPORTS", Path(tmp)):
                mock_trace = agent_mod.AgentTrace(question="q", backend="mock", final_answer="a")
                mock_trace.persist()
                self.assertTrue((Path(tmp) / "agent_traces.jsonl").exists())
                self.assertFalse((Path(tmp) / "llm_usage.jsonl").exists())

                real_trace = agent_mod.AgentTrace(question="q2", backend="anthropic", final_answer="a2")
                real_trace.add_usage(10, 5, model="claude-sonnet-4-5")
                real_trace.persist()
                self.assertTrue((Path(tmp) / "llm_usage.jsonl").exists())
                logged = json.loads((Path(tmp) / "llm_usage.jsonl").read_text().strip())
                self.assertEqual(logged["input_tokens"], 10)
                self.assertEqual(logged["output_tokens"], 5)

    def test_persist_survives_read_only_filesystem(self):
        """Regression test for a real failure found deploying to Vercel:
        serverless runtimes mount the deployment bundle read-only, so
        appending to reports/agent_traces.jsonl raised OSError and turned
        every /api/ask call into a 500. Persisting is best-effort now --
        the trace still comes back in the response payload."""
        from src import agent as agent_mod

        trace = agent_mod.AgentTrace(question="q", backend="mock", final_answer="a")
        with patch("builtins.open", side_effect=OSError(30, "Read-only file system")):
            with patch.object(Path, "mkdir", return_value=None):
                trace.persist()  # must not raise

        self.assertFalse(trace.persisted)
        self.assertEqual(trace.to_dict()["final_answer"], "a")  # audit trail intact on the response path

    def test_persist_honors_agent_trace_dir_override(self):
        import tempfile
        from src import agent as agent_mod

        with tempfile.TemporaryDirectory() as tmp:
            target = Path(tmp) / "nested"
            with patch.dict("os.environ", {"AGENT_TRACE_DIR": str(target)}, clear=False):
                trace = agent_mod.AgentTrace(question="q", backend="mock", final_answer="a")
                trace.persist()
            self.assertTrue(trace.persisted)
            self.assertTrue((target / "agent_traces.jsonl").exists())

    def test_anthropic_answer_retries_transient_error_then_succeeds_and_tracks_usage(self):
        """Integration test: the first HTTP call raises a retryable 500,
        the second succeeds with a final text answer -- confirms
        AnthropicLLM.answer actually threads retries + token usage into
        the trace, not just that the standalone helpers work in isolation."""
        import requests
        from src.agent import AnthropicLLM, AgentTrace

        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test-fake"}, clear=True):
            backend = AnthropicLLM()

        ok_response = MagicMock()
        ok_response.raise_for_status.return_value = None
        ok_response.json.return_value = {
            "content": [{"type": "text", "text": "Here's your answer."}],
            "usage": {"input_tokens": 42, "output_tokens": 17},
        }

        failing_response = MagicMock()
        failing_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
            response=MagicMock(status_code=503))

        backend._requests = MagicMock()
        backend._requests.post.side_effect = [failing_response, ok_response]

        trace = AgentTrace(question="What's the forecast for store 1 dept 1?")
        answer = backend.answer(trace.question, trace)

        self.assertEqual(answer, "Here's your answer.")
        self.assertEqual(trace.retries, 1)
        self.assertEqual(trace.token_usage, {"input_tokens": 42, "output_tokens": 17})
        self.assertIsNotNone(trace.estimated_cost_usd)
        self.assertEqual(backend._requests.post.call_count, 2)


if __name__ == "__main__":
    unittest.main()
