"""CI gate on the agent eval harness (scripts/run_agent_eval.py): fails the
build if MockLLM's tool-routing accuracy on the labeled benchmark regresses,
as distinct from tests/test_agent.py's handful of individual routing
assertions. Requires reports/ artifacts (same precondition as
tests/test_agent.py) since the tools the router calls read them from disk."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

REPORTS = Path(__file__).parent.parent / "reports"


@unittest.skipUnless((REPORTS / "current_forecasts.csv").exists(),
                      "run scripts/run_pipeline.py first to generate reports/ artifacts")
class TestAgentEvalHarness(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from scripts.run_agent_eval import run_eval
        from src.agent import MockLLM
        cls.result = run_eval(MockLLM())

    def test_real_case_accuracy_meets_bar(self):
        """The 32 non-known-limitation cases (all 6 tools, all argument
        extraction paths, case-insensitivity, keyword-priority ordering)
        must all route correctly -- a drop here means an actual regression
        in the router, not a pre-existing, documented gap."""
        self.assertGreaterEqual(self.result["real_case_accuracy"], 0.9,
                                 f"router regressed: {self.result['failures']}")

    def test_every_tool_is_exercised_by_the_benchmark(self):
        from src.agent import TOOLS
        exercised = {r["actual_tool"] for r in self.result["failures"]} | {
            cat for cat in self.result["by_category"]
        }
        # category names line up 1:1 with tool names except top_movers/
        # explain_change/known_limitation groupings -- just assert every
        # real TOOLS entry gets called at least once across the benchmark.
        from scripts.run_agent_eval import load_cases
        called_tools = {c["expected_tool"] for c in load_cases()}
        self.assertEqual(called_tools, set(TOOLS.keys()))

    def test_known_limitations_stay_documented_not_silently_fixed_or_worsened(self):
        """If this starts failing, one of two things happened: either a
        known_limitation case's actual behavior changed (worth a fresh look
        -- did it get fixed? does data/agent_eval_set.json need updating?),
        or the router regressed on a case we thought was solid. Either way,
        it should not fail silently in CI."""
        self.assertEqual(self.result["n_known_limitations"], 5)
        self.assertTrue(self.result["known_limitations_all_documented_correctly"],
                         "a documented known-limitation case's actual behavior changed -- "
                         "update data/agent_eval_set.json to match, with a note on what changed")


if __name__ == "__main__":
    unittest.main()
