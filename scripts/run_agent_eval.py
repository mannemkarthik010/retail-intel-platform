"""Agent tool-routing eval harness -- a labeled benchmark, scored automatically.

This is distinct from tests/test_agent.py's routing tests: those are a handful
of individual unittest assertions ("does this one question call this one
tool"), useful for regression-catching but not a benchmark you can quote a
number from. This script runs the FULL labeled set in data/agent_eval_set.json
(37 questions -- 32 real cases across all 6 tools plus every kind of param
extraction, 5 explicitly documented router limitations) against the
deterministic MockLLM router, scores tool-selection + argument-extraction
accuracy, and writes a real, inspectable report to
reports/agent_eval_report.json -- same "the number is what it is" spirit as
docs/EVAL_REPORT.md's forecasting/anomaly/interval numbers.

Known-limitation cases are scored like everything else (no special-casing
that inflates the headline number), but bucketed separately in the report so
a reader can tell "the router doesn't do this yet, and we know it" apart from
"something regressed." See each case's "note" field in the eval set for what
the ideal behavior would be.

If ANTHROPIC_API_KEY or OPENAI_API_KEY is set, the same question set is also
run against that real tool-calling backend and reported -- otherwise this is
recorded as skipped, honestly, rather than silently omitted (same pattern as
BedrockLLM only ever being exercised against a mocked boto3 in the test
suite -- see docs/ARCHITECTURE.md).

Usage: python scripts/run_agent_eval.py
"""
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import AgentTrace, MockLLM, get_llm_backend  # noqa: E402

EVAL_SET_PATH = ROOT / "data" / "agent_eval_set.json"
REPORT_PATH = ROOT / "reports" / "agent_eval_report.json"


def load_cases() -> list[dict]:
    return json.loads(EVAL_SET_PATH.read_text())


def score_case(backend, case: dict) -> dict:
    trace = AgentTrace(question=case["question"])
    backend.answer(case["question"], trace)
    actual_tool = trace.steps[0]["tool"] if trace.steps else None
    actual_args = trace.steps[0]["args"] if trace.steps else {}

    mismatches = []
    if actual_tool != case["expected_tool"]:
        mismatches.append(f"tool: expected {case['expected_tool']!r}, got {actual_tool!r}")
    for key, expected_val in case.get("expected_args", {}).items():
        actual_val = actual_args.get(key)
        if actual_val != expected_val:
            mismatches.append(f"arg '{key}': expected {expected_val!r}, got {actual_val!r}")

    return {
        "id": case["id"],
        "category": case["category"],
        "question": case["question"],
        "known_limitation": bool(case.get("known_limitation", False)),
        "passed": not mismatches,
        "mismatches": mismatches,
        "actual_tool": actual_tool,
        "actual_args": actual_args,
    }


def run_eval(backend) -> dict:
    cases = load_cases()
    results = [score_case(backend, c) for c in cases]

    real_cases = [r for r in results if not r["known_limitation"]]
    known_cases = [r for r in results if r["known_limitation"]]

    by_category: dict[str, dict] = {}
    for r in results:
        bucket = by_category.setdefault(r["category"], {"n": 0, "passed": 0})
        bucket["n"] += 1
        bucket["passed"] += int(r["passed"])
    for bucket in by_category.values():
        bucket["accuracy"] = round(bucket["passed"] / bucket["n"], 4)

    return {
        "backend": backend.name,
        "n_cases": len(results),
        "n_passed": sum(r["passed"] for r in results),
        "overall_accuracy": round(sum(r["passed"] for r in results) / len(results), 4),
        "real_case_accuracy": round(sum(r["passed"] for r in real_cases) / len(real_cases), 4) if real_cases else None,
        "n_known_limitations": len(known_cases),
        "known_limitations_all_documented_correctly": all(r["passed"] for r in known_cases),
        "by_category": by_category,
        "failures": [r for r in results if not r["passed"]],
        "known_limitation_cases": known_cases,
    }


def run_real_backend_eval() -> dict:
    """Runs the same benchmark against whichever real LLM backend is
    configured via env vars, if any. Skipped (honestly, not silently) when no
    API key is present -- this repo has never been run against a live LLM API
    call (see docs/ARCHITECTURE.md's disclosure section), so reporting a
    fabricated number here would be exactly the kind of thing this project's
    eval reports otherwise refuse to do."""
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("BEDROCK_MODEL_ID")):
        return {"skipped": True,
                "reason": "no ANTHROPIC_API_KEY / OPENAI_API_KEY / BEDROCK_MODEL_ID set in this "
                          "environment -- run again with one exported to also benchmark a real "
                          "tool-calling LLM instead of just the deterministic mock router"}
    backend = get_llm_backend()
    if isinstance(backend, MockLLM):
        return {"skipped": True, "reason": "get_llm_backend() resolved to MockLLM despite a key "
                                            "being present -- check LLM_BACKEND env var"}
    try:
        result = run_eval(backend)
        result["skipped"] = False
        return result
    except Exception as e:  # real network/auth errors from a live API call
        return {"skipped": True, "reason": f"real backend eval call failed: {e!r}"}


def main():
    mock_result = run_eval(MockLLM())
    real_result = run_real_backend_eval()

    report = {"mock_backend": mock_result, "real_backend": real_result}
    REPORT_PATH.parent.mkdir(exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=str))

    print(f"Agent eval (MockLLM router): {mock_result['n_passed']}/{mock_result['n_cases']} passed "
          f"({mock_result['overall_accuracy']*100:.1f}% overall, "
          f"{mock_result['real_case_accuracy']*100:.1f}% excluding known limitations)")
    for cat, stats in sorted(mock_result["by_category"].items()):
        print(f"  {cat:20s} {stats['passed']}/{stats['n']} ({stats['accuracy']*100:.0f}%)")
    if mock_result["failures"]:
        print("\nUnexpected failures (real regressions, not documented limitations):")
        for f in mock_result["failures"]:
            print(f"  [{f['id']}] {f['question']!r} -> {f['mismatches']}")
    else:
        print("\nNo unexpected failures -- every miss is a documented known_limitation case.")

    if real_result.get("skipped"):
        print(f"\nReal-LLM-backend eval skipped: {real_result['reason']}")
    else:
        print(f"\nAgent eval ({real_result['backend']} backend): {real_result['n_passed']}/"
              f"{real_result['n_cases']} passed ({real_result['overall_accuracy']*100:.1f}%)")

    print(f"\nFull report written to {REPORT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
