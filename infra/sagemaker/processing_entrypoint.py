"""SageMaker Processing entrypoint for the nightly batch pipeline (backtest
+ anomaly scan + forward forecast/intervals + explainability + monitoring
simulation) -- see docs/ARCHITECTURE.md for what each stage does and why
it's split from the online serving layer (app/server.py, src/agent.py).

This file is a THIN, S3-aware WRAPPER, not a rewrite: it imports and calls
the exact same generate()/main()/main() functions data/generate_data.py,
scripts/run_pipeline.py, and scripts/run_monitoring_sim.py already use for
a local run, so there is exactly one implementation of "how the pipeline
runs" -- this only adds the SageMaker Processing input/output-channel
conventions around it, plus a tiny bit of idempotency logic.

SageMaker Processing Job conventions this follows:
  - Input channels (if configured) are mounted at
    /opt/ml/processing/input/<channel-name> before the container starts.
  - Output channels are just local directories
    (/opt/ml/processing/output/<channel-name>) that SageMaker uploads to
    the S3 destination in the Processing Job's ProcessingOutputConfig
    *after* the container exits -- this script only has to write files
    there, not talk to S3 itself.
  - SageMaker pre-creates the base /opt/ml/processing/{input,output}
    directories for any real Processing Job; this script uses that
    directory's existence (rather than trying to create it) to detect
    "am I actually running as a Processing Job" vs. a local dry run, so a
    local `python infra/sagemaker/processing_entrypoint.py` never touches
    "/opt/ml" on a dev machine.

Processing Jobs are transient, API-invoked resources (each run is its own
`sagemaker:CreateProcessingJob` call, not a persistent stack), so unlike
the Lambda functions and API Gateway in this deployment, this container's
*invocation* isn't something Terraform manages directly -- the trigger
Lambda in infra/terraform/lambda.tf calls CreateProcessingJob via boto3 on
the nightly EventBridge schedule. What Terraform DOES manage is the ECR
repo / IAM role / S3 buckets this container and its outputs depend on --
see infra/terraform/sagemaker.tf.

Real-data note (see the "Using real data instead of the synthetic set"
section in README.md): if a `raw-data` input channel is present (i.e. this
job was invoked with real Walmart/M5/Favorita-schema CSVs staged to S3),
those are copied into data/ BEFORE generation, so the "regenerate only if
missing" check below leaves them alone. Omit that channel to keep using
the synthetic generator, which is what every existing doc/report/test in
this repo was produced against.

NOT run against a real SageMaker Processing Job in this build sandbox --
no boto3/AWS network access here (see infra/README.md's disclosure
section). This follows the documented SageMaker Processing container
contract; that's a "written to spec" claim, not an "observed running in
SageMaker" claim. What IS verified (tests/test_sagemaker_entrypoint.py) is
the input-staging and output-sync logic in isolation, against temp
directories standing in for /opt/ml/processing/*.
"""
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Imported at module level (rather than inside main()) so tests can patch
# these three names directly via unittest.mock.patch.object(entry, "...",
# ...) instead of manipulating sys.modules -- swapping sys.modules entries
# for real, already-imported packages (pandas/numpy underneath these) is
# fragile across a whole test-module run, since numpy's C extension can't
# be safely re-initialized once removed and re-added mid-process.
from data.generate_data import generate  # noqa: E402
from scripts.run_pipeline import main as run_pipeline_main  # noqa: E402
from scripts.run_monitoring_sim import main as run_monitoring_sim_main  # noqa: E402

SM_INPUT_BASE = Path(os.environ.get("SM_PROCESSING_INPUT", "/opt/ml/processing/input"))
SM_OUTPUT_BASE = Path(os.environ.get("SM_PROCESSING_OUTPUT", "/opt/ml/processing/output"))

DATA_DIR = ROOT / "data"
REPORTS_DIR = ROOT / "reports"


def _stage_real_data_if_present() -> bool:
    """If a `raw-data` input channel was mounted (real CSVs staged to S3 by
    the Processing Job's caller), copy its contents into data/ so they're
    what generate_data()'s own "skip if already present" guard sees,
    instead of letting it silently overwrite real data with synthetic
    data. Returns True if anything was staged."""
    raw_channel = SM_INPUT_BASE / "raw-data"
    if not raw_channel.exists():
        return False
    copied = 0
    for f in sorted(raw_channel.glob("*.csv")):
        shutil.copy(f, DATA_DIR / f.name)
        copied += 1
    if copied:
        print(f"Staged {copied} real data file(s) from input channel '{raw_channel}' into data/")
    return copied > 0


def _sync_outputs() -> None:
    """Copy reports/ (every CSV/JSON/joblib artifact scripts/run_pipeline.py
    and scripts/run_monitoring_sim.py produced) and the data/ CSVs+JSON into
    the SageMaker output directory, so the Processing Job's
    ProcessingOutputConfig uploads them to S3 when the container exits.
    Plain shutil -- SageMaker does the actual S3 upload; this container
    only ever writes local files."""
    reports_out = SM_OUTPUT_BASE / "reports"
    if reports_out.exists():
        shutil.rmtree(reports_out)
    shutil.copytree(REPORTS_DIR, reports_out)

    data_out = SM_OUTPUT_BASE / "data"
    data_out.mkdir(parents=True, exist_ok=True)
    for f in list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.json")):
        shutil.copy(f, data_out / f.name)

    print(f"Synced reports/ -> {reports_out}, data CSVs/JSON -> {data_out}")


def _ensure_writable_code_root() -> Path:
    """SageMaker Processing input channels -- including the "code" channel
    this repo's tree arrives on (see infra/trigger_lambda/handler.py's
    ProcessingInputs) -- are mounted READ-ONLY. But data/generate_data.py
    and scripts/run_pipeline.py (imported below) both derive their own
    data/reports paths from their own `__file__` location, precisely so
    this repo runs identically whether invoked locally or from here -- so
    if ROOT isn't writable, copy the whole repo tree once to a scratch
    directory and import everything from that writable copy instead.

    Returns the writable root to use (ROOT unchanged if it was already
    writable, e.g. every local/test run in this repo)."""
    if os.access(ROOT, os.W_OK):
        return ROOT
    workspace = Path(os.environ.get("SM_PROCESSING_WORKSPACE", "/tmp/retail-intel-workspace"))
    if workspace.exists():
        shutil.rmtree(workspace)
    shutil.copytree(ROOT, workspace, ignore=shutil.ignore_patterns("__pycache__", ".git", "reports"))
    (workspace / "reports").mkdir(exist_ok=True)
    sys.path.insert(0, str(workspace))
    print(f"Code root {ROOT} is read-only -- copied repo to writable workspace {workspace}")
    return workspace


def _publish_monitoring_metrics() -> None:
    """Best-effort: publish the latest fleet-level WAPE and the count of
    series flagged for a retraining review (both already computed by
    scripts/run_monitoring_sim.py -- see reports/monitoring_fleet_log.csv
    and reports/monitoring_series_alerts.csv) to CloudWatch, under the
    namespace CLOUDWATCH_METRIC_NAMESPACE names. This is what
    infra/terraform/cloudwatch.tf's model-drift alarm actually watches --
    without this, that alarm would reference a metric nothing ever
    publishes.

    Wrapped broadly (missing boto3, no CloudWatch permission, no network
    access, no CLOUDWATCH_METRIC_NAMESPACE set) so a metrics-publishing
    hiccup never fails the batch pipeline run itself -- the pipeline's own
    CSV outputs (synced to S3 by _sync_outputs()) are the source of truth
    either way; CloudWatch is a secondary, best-effort mirror of two
    numbers from them, purely for alerting."""
    try:
        import pandas as pd  # imported inside the try, not at module level --
                              # this function must degrade gracefully even if
                              # something is wrong with the pandas install,
                              # consistent with the rest of this function's
                              # broad error handling.
        fleet_log_path = REPORTS_DIR / "monitoring_fleet_log.csv"
        fleet_log = pd.read_csv(fleet_log_path)
        if fleet_log.empty:
            print("monitoring_fleet_log.csv is empty -- nothing to publish to CloudWatch.")
            return
        latest = fleet_log.iloc[-1]
        fleet_wape_pct = float(latest["wape"]) * 100

        alerts_path = REPORTS_DIR / "monitoring_series_alerts.csv"
        n_flagged = len(pd.read_csv(alerts_path)) if alerts_path.exists() else 0

        namespace = os.environ.get("CLOUDWATCH_METRIC_NAMESPACE")
        if not namespace:
            print("CLOUDWATCH_METRIC_NAMESPACE not set -- skipping CloudWatch metric publish "
                  "(expected for a local dry run; Terraform sets this for the real Processing Job).")
            return

        import boto3
        client = boto3.client("cloudwatch", region_name=os.environ.get("AWS_REGION", "us-east-1"))
        client.put_metric_data(Namespace=namespace, MetricData=[
            {"MetricName": "FleetMeanWAPEPercent", "Value": fleet_wape_pct, "Unit": "Percent"},
            {"MetricName": "SeriesFlaggedForRetraining", "Value": float(n_flagged), "Unit": "Count"},
        ])
        print(f"Published FleetMeanWAPEPercent={fleet_wape_pct:.2f}, "
              f"SeriesFlaggedForRetraining={n_flagged} to CloudWatch namespace '{namespace}'")
    except Exception as e:  # noqa: BLE001 -- deliberately broad, see docstring above
        print(f"Could not publish monitoring metrics to CloudWatch: {e}")


def main():
    global DATA_DIR, REPORTS_DIR

    workspace = _ensure_writable_code_root()
    if workspace != ROOT:
        DATA_DIR = workspace / "data"
        REPORTS_DIR = workspace / "reports"

    used_real_data = _stage_real_data_if_present()

    if not (DATA_DIR / "retail_sales.csv").exists():
        if used_real_data:
            print("Staged real data doesn't include retail_sales.csv -- falling back to the "
                  "synthetic generator for any missing file(s).")
        else:
            print("No existing dataset found -- generating the synthetic dataset (see "
                  "data/generate_data.py's module docstring for schema/provenance).")
        generate()
    elif used_real_data:
        print("Using staged real data from the 'raw-data' input channel -- skipping synthetic generation.")
    else:
        print("Reusing existing data/*.csv already present in this container image/volume.")

    print("Running the offline pipeline: backtest, anomaly scan, forward forecast + intervals, explainability...")
    run_pipeline_main()

    print("Running the monitoring simulation...")
    run_monitoring_sim_main()

    _publish_monitoring_metrics()

    if SM_OUTPUT_BASE.exists():
        _sync_outputs()
    else:
        print(f"No SageMaker output directory at {SM_OUTPUT_BASE} -- assuming a local dry run "
              f"(not an actual Processing Job container), so artifacts are left in reports/ and "
              f"data/ only, not synced to a processing output channel.")


if __name__ == "__main__":
    main()
