# Data provenance

**The dataset used in this repo is synthetic.** This document explains why,
exactly how it was generated, and how to replace it with the real thing.

## Why not real data

The original plan was to use the real, well-known ["Walmart Recruiting - Store
Sales Forecasting"](https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting)
Kaggle dataset (or M5 / Favorita as alternatives). That didn't happen because
this project was built inside a sandboxed cloud container whose outbound
network access is restricted to a small allowlist that doesn't include
Kaggle, the UCI ML repository, or FRED, and the available web-fetch tooling
times out on unattended approval prompts for bulk file downloads. Rather than
ship a broken or partial download, this repo generates data instead — with
the exact same schema, at the exact same statistical character, with the
generative process and every embedded "ground truth" event documented and
checked (see `scripts/run_anomaly_scan.py`'s validation section).

**This is disclosed here, in the README, and in `data/generate_data.py`'s
module docstring — nowhere does this project claim to have modeled real
Walmart sales.**

## What's in the generated dataset

- **20 stores** (types A/B/C, sized like the real dataset's store-size
  distribution), **12 departments**, **260 weeks** (Jan 2019 – Dec 2023) —
  62,400 rows.
- **Real US calendar holidays** used as flags: Super Bowl, Labor Day,
  Thanksgiving, Christmas — same four the real Walmart dataset flags.
- **Annual seasonality + per-store growth/decline trend.**
- **Five `MarkDown` promo columns** (same names as the real dataset), active
  in bursts, which lift sales while active — an *expected, explainable* spike.
- **A COVID-like demand shock, March–August 2020**, sized and dated to the
  real pandemic period — deliberately **asymmetric by department**: some
  departments (grocery/home-improvement-like) spike, others
  (apparel/electronics-like) crash, because that's what actually happened at
  real retailers in 2020, and a system that only models a single global
  shock multiplier would miss it entirely.
- **One localized, genuinely unexplained supply-disruption dip** for a single
  (store, department) pair, lasting 5 weeks, with no promo/holiday/macro cause
  — a stress test for whether the anomaly detector can find something that
  isn't seasonal, isn't COVID, and isn't a data error.
- **A small number of data-entry-error rows** (negative sales / implausible
  spikes) scattered at random — the kind of garbage every real POS export
  contains.

Every one of these embedded events is checked against what the pipeline
actually detects — see the "Validation against embedded ground-truth events"
section that `python scripts/run_anomaly_scan.py` prints, and the summary in
`docs/EVAL_REPORT.md`.

## How to swap in the real dataset

The pipeline is schema-driven, not generator-driven — `src/` doesn't know or
care whether the CSVs in `data/` came from `generate_data.py` or from Kaggle.

1. Download `train.csv`, `features.csv`, `stores.csv` from the real Walmart
   competition (or the equivalent files from M5 / Favorita).
2. Rename/reshape columns to match what's in `data/retail_sales.csv`,
   `data/store_meta.csv`, `data/macro_features.csv` right now — run
   `python data/generate_data.py` once and inspect the headers if you want an
   exact reference.
3. Drop your real CSVs in `data/` under those same three filenames.
4. Run `python scripts/run_pipeline.py` — nothing else changes.

The one thing that WILL need attention on real data: `data/ground_truth_events.json`
and the validation block in `scripts/run_anomaly_scan.py` reference this
generator's specific injected events (a specific store/dept disruption, a
specific COVID window). On real data those don't exist — the anomaly
detector will still run, you'd just validate it against whatever real
anomalies you already know are in your own sales history instead.
