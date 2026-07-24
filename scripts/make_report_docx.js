const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, ImageRun, Table, TableRow,
  TableCell, WidthType, ShadingType, BorderStyle, AlignmentType, PageBreak,
  Header, Footer, PageNumber, LevelFormat, convertInchesToTwip,
} = require("docx");

const ROOT = path.join(__dirname, "..");
const FIG = path.join(ROOT, "reports", "figures");

const INK = "0B0B0B", SECONDARY = "52514E", MUTED = "898781";
const BLUE = "2A78D6", ORANGE = "EB6834", AQUA = "1BAF7A", RED = "D03B3B";

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 360, after: 160 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 280, after: 120 } });
}
function p(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    spacing: { after: 160 },
  });
}
function bullet(text, opts = {}) {
  return new Paragraph({
    children: [new TextRun({ text, ...opts })],
    numbering: { reference: "bullets", level: 0 },
    spacing: { after: 80 },
  });
}
function caption(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, size: 18, color: SECONDARY })],
    spacing: { after: 240 },
    alignment: AlignmentType.CENTER,
  });
}
function img(file, widthPx) {
  const buf = fs.readFileSync(path.join(FIG, file));
  const w = widthPx || 560;
  return new Paragraph({
    children: [new ImageRun({ data: buf, transformation: { width: w, height: Math.round(w * 0.55) }, type: "png" })],
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 60 },
  });
}

function cell(text, opts = {}) {
  const { width = 2000, header = false, color } = opts;
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: header ? { type: ShadingType.CLEAR, fill: "F0EFEC" } : undefined,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      children: [new TextRun({ text: String(text), bold: header, size: 20, color: color || INK })],
    })],
  });
}

function table(headerRow, rows, widths) {
  const w = widths || headerRow.map(() => Math.floor(9000 / headerRow.length));
  return new Table({
    width: { size: 9000, type: WidthType.DXA },
    columnWidths: w,
    rows: [
      new TableRow({ children: headerRow.map((t, i) => cell(t, { width: w[i], header: true })) }),
      ...rows.map(r => new TableRow({ children: r.map((t, i) => cell(t, { width: w[i] })) })),
    ],
  });
}

const doc = new Document({
  numbering: {
    config: [{
      reference: "bullets",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: convertInchesToTwip(0.35), hanging: convertInchesToTwip(0.2) } } } }],
    }],
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1080, bottom: 1080, left: 1200, right: 1200 },
      },
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "Retail Demand Intelligence Platform — Project Report", size: 16, color: MUTED })],
        })],
      }),
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [new TextRun({ children: ["Page ", PageNumber.CURRENT], size: 16, color: MUTED })],
        })],
      }),
    },
    children: [
      // ---- Title ----
      new Paragraph({ text: "", spacing: { before: 600 } }),
      new Paragraph({
        children: [new TextRun({ text: "Retail Demand Intelligence Platform", bold: true, size: 44, color: INK })],
        spacing: { after: 160 },
      }),
      new Paragraph({
        children: [new TextRun({
          text: "Forecasting, Anomaly Detection, and an Agentic Q&A Layer over Multi-Store Retail Demand Data",
          size: 26, color: SECONDARY,
        })],
        spacing: { after: 100 },
      }),
      new Paragraph({
        children: [new TextRun({
          text: "A portfolio project built to demonstrate senior ML/AI engineering practice — rolling-origin backtesting, per-series model selection, explainable anomaly detection, tool-calling agents with an audit trail, and production monitoring.",
          size: 22, color: SECONDARY, italics: true,
        })],
        spacing: { after: 400 },
      }),
      new Paragraph({ children: [new TextRun({ text: "Karthik Mannem", size: 22, color: INK, bold: true })], spacing: { after: 40 } }),
      new Paragraph({ children: [new TextRun({ text: "July 2026", size: 20, color: MUTED })], spacing: { after: 40 } }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- Executive summary ----
      h1("Executive Summary"),
      p("This project is a small, end-to-end retail demand intelligence system: it forecasts demand for 240 store/department series, detects and explains anomalies, answers natural-language questions through a tool-calling agent with a full audit trail, and monitors its own accuracy over time. It was built to speak directly to five senior ML/AI engineering job descriptions (Merciv, Uber Freight, Sigma, Confido, Condor) that all ask for some combination of demand forecasting, agentic AI, RAG/LLM integration, and production-grade monitoring — without yet having the years of production experience those roles ask for. See docs/JOB_MAPPING.md in the repository for the exact requirement-by-requirement crosswalk."),
      p("The results below are real, not illustrative: they come directly from running the pipeline, not from a hand-picked example. The gradient-boosted model beat a seasonal-naive baseline by less than half a percentage point of WAPE and only won on 140 of 240 series — a genuinely modest, honest result that the report does not round up.", { color: SECONDARY }),
      h2("Key results at a glance"),
      table(
        ["Metric", "Result"],
        [
          ["Series forecasted", "240 (20 stores × 12 departments)"],
          ["Best model (mean WAPE)", "Global gradient-boosted model — 8.2%"],
          ["Series where seasonal-naive still won", "90 / 240 (37.5%)"],
          ["Injected supply disruption caught", "4 of 5 weeks flagged"],
          ["Data-entry errors caught", "61 of 61 (100%)"],
          ["Promo/holiday spikes correctly marked \"explained\"", "1,218 of 1,218"],
          ["Series flagged for retraining review by monitoring", "4"],
        ],
        [3200, 5800],
      ),
      new Paragraph({ text: "", spacing: { after: 200 } }),

      // ---- Data disclosure (up front, deliberately, not buried) ----
      h1("A Note on the Data"),
      p("The dataset used throughout this report is synthetic. The original intent was to use the real, well-known “Walmart Recruiting – Store Sales Forecasting” Kaggle dataset. That was not possible because this project was built inside a sandboxed cloud environment whose outbound network access is restricted and does not reach Kaggle, UCI, or FRED. Rather than ship a partial or broken download, a generator was built instead that reproduces the exact same schema (store/department/week, IsHoliday, five MarkDown promo columns, CPI/Unemployment/Fuel_Price) and the same statistical character — real US holiday weeks, department-level seasonality, a COVID-like demand shock in March–August 2020 sized to the real pandemic period, and a handful of injected data-quality errors."),
      p("This is disclosed here first, deliberately, rather than left for an appendix. Every number and chart in this report is real output from the pipeline running against this synthetic data — nothing is hand-picked or adjusted — but the input data itself is not scraped or downloaded. See docs/DATA_PROVENANCE.md in the repository for the full generation methodology and exact instructions for swapping in the real Kaggle/M5/Favorita dataset, which requires no changes to the modeling code.", { color: SECONDARY }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- Architecture ----
      h1("Architecture"),
      p("The core design decision is a split between an offline batch pipeline (data generation → rolling-origin backtest → anomaly scan → forward forecast → monitoring) and a thin online serving layer (a tool-calling agent and a Flask API) that only ever reads the artifacts the batch pipeline produced. This mirrors how forecasting and agentic systems are actually deployed at scale: online request latency cannot depend on how expensive the modeling is, and every agent answer is backed by a specific, versioned batch of scored predictions that can be audited after the fact."),
      img("architecture_diagram.png", 620),
      caption("Figure 1. Offline batch pipeline (top) vs. online serving layer (bottom)."),
      h2("Why three forecasting models, competed per series"),
      p("A single model chosen once either overfits to whichever benchmark it was first validated against, or quietly loses to a trivial baseline on some slice of data nobody examined. This system competes seasonal-naive, Holt-Winters, and a global gradient-boosted model against each other via rolling-origin backtesting and selects a winner per series — the “best-fit selection” pattern named explicitly in the Confido job description."),
      h2("Substitutions made because of the build environment"),
      p("Two libraries that would normally just be pip-installed were not available in this sandboxed build environment (outbound package installs beyond a pre-cached set were blocked): LightGBM was replaced with scikit-learn's HistGradientBoostingRegressor (same algorithm family — histogram-based gradient-boosted trees), and FastAPI was replaced with Flask (same REST semantics, no async support). Both substitutions are confined to a single file each and documented in docs/ARCHITECTURE.md; nothing about the pipeline's design depends on which specific library sits behind them."),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- Forecasting results ----
      h1("Forecasting Results"),
      p("Six rolling-origin cutoffs, spaced eight weeks apart across the last ~48 weeks of the five-year synthetic history, each forecasting eight weeks ahead for all 240 series."),
      table(
        ["Model", "Mean WAPE", "Median WAPE", "Mean MAPE", "Series won (/240)"],
        [
          ["Global gradient-boosted model", "8.21%", "7.08%", "9.30%", "140"],
          ["Seasonal-naive", "8.58%", "7.63%", "9.50%", "90"],
          ["Holt-Winters (fixed-parameter)", "11.16%", "10.16%", "12.26%", "10"],
        ],
        [3200, 1450, 1450, 1450, 1450],
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      img("wape_by_model.png", 460),
      caption("Figure 2. Mean WAPE by model across all 240 series and 6 backtest cutoffs."),
      img("series_wins.png", 460),
      caption("Figure 3. Per-series model selection outcome — the winner is chosen per series, not globally."),
      p("Reading this honestly: the gradient-boosted model wins on more series and has the best mean WAPE, but by less than half a point over seasonal-naive — and seasonal-naive still wins outright on over a third of all series. Those are overwhelmingly series with strong, low-noise annual seasonality where “same week last year” is close to the ceiling of what's predictable. A system that always deployed the fancier model everywhere would be quietly worse on those series than the one built here, which is exactly why the pipeline selects per series instead of globally."),
      img("example_forecast.png", 620),
      caption("Figure 4. Store 3 / Dept 5 — last 60 weeks of actuals, flagged anomalies, and the 8-week forecast (model: global GBM, backtest WAPE 4.4%)."),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- Anomaly detection ----
      h1("Anomaly Detection"),
      p("Anomaly detection combines a transparent classical decomposition (trailing trend + seasonal index + robust rolling z-score) with an IsolationForest cross-check, and — critically — separates statistically flagged points into “already explained” (an active promotion or a known calendar holiday) versus “genuinely unexplained, needs investigation.” A detector that flags promo-driven or holiday-driven spikes as anomalies would cry wolf constantly and nobody would trust it, which is exactly the failure mode the Merciv job description's “systems enterprise users can actually trust” language is about."),
      table(
        ["Category", "Count (of 48,960 series-weeks)"],
        [
          ["Normal", "45,028"],
          ["Unexplained anomaly (needs investigation)", "2,653"],
          ["Explained — holiday spike", "895"],
          ["Explained — promo/markdown spike", "323"],
          ["Data quality error (negative/implausible)", "61"],
          ["High-confidence (statistical + IsolationForest agree)", "473"],
        ],
        [6200, 2800],
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      h2("Validation against known ground-truth events"),
      p("The synthetic data generator embeds several events with a known answer, so detection quality can be checked rather than assumed:"),
      bullet("Localized supply disruption (one store/department, 5-week window, no promo/holiday/COVID cause): 4 of 5 weeks correctly flagged."),
      bullet("Data-entry errors (negative sales rows): 61 of 61 (100%) flagged as data_quality_error."),
      bullet("COVID crash window (Mar–May 2020): 211 of 240 series had at least one flagged week — consistent with a shock whose sign and magnitude vary by department."),
      bullet("Promo-driven spikes: 323 correctly separated into “explained,” never raised as something to investigate."),
      img("covid_regime_shift.png", 620),
      caption("Figure 5. The same 2020 shock produces opposite department-level responses — why anomaly detection runs per-series, not against one fleet-wide baseline."),
      h2("A limitation found and fixed during this build"),
      p("The first version of the anomaly detector used a naive week-of-year modular index for seasonality, which does not perfectly align with real calendar holidays (Thanksgiving, Labor Day, etc. do not repeat on an exact 52-week cycle). That version flagged a meaningful share of ordinary December holiday spikes as “unexplained anomalies.” The fix was to cross-check every statistically-flagged point against the same IsHoliday flag the forecasting model already uses. This dropped false “unexplained” flags from 3,548 to 2,653 without weakening detection of the real injected disruption or the COVID window. This is disclosed here as a real limitation that was found and fixed, not hidden — the lesson generalizes: a decomposition-based anomaly detector that only looks at its own residual will always need an explicit calendar/event cross-check, because real calendars don't repeat on clean fixed cycles.", { color: SECONDARY }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- Agent layer ----
      h1("Agentic Reasoning Layer"),
      p("On top of the forecast and anomaly artifacts sits a small tool-calling agent with four tools — get_forecast, get_anomalies, explain_change, top_movers — behind a pluggable “brain.” A deterministic MockLLM router is the default (no API key, no external calls, fully reproducible), and a real tool-calling loop against the Anthropic Messages API activates automatically the moment an API key is set in the environment, with zero changes to the tools or the trace format."),
      p("Every question, real or mock, produces a full audit trail: the question, each tool call with its arguments and result, and the final answer, persisted for later review. This directly addresses the auditability requirement named explicitly in the Condor job description (“correctness and auditability are non-negotiable”) and generalizes to any domain where trusting the AI's answer needs to mean “and here is exactly how it got there,” not just “trust me.”"),
      h2("Example"),
      p("Question: “Why did store 4 dept 8 change recently?”", { bold: true }),
      p("Agent trace: called explain_change(store=4, dept=8) → latest value $20,033 on 2023-12-24, +3.0% year-over-year, reason: “within normal trend + seasonal expectation, nothing unusual.” The same tool, pointed at the week of the injected supply disruption (2021-09-12), correctly surfaces it as an unexplained anomaly with a z-score of -7.18 — the agent's explanation is only ever as good as the underlying detection, which is exactly why the two are built and evaluated together rather than treating the LLM layer as if it manufactures its own ground truth.", { color: SECONDARY }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- Monitoring ----
      h1("Production Monitoring"),
      p("Using each series' own deployed (selected) model's accuracy at each of the six rolling-origin cutoffs as a stand-in for “a new week of ground truth arrived,” the monitoring layer checks for drift at two levels: the whole fleet, and each series against its own history."),
      img("monitoring_fleet.png", 460),
      caption("Figure 6. Fleet-wide accuracy of the deployed model per series, over time — no drift detected in this window."),
      p("The fleet-level view shows no drift — but averages hide individual series degrading. Checking each series against its own history surfaces four series a real team would queue for a retraining review, each jumping more than 10 points of WAPE versus its own prior average, even while the fleet-wide number looks healthy:"),
      table(
        ["Series", "Deployed model", "Latest WAPE", "Prior avg WAPE", "Delta"],
        [
          ["S13_D08", "seasonal_naive", "22.7%", "5.8%", "+16.9pts"],
          ["S19_D07", "global_gbm", "19.1%", "4.1%", "+15.0pts"],
          ["S10_D03", "global_gbm", "15.7%", "5.0%", "+10.7pts"],
          ["S11_D12", "holt_winters", "19.8%", "9.4%", "+10.4pts"],
        ],
        [1800, 2000, 1700, 1900, 1600],
      ),
      new Paragraph({ text: "", spacing: { after: 160 } }),
      p("This is the point of monitoring at the series level rather than only the fleet level: “the average looks fine” and “every individual series is fine” are different claims, and only checking the first is how a production system quietly lets a handful of series go stale.", { color: SECONDARY }),
      new Paragraph({ children: [new PageBreak()] }),

      // ---- Limitations & future work ----
      h1("Limitations and Future Work"),
      bullet("Fit Holt-Winters smoothing constants via maximum likelihood (statsmodels) instead of a small fixed grid search — the current fixed-parameter approach is the one substitution with a real, visible accuracy cost."),
      bullet("Forecast the macro covariates (CPI, unemployment, fuel price) instead of holding them at their last observed value for the forward-looking horizon."),
      bullet("Add explicit date-distance-to-holiday features to the anomaly detector's seasonal model, rather than relying on a same-week boolean cross-check as a second pass."),
      bullet("Replace the fixed z-score/MAD threshold with a per-series-calibrated threshold — some series are inherently noisier than others."),
      bullet("Wire the monitoring simulation to a real scheduler against live incoming weekly data instead of replaying historical backtest cutoffs."),
      bullet("Extend into the areas this project deliberately does not cover: document/unstructured-data extraction (Confido), knowledge-graph-backed retrieval (Merciv's bonus points), and an actual AWS deployment (Condor's SageMaker/Bedrock/Lambda preference) — see docs/JOB_MAPPING.md for the full scope discussion."),
      new Paragraph({ text: "", spacing: { after: 200 } }),

      h1("Repository"),
      p("Full source, tests, and documentation (README.md, docs/ARCHITECTURE.md, docs/DATA_PROVENANCE.md, docs/EVAL_REPORT.md, docs/JOB_MAPPING.md) are included alongside this report. Every figure and number in this document was generated by running the pipeline end-to-end — python data/generate_data.py, python scripts/run_pipeline.py, python scripts/run_monitoring_sim.py, python scripts/make_report_figures.py — and can be reproduced exactly by re-running those four commands."),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  const out = path.join(ROOT, "docs", "Retail_Intelligence_Platform_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("Wrote", out);
});
