import { useState } from "react";
import { useApi } from "../hooks/useApi";
import { companyName } from "../data/companyNames";
const pct = (v) => (Number.isFinite(v) ? v.toFixed(1) + "%" : "—");
const num = (v) => (Number.isFinite(v) ? v.toFixed(4) : "—");
const points = (v) =>
  Number.isFinite(v) ? (v > 0 ? "+" : "") + v.toFixed(1) + " pp" : "—";
const utc = (v) =>
  v && Number.isFinite(new Date(v).getTime())
    ? new Date(v).toISOString().replace("T", " ").slice(0, 19)
    : "Unknown";
function Table({ label, headings, rows }) {
  return (
    <div
      role="region"
      aria-label={label}
      tabIndex={0}
      className="overflow-x-auto max-w-full"
    >
      <table className="data-table">
        <thead>
          <tr>
            {headings.map((h) => (
              <th key={h}>{h}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.length ? (
            rows.map((cells, i) => (
              <tr key={i}>
                {cells.map((cell, j) => (
                  <td key={j}>{cell}</td>
                ))}
              </tr>
            ))
          ) : (
            <tr>
              <td
                colSpan={headings.length}
                className="text-center py-6 text-gray-400"
              >
                No records returned for this window.
              </td>
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}
function Metric({ label, value, note }) {
  return (
    <div className="card">
      <p className="text-xs text-gray-400">{label}</p>
      <p className="text-2xl font-semibold mt-2 tabular-nums">{value}</p>
      <p className="text-xs text-gray-400 mt-2">{note}</p>
    </div>
  );
}
export default function TrackRecord() {
  const [days, setDays] = useState(90);
  const [tab, setTab] = useState("resolved");
  const [exportError, setExportError] = useState("");
  const { data, loading, error, refetch } = useApi(
    "/live-track-record?days=" + days,
  );
  const e = data?.evidence,
    m = e?.metrics,
    ci = e?.intervals;
  const records =
    tab === "resolved"
      ? data?.recent_resolved || data?.recent?.filter((r) => r.resolved) || []
      : data?.recent_pending || data?.recent?.filter((r) => !r.resolved) || [];
  function download() {
    setExportError("");
    try {
      const snapshot = {
        exported_at: new Date().toISOString(),
        as_of: data.as_of,
        days: data.days,
        since_date: data.since_date,
        protocol_version: data.protocol_version,
        horizon_sessions: data.horizon_sessions,
        legacy_excluded: data.legacy_excluded,
        total_logged: data.total_logged,
        total_resolved: data.total_resolved,
        total_pending: data.total_pending,
        evidence: e || null,
        legacy_metrics: {
          accuracy_pct: data.accuracy_pct,
          always_up_accuracy_pct: data.always_up_accuracy_pct,
          brier_score: data.brier_score,
        },
      };
      const url = URL.createObjectURL(
        new Blob([JSON.stringify(snapshot, null, 2)], {
          type: "application/json",
        }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download =
        "quantsight-forward-" +
        data.days +
        "d-" +
        (data.as_of?.slice(0, 10) || "snapshot") +
        ".json";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch {
      setExportError(
        "The evaluation snapshot could not be downloaded. Please retry.",
      );
    }
  }
  const gain = m?.accuracy_difference_pp;
  const comparison = Number.isFinite(gain)
    ? gain === 0
      ? "Recorded accuracy equals always predicting UP in this window."
      : "Recorded accuracy is " +
        Math.abs(gain).toFixed(1) +
        " percentage points " +
        (gain > 0 ? "above" : "below") +
        " always predicting UP in this window."
    : "A baseline difference is not available yet.";
  return (
    <div className="max-w-7xl mx-auto space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div className="max-w-2xl">
          <h1 className="text-xl font-semibold">Forward results</h1>
          <p className="text-sm text-gray-400 mt-2">
            Forecasts recorded after a completed close and before the next
            market open. Outcomes use the close five NYSE trading sessions
            later.
          </p>
        </div>
        <div className="flex flex-wrap gap-3 items-end">
          <label className="text-xs text-gray-400">
            Forecast window
            <select
              className="input block mt-1"
              value={days}
              onChange={(event) => {
                setDays(Number(event.target.value));
                setExportError("");
              }}
            >
              {[30, 90, 180, 365].map((v) => (
                <option key={v} value={v}>
                  Last {v} days
                </option>
              ))}
            </select>
          </label>
          <button
            className="btn-secondary"
            disabled={loading}
            onClick={refetch}
          >
            Refresh results
          </button>
          <button
            className="btn-secondary"
            disabled={loading || !data?.available}
            onClick={download}
          >
            Export evaluation JSON
          </button>
        </div>
      </div>
      {loading ? (
        <div role="status" className="card">
          Loading forward evaluation…
        </div>
      ) : error ? (
        <div role="alert" className="card">
          <p>The forward results could not be loaded.</p>
          <button className="btn-secondary mt-3" onClick={refetch}>
            Retry forward results
          </button>
        </div>
      ) : !data?.available ? (
        <div className="card">
          <p>{data?.reason || "Forward evaluation is unavailable."}</p>
          <button className="btn-secondary mt-3" onClick={refetch}>
            Retry forward results
          </button>
        </div>
      ) : (
        <>
          <p className="text-xs text-gray-400">
            Forecast dates from {data.since_date || "Unknown"} · checked{" "}
            {utc(data.as_of)} UTC · {data.total_logged ?? "Unknown"} logged
            predictions
          </p>
          {data.days !== days && (
            <p className="text-sm text-amber-300">
              This backend returned a {data.days ?? "unknown"}-day window
              instead of the requested {days} days. Results and exports use the
              returned window. Refresh after the backend deployment finishes.
            </p>
          )}
          {exportError && (
            <p role="alert" className="text-sm text-amber-300">
              {exportError}
            </p>
          )}
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
            <Metric
              label="Recorded model accuracy"
              value={pct(m?.accuracy_pct ?? data.accuracy_pct)}
              note={
                (data.total_resolved ?? "Unknown") + " resolved stock forecasts"
              }
            />
            <Metric
              label="Always-UP accuracy"
              value={pct(
                m?.always_up_accuracy_pct ?? data.always_up_accuracy_pct,
              )}
              note="Same resolved stocks and dates"
            />
            <Metric
              label="Resolved forecast dates"
              value={e?.resolved_forecast_dates ?? "Unknown"}
              note="Stocks from one date share market conditions"
            />
            <Metric
              label="Pending forecasts"
              value={data.total_pending ?? "Unknown"}
              note="Excluded from performance scores"
            />
          </div>
          <section
            className="card space-y-3"
            aria-label="Forward evidence assessment"
          >
            <h2 className="text-base font-semibold">What these results show</h2>
            <p className="text-sm text-gray-300">{comparison}</p>
            {e ? (
              <>
                <p className="text-sm text-gray-400">
                  {e.resolved_forecast_dates <
                  ci.minimum_resolved_forecast_dates
                    ? "Early results: " +
                      e.resolved_forecast_dates +
                      " resolved forecast date" +
                      (e.resolved_forecast_dates === 1 ? "" : "s") +
                      ". At least " +
                      ci.minimum_resolved_forecast_dates +
                      " dates are required before displaying exploratory uncertainty intervals."
                    : "Results describe the selected window. A positive difference alone does not establish a reliable advantage."}
                </p>
                {e.mixed_versions && (
                  <p className="text-sm text-amber-300">
                    These totals combine different model or feature versions.
                    Separate results are shown below; pooled uncertainty is
                    unavailable.
                  </p>
                )}
                {e.partial_forecast_dates.length > 0 && (
                  <p className="text-sm text-amber-300">
                    {e.partial_forecast_dates.length} forecast date(s) have
                    partially resolved outcomes. Scores may change as remaining
                    outcomes arrive.
                  </p>
                )}
                {e.overdue_pending > 0 && (
                  <p className="text-sm text-amber-300">
                    {e.overdue_pending} pending forecast(s) have passed their
                    target close plus the 20-minute data delay. Check outcome
                    collection; these remain excluded from scores.
                  </p>
                )}
                {e.collection?.missing_sessions.length > 0 && (
                  <details className="text-sm border-t border-gray-800 pt-3">
                    <summary className="cursor-pointer text-amber-300">
                      Forecast collection has gaps:{" "}
                      {e.collection.missing_sessions.length} unrecorded market
                      sessions
                    </summary>
                    <p className="mt-2 text-gray-400">
                      Recorded on {e.collection.recorded_sessions} of{" "}
                      {e.collection.expected_sessions} expected sessions since
                      the first log in this window, through{" "}
                      {e.collection.through_session}. Check the collection job.
                      Missing forecasts are not counted as incorrect
                      predictions.
                    </p>
                    <p className="mt-2 text-xs text-gray-400 break-words">
                      Missing dates: {e.collection.missing_sessions.join(", ")}
                    </p>
                    <p className="mt-2 text-xs text-gray-400">
                      A session is flagged after 23:00 UTC, allowing one hour
                      beyond the scheduled job. Records cannot identify the
                      failure cause.
                    </p>
                  </details>
                )}
              </>
            ) : (
              <p className="text-sm text-amber-300">
                Detailed date-level evidence is not available from this backend
                yet.
              </p>
            )}
            <p className="text-xs text-gray-400">
              Stocks share market events and five-session outcomes overlap.
              These are direction estimates, not portfolio returns or Sharpe
              ratios.
            </p>
          </section>
          <section className="card" aria-label="Probability quality">
            <h2 className="text-base font-semibold mb-3">
              Probability quality
            </h2>
            <div className="grid gap-4 sm:grid-cols-3">
              <div>
                <p className="text-xs text-gray-400">Model Brier loss</p>
                <p className="mt-1 font-mono text-lg">
                  {num(m?.brier_score ?? data.brier_score)}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-400">
                  Fixed 50% forecast Brier loss
                </p>
                <p className="mt-1 font-mono text-lg">
                  {num(m?.flat_probability_brier)}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-400">
                  Balanced direction accuracy
                </p>
                <p className="mt-1 font-mono text-lg">
                  {pct(m?.balanced_accuracy_pct)}
                </p>
              </div>
            </div>
            <p className="text-xs text-gray-400 mt-3">
              Lower Brier loss is better. It scores recorded calibrated P(UP),
              while direction accuracy scores the model’s recorded raw-score
              decision. Balanced accuracy gives UP and DOWN equal weight and
              needs both outcomes.
            </p>
            {e && (
              <details className="mt-4 border-t border-gray-800 pt-3">
                <summary className="cursor-pointer text-sm text-gray-300">
                  Observed probability groups
                </summary>
                <p className="text-xs text-gray-400 my-3">
                  Compare average P(UP) with the actual UP rate. Small groups
                  are descriptive; they do not validate calibration.
                </p>
                <Table
                  label="Observed probability groups"
                  headings={[
                    "P(UP) range",
                    "Resolved rows",
                    "Average P(UP)",
                    "Actual UP rate",
                  ]}
                  rows={e.probability_bins.map((b, i) => [
                    (b.lower * 100).toFixed(0) +
                      "%–" +
                      (b.upper * 100).toFixed(0) +
                      (i === 4 ? "% inclusive" : "% upper excluded"),
                    b.rows,
                    pct(
                      b.mean_probability_up == null
                        ? null
                        : b.mean_probability_up * 100,
                    ),
                    pct(
                      b.actual_up_fraction == null
                        ? null
                        : b.actual_up_fraction * 100,
                    ),
                  ])}
                />
              </details>
            )}
          </section>
          {e && (
            <>
              <details className="card">
                <summary className="cursor-pointer font-medium">
                  Results by forecast date
                </summary>
                <p className="text-xs text-gray-400 my-3">
                  A cohort contains forecasts for the same market session.
                  Accuracy uses resolved outcomes only; partial cohorts can
                  change. Logged counts do not guarantee full instrument
                  coverage.
                </p>
                <Table
                  label="Forecast date results"
                  headings={[
                    "Forecast date",
                    "Logged",
                    "Resolved",
                    "Pending",
                    "Model accuracy",
                    "Always UP",
                    "Difference",
                    "Status",
                  ]}
                  rows={[...e.cohorts]
                    .reverse()
                    .map((c) => [
                      c.date,
                      c.logged,
                      c.resolved,
                      c.pending,
                      pct(c.metrics.accuracy_pct),
                      pct(c.metrics.always_up_accuracy_pct),
                      points(c.metrics.accuracy_difference_pp),
                      c.status,
                    ])}
                />
              </details>
              <details className="card">
                <summary className="cursor-pointer font-medium">
                  Uncertainty and evaluation method
                </summary>
                <div className="mt-3 space-y-3 text-sm text-gray-400">
                  {ci.available ? (
                    <Table
                      label="Exploratory uncertainty intervals"
                      headings={[
                        "Metric",
                        "Observed",
                        "Exploratory 95% interval",
                      ]}
                      rows={[
                        ["Direction accuracy", "accuracy_pct", pct],
                        [
                          "Accuracy minus always UP",
                          "accuracy_difference_pp",
                          points,
                        ],
                        ["Brier loss", "brier_score", num],
                      ].map(([label, key, format]) => [
                        label,
                        format(m[key]),
                        format(ci.bounds[key][0]) +
                          " to " +
                          format(ci.bounds[key][1]),
                      ])}
                    />
                  ) : (
                    <>
                      <p className="text-gray-300">
                        Uncertainty intervals are not available.
                      </p>
                      <ul className="list-disc pl-5 space-y-1">
                        {ci.reasons.map((reason) => (
                          <li key={reason}>{reason}</li>
                        ))}
                      </ul>
                    </>
                  )}
                  <p>
                    When eligible, we resample {ci.block_sessions}-session NYSE
                    blocks, retaining all stocks on each date together. Empty
                    sessions stay on the calendar grid. The calculation uses{" "}
                    {ci.repetitions.toLocaleString()} fixed-seed draws.
                  </p>
                  <p>
                    The minimum date count is a display guard. It does not
                    guarantee statistical power, a stable market regime or a
                    profitable strategy. Repeated monitoring and prior model
                    choices can affect interpretation.
                  </p>
                  <p>
                    Old horizon-unknown records and historical simulations are
                    excluded. Database administrators remain trusted. Only
                    forecasts dated within the selected window enter this
                    report.
                  </p>
                  <p className="text-xs">
                    Reporting protocol: {e.version} · forecast protocol:{" "}
                    {data.protocol_version}
                  </p>
                </div>
              </details>
              <details className="card">
                <summary className="cursor-pointer font-medium">
                  Results by model version
                </summary>
                <p className="text-xs text-gray-400 my-3">
                  A group shares one model artifact, feature definition and
                  calibration method. These are descriptive results, not a
                  model-selection leaderboard.
                </p>
                <Table
                  label="Forward model version results"
                  headings={[
                    "Model / features",
                    "Calibration",
                    "Dates",
                    "Resolved rows",
                    "Model accuracy",
                    "Always UP",
                    "Brier loss",
                  ]}
                  rows={e.model_groups.map((g) => [
                    <div>
                      <div className="font-mono" title={g.model_version}>
                        {g.model_version.slice(0, 12)}
                      </div>
                      <div
                        className="font-mono text-xs text-gray-400"
                        title={g.feature_version}
                      >
                        {g.feature_version.slice(0, 22)}
                      </div>
                    </div>,
                    g.calibration_method,
                    g.forecast_dates,
                    g.metrics.rows,
                    pct(g.metrics.accuracy_pct),
                    pct(g.metrics.always_up_accuracy_pct),
                    num(g.metrics.brier_score),
                  ])}
                />
              </details>
            </>
          )}
          <section
            className="card !p-0 overflow-hidden"
            aria-label="Recent forecast records"
          >
            <div className="px-5 pt-5 pb-3 flex flex-wrap items-center justify-between gap-3">
              <h2 className="text-base font-semibold">Recent forecasts</h2>
              <div
                className="flex gap-2"
                role="group"
                aria-label="Forecast record status"
              >
                {["resolved", "pending"].map((v) => (
                  <button
                    key={v}
                    className={tab === v ? "btn-primary" : "btn-secondary"}
                    aria-pressed={tab === v}
                    onClick={() => setTab(v)}
                  >
                    {v === "resolved" ? "Resolved" : "Pending"}
                  </button>
                ))}
              </div>
            </div>
            <p className="px-5 pb-3 text-xs text-gray-400">
              Up to 20 most recent {tab} forecasts. P(UP) is a probability
              estimate, not observed accuracy.
            </p>
            <Table
              label={
                tab === "resolved"
                  ? "Resolved forecast records"
                  : "Pending forecast records"
              }
              headings={[
                "Instrument",
                "Forecast session",
                "Target session",
                "Recorded direction",
                "P(UP)",
                "Actual direction",
                "Status",
              ]}
              rows={records.map((r) => [
                <div>
                  <span className="font-mono">{r.ticker}</span>
                  <div className="text-xs text-gray-400">
                    {companyName(r.ticker)}
                  </div>
                </div>,
                <div>
                  {r.predicted_date}
                  <div className="text-xs text-gray-400">
                    Recorded {utc(r.created_at)} UTC
                  </div>
                </div>,
                r.target_date || "Unknown",
                r.predicted_signal === "BUY"
                  ? "UP"
                  : r.predicted_signal === "SELL"
                    ? "DOWN"
                    : "Unknown",
                Number.isFinite(r.probability_up)
                  ? pct(r.probability_up * 100)
                  : "—",
                r.resolved
                  ? r.actual_signal === "BUY"
                    ? "UP"
                    : r.actual_signal === "SELL"
                      ? "DOWN / tie"
                      : "Unknown"
                  : "—",
                r.resolved ? (
                  <span
                    className={r.correct ? "text-emerald-300" : "text-rose-300"}
                  >
                    {r.correct ? "Correct" : "Incorrect"}
                  </span>
                ) : (
                  "Pending"
                ),
              ])}
            />
          </section>
        </>
      )}
    </div>
  );
}
