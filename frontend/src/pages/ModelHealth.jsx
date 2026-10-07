import { useState } from "react";
import { Link } from "react-router-dom";
import { useApi } from "../hooks/useApi";

const probability = (value) => Number.isFinite(value) ? (value * 100).toFixed(1) + "%" : "—";
const percent = (value) => Number.isFinite(value) ? value.toFixed(1) + "%" : "—";
const time = (value) => value ? new Date(value).toISOString().slice(0, 19).replace("T", " ") + " UTC" : "Unknown";
const groupKey = (group) => [group.date, group.model_version, group.feature_version, group.calibration_method].join("|");
function Metric({ label, value, note }) {
  return <div className="card"><p className="text-xs text-gray-400">{label}</p><p className="text-2xl font-semibold mt-2 tabular-nums">{value}</p><p className="text-xs text-gray-400 mt-2">{note}</p></div>;
}
function Distribution({ title, data }) {
  return <section className="card" aria-label={title}>
    <h2 className="font-medium">{title}</h2>
    <p className="text-sm text-gray-300 mt-2">{probability(data.min)} to {probability(data.max)}</p>
    <p className="text-xs text-gray-400 mt-1">{data.n} eligible recorded probabilities · mean {probability(data.mean)}</p>
    <p className="text-xs text-gray-400 mt-1">{data.distinct_values} distinct values · largest tied group: {data.largest_tie_count}/{data.n}</p>
    <div className="space-y-3 mt-5">
      {data.bins.map((bin) => <div key={bin.lower} className="grid grid-cols-[5rem_1fr_2rem] gap-3 items-center text-xs">
        <span className="text-gray-400">{bin.lower * 100}–{bin.upper * 100}%</span>
        <div className="h-2 bg-gray-800 rounded-sm" aria-hidden="true"><div className="h-2 bg-indigo-400 rounded-sm" style={{ width: data.n ? bin.count / data.n * 100 + "%" : "0%" }} /></div>
        <span className="text-right tabular-nums">{bin.count}</span>
      </div>)}
      {!data.n && <p className="text-sm text-gray-400">No eligible probabilities in this group.</p>}
    </div>
  </section>;
}
const names = {
  unknown_ticker_rows: "Records outside the monitored ticker list",
  invalid_probability_rows: "Records with missing or invalid probabilities",
  direction_mismatch_rows: "Decisions that disagree with the raw 50% rule",
  timestamp_problem_rows: "Records with missing or invalid timing metadata",
  input_problem_rows: "Records with missing or non-finite required inputs",
  missing_or_nonfinite_feature_cells: "Missing or non-finite input values",
};
export default function ModelHealth() {
  const [days, setDays] = useState(30);
  const [selection, setSelection] = useState("");
  const [exportError, setExportError] = useState("");
  const { data, loading, error, refetch } = useApi("/model-health?days=" + days);
  const group = data?.groups?.find((item) => groupKey(item) === selection) || data?.groups?.[0];
  const review = group?.input_ranges.filter((item) => item.review) || [];
  const probabilityStatus = group?.probability_status;
  function download() {
    setExportError("");
    try {
      const url = URL.createObjectURL(new Blob([JSON.stringify({ exported_at: new Date().toISOString(), ...data }, null, 2)], { type: "application/json" }));
      const link = document.createElement("a"); link.href = url;
      link.download = "quantsight-model-health-" + data.days + "d-" + data.as_of.slice(0, 10) + ".json";
      link.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch { setExportError("The monitoring report could not be downloaded. Please retry."); }
  }
  return <div className="max-w-7xl mx-auto space-y-6">
    <div className="flex flex-wrap items-start justify-between gap-4">
      <div className="max-w-2xl"><h1 className="text-xl font-semibold">Model health</h1><p className="text-sm text-gray-400 mt-2">Check saved inputs and probability spread for the live finance XGBoost model. Measure prediction success in <Link className="text-indigo-300 underline" to="/track-record">Forward results</Link>.</p></div>
      <div className="flex flex-wrap items-end gap-3">
        <label className="text-xs text-gray-400">Monitoring window<select className="input block mt-1" value={days} onChange={(event) => { setDays(Number(event.target.value)); setSelection(""); setExportError(""); }}>
          {[7, 30, 90].map((value) => <option key={value} value={value}>{value} days</option>)}
        </select></label>
        <button className="btn-secondary" onClick={refetch} disabled={loading}>Refresh monitoring</button>
        <button className="btn-secondary" onClick={download} disabled={loading || !data?.available}>Export monitoring JSON</button>
      </div>
    </div>
    {loading && <p className="text-gray-400" role="status">Loading saved forecast checks…</p>}
    {!loading && error && <div className="card" role="alert"><p>Model health could not be loaded.</p><button className="btn-secondary mt-3" onClick={refetch}>Retry model health</button></div>}
    {!loading && data?.available === false && <div className="card" role="status"><p>{data.reason}</p><button className="btn-secondary mt-3" onClick={refetch}>Retry model health</button></div>}
    {exportError && <p role="alert" className="text-sm text-amber-300">{exportError}</p>}
    {!loading && !error && data?.available && <>
      <section className="card" aria-label="Collection freshness">
        <div className="flex flex-wrap justify-between gap-3"><div><h2 className="font-medium">Collection freshness</h2><p className="text-sm text-gray-400 mt-2">Latest completed NYSE session: {data.latest_completed_session}. Latest recorded date in this window: {data.latest_recorded_date || "None"}.</p></div><span className="text-sm tabular-nums">{data.latest_session_coverage}/{data.expected_tickers} latest-session forecasts saved</span></div>
        {data.latest_session_coverage < data.expected_tickers && <p className="text-sm text-amber-200 mt-3">Latest-session collection is incomplete. {data.recording_window_open ? "The recording window is still open; the scheduled collector may still run." : "The recording window has closed. Missing forecasts cannot be recorded retrospectively."}</p>}
        <p className="text-xs text-gray-500 mt-3">Read-only check · {time(data.as_of)} · {data.records} saved forecasts across {data.forecast_dates} dates</p>
      </section>
      {!group ? <div className="card"><h2 className="font-medium">No saved forecasts in this window</h2><p className="text-sm text-gray-400 mt-2">Checks become available after the collector saves forecasts. Choose a longer window to view earlier records.</p></div> : <>
        <div className="flex flex-wrap items-end justify-between gap-4">
          <div><h2 className="font-medium">Recorded snapshot</h2><p className="text-sm text-gray-400 mt-1">Each date and model implementation is checked separately.</p></div>
          <label className="text-xs text-gray-400 max-w-full">Forecast date and version<select className="input block mt-1 max-w-full" value={groupKey(group)} onChange={(event) => setSelection(event.target.value)}>
            {data.groups.map((item) => <option key={groupKey(item)} value={groupKey(item)}>{item.date} · {item.recorded} records · model {item.model_version.slice(0, 8)} · features {item.feature_version.slice(-8)}</option>)}
          </select></label>
        </div>
        <div className="grid sm:grid-cols-3 gap-4">
          <Metric label="Snapshot coverage" value={group.recorded + "/" + group.expected} note={group.complete_cohort ? "All monitored tickers are represented." : "Partial snapshot; inspect missing coverage."} />
          <Metric label="Records with input issues" value={group.reference_supported ? group.quality.input_problem_rows : "—"} note={group.reference_supported ? "Required model inputs that are missing or not finite." : "This model does not match the input reference."} />
          <Metric label="Features to review" value={group.reference_supported ? group.range_review_count : "—"} note="Historical range checks; these are not accuracy scores." />
        </div>
        <section className="card" aria-label="Monitoring assessment">
          <h2 className="font-medium">What the checks mean</h2>
          {!group.reference_supported && <p className="text-sm text-amber-200 mt-3">The recorded model differs from the reference model. Input comparisons are unavailable for this group.</p>}
          {probabilityStatus === "calibration_compression" && <p className="text-sm text-amber-200 mt-3">Calibrated probabilities are nearly identical even though the raw model varies. The calibrated values provide little information for ranking stocks in this snapshot.</p>}
          {probabilityStatus === "limited_spread" && <p className="text-sm text-amber-200 mt-3">Prediction probabilities have little spread in this snapshot. Inspect the raw and calibrated ranges before using them to rank stocks.</p>}
          {probabilityStatus === "insufficient_sample" && <p className="text-sm text-gray-400 mt-3">Fewer than {data.rules.probability_flag.minimum_forecasts} eligible forecasts are available. Probability-spread flags are withheld.</p>}
          {probabilityStatus === "no_spread_flag" && <p className="text-sm text-gray-300 mt-3">The fixed probability-spread rule was not triggered. The distributions and tied counts below show how much ranking information is available.</p>}
          {review.length > 0 && <p className="text-sm text-gray-300 mt-3">Historical range flags: {review.slice(0, 6).map((item) => item.feature).join(", ")}{review.length > 6 ? " and " + (review.length - 6) + " more" : ""}. Prices, liquidity and market conditions can change these values; review the input ranges below.</p>}
          {Object.entries(group.quality).some(([, value]) => value > 0) && <p className="text-sm text-amber-200 mt-3">Record quality checks need review. Open the record checks below for details.</p>}
          <p className="text-sm text-gray-400 mt-3">BUY and SELL follow the raw model's 50% rule. Isotonic calibration can create tied probabilities; tied values alone do not prove an error. {group.buy_count} BUY and {group.sell_count} SELL decisions have eligible probabilities.</p>
          <p className="text-xs text-gray-500 mt-3">A cross-section of stocks is not a set of independent forecast dates. Range flags describe unusual inputs, not proven drift or lost predictive skill.</p>
        </section>
        <div className="grid md:grid-cols-2 gap-4"><Distribution title="Raw probability of UP" data={group.raw_probability} /><Distribution title="Calibrated probability of UP" data={group.calibrated_probability} /></div>
        <details className="card"><summary className="cursor-pointer font-medium">Input ranges by feature</summary>
          <p className="text-sm text-gray-400 mt-3">Compare each ticker with its own historical 1st–99th percentile band. A review flag requires at least {data.rules.range_flag.minimum_finite_forecasts} finite observations and {data.rules.range_flag.minimum_outside_fraction * 100}% outside the band. Values on the boundary are inside.</p>
          <div role="region" aria-label="Input range checks" tabIndex={0} className="overflow-x-auto max-w-full mt-4"><table className="data-table"><thead><tr><th>Model input</th><th>Finite observations</th><th>Outside band</th><th>Outside (%)</th><th>Check</th></tr></thead><tbody>
            {group.input_ranges.map((item) => <tr key={item.feature}><td>{item.feature}</td><td>{item.n}</td><td>{item.outside}</td><td>{percent(item.outside_pct)}</td><td>{item.review === null ? "Small sample" : item.review ? "Review" : "No flag"}</td></tr>)}
            {!group.input_ranges.length && <tr><td colSpan={5}>No supported input reference for this model.</td></tr>}
          </tbody></table></div>
        </details>
        <details className="card"><summary className="cursor-pointer font-medium">Record checks and versions</summary>
          <dl className="grid sm:grid-cols-2 gap-x-6 gap-y-3 mt-4 text-sm">{Object.entries(group.quality).map(([key, value]) => <div key={key}><dt className="text-gray-400">{names[key]}</dt><dd className="mt-1 tabular-nums">{key.includes("input") || key.includes("feature") ? group.reference_supported ? value : "Not checked for this model" : value}</dd></div>)}</dl>
          <p className="text-xs text-gray-400 mt-4">{group.calibrated_half_disagreements} eligible decisions differ from thresholding calibrated P(UP) at 50%. That is expected when calibration crosses 50%; recorded directions are checked against the raw decision rule.</p>
          <dl className="text-xs mt-4 space-y-3 break-all"><div><dt className="text-gray-400">Model artifact</dt><dd>{group.model_version}</dd></div><div><dt className="text-gray-400">Feature implementation</dt><dd>{group.feature_version}</dd></div><div><dt className="text-gray-400">Calibration</dt><dd>{group.calibration_method}</dd></div></dl>
        </details>
      </>}
      <details className="card"><summary className="cursor-pointer font-medium">Historical reference and method</summary><div className="text-sm text-gray-400 space-y-3 mt-4">
        <p>{data.reference.description}. Dates: {data.reference.start} to {data.reference.end}; {data.reference.rows.toLocaleString()} rows, {data.reference.tickers} tickers and {data.reference.features} model inputs.</p>
        <p>The saved model has no training-date manifest. This reference describes historical development inputs; it cannot establish the exact original training distribution. Absolute dollar and volume inputs can change as prices and liquidity change.</p>
        <p>The {data.rules.probability_flag.maximum_flat_span * 100} percentage-point spread threshold and historical range thresholds are fixed review rules. Outcome labels are not used in this report. Check Forward results for measured predictive performance.</p>
        <p className="text-xs break-all">Method: {data.protocol_id}<br />Reference source: {data.reference.source_sha256}<br />Rules: {data.reference.protocol_sha256}</p>
      </div></details>
    </>}
  </div>;
}
