import { useState } from "react";
import { useApi } from "../hooks/useApi";

const num = (v, digits = 4) => (Number.isFinite(v) ? v.toFixed(digits) : "—");
const overlap = (v) =>
  Number.isFinite(v) ? `${(v * 100).toFixed(1)}%` : "Not measured";
function Study() {
  const { data, loading, error, refetch } = useApi("/research/xai-comparison");
  const [model, setModel] = useState("xgb_finance");
  const [ticker, setTicker] = useState("AAPL");
  const report = data?.available ? data.report : null;
  const rows = report?.rows.filter((r) => r.model === model) || [];
  const row = rows.find((r) => r.ticker === ticker);
  return (
    <div className="space-y-5 mt-5 border-t border-gray-800 pt-5">
      {loading ? (
        <p role="status" className="text-sm text-gray-400">
          Loading prepared explanation comparison…
        </p>
      ) : error ? (
        <div role="alert">
          <p className="text-sm text-amber-300">{error}</p>
          <button onClick={refetch} className="btn-secondary mt-3">
            Retry XAI comparison
          </button>
        </div>
      ) : !report ? (
        <p className="text-sm text-gray-400">
          No verified XAI comparison is available.
        </p>
      ) : (
        <>
          <p className="text-sm text-gray-300">
            44 instruments per model · saved predictions from{" "}
            {report.prediction_date} · raw model log-odds before calibration
          </p>
          <div className="flex flex-wrap gap-4">
            <label className="text-xs text-gray-400">
              Comparison model
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className="input block mt-2"
              >
                <option value="xgb_finance">XGBoost Finance</option>
                <option value="lstm_finance">
                  LSTM with attention · Finance
                </option>
              </select>
            </label>
            <label className="text-xs text-gray-400">
              Comparison instrument
              <select
                value={ticker}
                onChange={(e) => setTicker(e.target.value)}
                className="input block mt-2"
              >
                {rows.map((r) => (
                  <option key={r.ticker}>{r.ticker}</option>
                ))}
              </select>
            </label>
          </div>
          <div
            role="region"
            aria-label="XAI method measurements"
            tabIndex={0}
            className="overflow-x-auto"
          >
            <table className="data-table">
              <caption className="sr-only">
                Same model and inputs, synthetic cumulative removal response,
                random control and local CPU runtime
              </caption>
              <thead>
                <tr>
                  <th scope="col">Method</th>
                  <th scope="col">Removal response</th>
                  <th scope="col">Above random</th>
                  <th scope="col">Local CPU time</th>
                </tr>
              </thead>
              <tbody>
                {report.summaries
                  .filter((s) => s.model === model)
                  .map((s) => (
                    <tr key={s.method}>
                      <th
                        scope="row"
                        className="text-left py-3 px-4 text-sm font-medium"
                      >
                        {s.name}
                      </th>
                      <td>{num(s.removal_response)}</td>
                      <td>
                        {s.response_above_random >= 0 ? "+" : ""}
                        {num(s.response_above_random)}
                      </td>
                      <td>{num(s.runtime_ms, 1)} ms</td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
          <p className="text-xs leading-relaxed text-gray-400">
            Removal response is the mean absolute change in model log-odds as
            ranked features or days are replaced by a reference. Larger means
            more model response in this synthetic test. Compare within each
            model only; it is not prediction accuracy. Runtime was measured
            locally and is not website latency.
          </p>
          {row && (
            <>
              <div className="border-y border-gray-800 py-3 text-xs flex flex-wrap gap-4">
                <span>
                  {row.ticker} · {row.direction} · {row.date}
                </span>
                <span>Raw P(UP): {overlap(row.raw_probability_up)}</span>
                <span>
                  Calibrated P(UP): {overlap(row.calibrated_probability_up)}
                </span>
                {row.window_start && (
                  <span>
                    Input window: {row.window_start} to {row.window_end}
                  </span>
                )}
              </div>
              <div className="grid gap-4 md:grid-cols-2">
                {row.methods.map((m) => {
                  const top = row.units
                    .map((unit, i) => ({ unit, value: m.values[i] }))
                    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
                    .slice(0, 5);
                  const s = report.summaries.find(
                    (s) => s.model === model && s.method === m.key,
                  );
                  return (
                    <section
                      key={m.key}
                      className="min-w-0 rounded-lg border border-gray-700 p-4"
                    >
                      <h3 className="text-sm font-medium">{m.name}</h3>
                      <p className="mt-2 text-xs text-gray-400">
                        {model === "lstm_finance"
                          ? "Highest-ranked input days"
                          : "Highest-ranked features"}{" "}
                        · signed raw log-odds values
                      </p>
                      <ol className="divide-y divide-gray-800 mt-3">
                        {top.map((t) => (
                          <li
                            key={t.unit}
                            className="py-2 flex justify-between gap-3 text-xs"
                          >
                            <span className="break-words min-w-0">
                              {t.unit.replaceAll("_", " ")}
                            </span>
                            <span className="font-mono shrink-0">
                              {t.value >= 0 ? "+" : ""}
                              {num(t.value)}
                            </span>
                          </li>
                        ))}
                      </ol>
                      <dl className="mt-3 text-xs text-gray-400 space-y-2">
                        {m.completeness_error != null && (
                          <div>
                            <dt>
                              Reconstruction / integration error for this input
                            </dt>
                            <dd className="text-gray-200 mt-1">
                              {m.completeness_error.toExponential(2)} ·
                              numerical completeness, not accuracy
                            </dd>
                          </div>
                        )}
                        {m.key === "lime_historical_neighborhood" && (
                          <>
                            <div>
                              <dt>Mean neighborhood fit (weighted R²)</dt>
                              <dd className="text-gray-200">
                                {num(s.weighted_r2, 3)} · low fit limits trust
                                in the surrogate
                              </dd>
                            </div>
                            <div>
                              <dt>
                                Mean top-three overlap across random seeds
                              </dt>
                              <dd className="text-gray-200">
                                {overlap(s.seed_top3_overlap)}
                              </dd>
                            </div>
                            <div>
                              <dt>Mean original-input approximation error</dt>
                              <dd className="text-gray-200">
                                {num(s.point_error)} log-odds
                              </dd>
                            </div>
                          </>
                        )}
                        {m.reference_top3_overlap != null && (
                          <div>
                            <dt>
                              Mean top-three overlap across mean / median
                              references
                            </dt>
                            <dd className="text-gray-200">
                              {overlap(s.reference_top3_overlap)}
                            </dd>
                          </div>
                        )}
                      </dl>
                    </section>
                  );
                })}
              </div>
            </>
          )}
          <details className="text-xs text-gray-400">
            <summary className="text-gray-200">
              Methods, references and limits
            </summary>
            <ul className="mt-3 space-y-2 list-disc pl-5">
              <li>
                TreeSHAP decomposes the tree score exactly; it uses the model's
                tree reference. LIME fits a weighted ridge surrogate to real
                pre-2023 historical rows; this adapts LimeBase and avoids
                independently generated indicator columns.
              </li>
              <li>
                Integrated Gradients sums gradient contributions along a path
                from a reference to the sequence. Occlusion replaces all
                features in one day and measures the score change. Their signed
                values have different meanings and are not interchangeable.
              </li>
              <li>
                Sequence inputs follow the original saved export: ten sessions
                before the prediction date, excluding the prediction date
                itself.
              </li>
              <li>
                Removal and interpolation can break financial indicator
                relationships and sequence continuity. They are sensitivity
                checks, not feasible market scenarios or causal effects.
                Reference choices can change rankings.
              </li>
              <li>
                Three LIME seeds, 20 random removal orders, mean and median
                reference checks. No universal winner was defined. No serving
                model was retrained or changed; the 2025 candidate holdout was
                not opened.
              </li>
              <li>
                User comprehension has not yet been evaluated. Offline
                preparation keeps PyTorch, Captum and LIME out of the serving
                API.
              </li>
            </ul>
          </details>
        </>
      )}
    </div>
  );
}
export default function XaiComparison() {
  const [open, setOpen] = useState(false);
  return (
    <section className="card" aria-labelledby="xai-study-title">
      <h2 id="xai-study-title" className="text-lg font-semibold">
        Explanation method comparison
      </h2>
      <p className="mt-2 text-sm leading-relaxed text-gray-400">
        TreeSHAP versus LIME for XGBoost; Integrated Gradients versus day-window
        occlusion for the attention LSTM. A fixed exploratory study of
        explanation behavior.
      </p>
      <button
        className="btn-secondary mt-4"
        aria-expanded={open}
        aria-controls="xai-study-content"
        onClick={() => setOpen((v) => !v)}
      >
        {open ? "Hide XAI study" : "Open XAI study"}
      </button>
      <div id="xai-study-content">{open && <Study />}</div>
    </section>
  );
}
