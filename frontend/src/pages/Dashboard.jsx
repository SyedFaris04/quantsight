import { useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { companyName } from "../data/companyNames";

const pct = (v) =>
  Number.isFinite(v) ? `${(v * 100).toFixed(2)}%` : "Unavailable";
const score = (v) => (Number.isFinite(v) ? v.toFixed(2) : "Unavailable");
function Failure({ message, retry }) {
  return (
    <div role="alert" className="py-5 text-sm text-gray-300">
      <p>{message}</p>
      <button onClick={retry} className="btn-secondary mt-3">
        Try again
      </button>
    </div>
  );
}
function BacktestPreview() {
  const { data, loading, error, refetch, isSlow } = useApi("/backtest/summary");
  return (
    <section className="card" aria-labelledby="performance-title">
      <div className="flex flex-wrap justify-between items-center gap-3">
        <h2 id="performance-title" className="text-lg font-semibold">
          Historical performance
        </h2>
        <Link to="/backtesting" className="text-sm text-indigo-300">
          Full backtest →
        </Link>
      </div>
      {loading ? (
        <p role="status" className="mt-4 text-sm text-gray-400">
          {isSlow
            ? "Still waiting. The server may be starting or busy."
            : "Loading saved backtest…"}
        </p>
      ) : error ? (
        <Failure message={error} retry={refetch} />
      ) : !data?.available ? (
        <p className="mt-4 text-sm text-gray-400">
          {data?.message || "Backtest unavailable."}
        </p>
      ) : (
        <>
          <p className="mt-2 text-xs text-gray-400">
            {data.evaluation.start} to {data.evaluation.end} ·{" "}
            {data.evaluation.sessions} sessions · net of costs
          </p>
          <div className="overflow-x-auto mt-4">
            <table className="data-table">
              <caption className="sr-only">
                Historical ensemble and SPY results on the same dates
              </caption>
              <thead>
                <tr>
                  <th scope="col">Strategy</th>
                  <th scope="col">Net return</th>
                  <th scope="col">Sharpe</th>
                  <th scope="col">Max. drawdown</th>
                </tr>
              </thead>
              <tbody>
                {data.strategies.map((s) => (
                  <tr key={s.key}>
                    <th
                      scope="row"
                      className="px-4 py-3 text-left text-sm font-medium"
                    >
                      {s.key === "ensemble"
                        ? "Model ensemble"
                        : "SPY benchmark"}
                    </th>
                    <td className="tabular-nums">
                      {pct(s.metrics.total_return)}
                    </td>
                    <td className="tabular-nums">{score(s.metrics.sharpe)}</td>
                    <td className="tabular-nums">
                      {pct(s.metrics.max_drawdown)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-4 text-xs leading-relaxed text-gray-400">
            Sharpe measures return relative to volatility; drawdown shows the
            largest peak-to-trough loss. A higher return alone does not mean a
            better strategy.
          </p>
          <p className="mt-2 text-xs text-gray-400">
            Costs: {data.config.commission_bps} bps commission +{" "}
            {data.config.slippage_bps} bps slippage per side. Risk-free rate:{" "}
            {pct(data.config.annual_risk_free_rate)}.
          </p>
          <p className="mt-3 text-xs text-amber-200/90">{data.limitation}</p>
        </>
      )}
    </section>
  );
}
function NewsArchive() {
  const { data, loading, error, refetch } = useApi("/market-sentiment");
  return (
    <div className="mt-4 border-t border-gray-800 pt-4">
      {loading ? (
        <p role="status">Loading saved news…</p>
      ) : error ? (
        <Failure message={error} retry={refetch} />
      ) : !data?.article_count ? (
        <p>No dated headlines are available to score.</p>
      ) : (
        <>
          <p className="text-sm">
            Archive: {data.window_start || "Unknown"} to{" "}
            {data.window_end || "Unknown"}
          </p>
          <p className="mt-2 text-xs text-gray-400">
            Latest article:{" "}
            {data.age_days == null
              ? "age unavailable"
              : `${data.age_days} days old`}
            . This is not a live news feed.
          </p>
          <p className="mt-3 text-sm">
            {data.article_count.toLocaleString()} headlines · Positive{" "}
            {data.positive_pct ?? "Unknown"}% · Neutral{" "}
            {data.neutral_pct ?? "Unknown"}% · Negative{" "}
            {data.negative_pct ?? "Unknown"}%
          </p>
          <p className="mt-2 text-xs text-gray-400">
            VADER scores language sentiment. These dates are separate from the
            historical prediction snapshot.
          </p>
          <details className="mt-3 text-xs">
            <summary>Daily sentiment values</summary>
            <div className="max-h-48 overflow-auto mt-3">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Article date</th>
                    <th scope="col">Mean VADER score</th>
                  </tr>
                </thead>
                <tbody>
                  {data.trend?.map((r) => (
                    <tr key={r.date}>
                      <td>{r.date}</td>
                      <td>{r.avg_compound.toFixed(3)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      )}
    </div>
  );
}
export default function Dashboard() {
  const { data, loading, error, refetch, isSlow } = useApi("/dashboard");
  const [query, setQuery] = useState("");
  const [all, setAll] = useState(false);
  const [archive, setArchive] = useState(false);
  const rows = useMemo(
    () =>
      (data?.instruments || [])
        .filter((r) =>
          `${r.ticker} ${companyName(r.ticker)}`
            .toLowerCase()
            .includes(query.trim().toLowerCase()),
        )
        .sort((a, b) => a.ticker.localeCompare(b.ticker)),
    [data, query],
  );
  const s = data?.snapshot;
  const dates = s?.latest_signal_date
    ? s.earliest_signal_date === s.latest_signal_date
      ? s.latest_signal_date
      : `${s.earliest_signal_date} to ${s.latest_signal_date}`
    : "Date unavailable";
  return (
    <div className="mx-auto max-w-7xl space-y-6">
      <header className="page-heading">
        <div>
          <p className="eyebrow">Research workspace</p>
          <h1>Dashboard</h1>
          <p>
            Find an instrument, inspect its prediction, and check the evidence.
          </p>
        </div>
        <Link to="/market" className="btn-secondary">
          Market explorer →
        </Link>
      </header>
      <section
        className="border-y border-gray-800 py-4 flex flex-wrap gap-x-10 gap-y-3 text-sm"
        aria-labelledby="snapshot-title"
      >
        <div>
          <h2 id="snapshot-title" className="font-medium">
            Historical model snapshot
          </h2>
          <p className="mt-1 text-xs text-gray-400">
            Saved outputs · five NYSE trading sessions ahead
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-400">Signal dates</p>
          <p className="mt-1 font-mono">
            {loading ? "Loading…" : error ? "Unavailable" : dates}
          </p>
        </div>
        <div>
          <p className="text-xs text-gray-400">Coverage</p>
          <p className="mt-1">
            {loading
              ? "Loading…"
              : error || !data?.kpis
                ? "Unavailable"
                : `${data.kpis.total_tickers} stocks and ETFs`}
          </p>
        </div>
        <p className="text-xs text-amber-200/90 self-center">
          Historical outputs are not current trading signals.
        </p>
      </section>
      {(s?.mixed_signal_dates || s?.undated_signals > 0) && (
        <p className="text-sm text-amber-200">
          Some signals have different or missing dates. This is not one
          synchronized observation.
        </p>
      )}
      <section className="card" aria-labelledby="instrument-title">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <h2 id="instrument-title" className="text-lg font-semibold">
              Find an instrument
            </h2>
            <p className="mt-1 text-xs text-gray-400">
              Open an analysis for model votes, price history and explanations.
            </p>
          </div>
          <div className="w-full sm:w-72">
            <label htmlFor="instrument-search" className="sr-only">
              Search ticker or company
            </label>
            <input
              id="instrument-search"
              type="search"
              className="input w-full"
              placeholder="Search ticker or company"
              value={query}
              onChange={(e) => {
                setQuery(e.target.value);
                setAll(false);
              }}
            />
          </div>
        </div>
        {loading ? (
          <p role="status" className="py-6 text-sm text-gray-400">
            {isSlow
              ? "Still waiting. The server may be starting or busy."
              : "Loading instruments…"}
          </p>
        ) : error ? (
          <Failure message={error} retry={refetch} />
        ) : !data?.instruments ? (
          <p className="py-6">
            Instrument list unavailable in this backend version.{" "}
            <Link to="/market">Open market explorer</Link>.
          </p>
        ) : !rows.length ? (
          <p className="py-6">No instruments match your search.</p>
        ) : (
          <>
            <div className="overflow-x-auto mt-5">
              <table className="data-table">
                <thead>
                  <tr>
                    <th scope="col">Instrument</th>
                    <th scope="col">Saved vote</th>
                    <th scope="col">Model agreement</th>
                  </tr>
                </thead>
                <tbody>
                  {(all ? rows : rows.slice(0, 8)).map((r) => (
                    <tr key={r.ticker}>
                      <th scope="row" className="px-4 py-3 text-left">
                        <Link
                          to={`/detail/${r.ticker}`}
                          className="inline-block text-sm font-semibold text-white hover:text-indigo-300"
                        >
                          {r.ticker}
                          <span className="block text-xs font-normal text-gray-400 mt-1">
                            {companyName(r.ticker)}
                          </span>
                        </Link>
                      </th>
                      <td>
                        <span
                          className={
                            r.overall_signal === "BUY"
                              ? "text-green-400"
                              : r.overall_signal === "SELL"
                                ? "text-red-400"
                                : "text-amber-300"
                          }
                        >
                          {r.overall_signal || "Unavailable"}
                        </span>
                      </td>
                      <td className="text-gray-300">
                        {r.agreement_level || "Unavailable"}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="mt-4 flex flex-wrap justify-between gap-3 text-xs text-gray-400">
              <p>
                Showing {all ? rows.length : Math.min(rows.length, 8)} of{" "}
                {rows.length} instruments · agreement is not measured accuracy
              </p>
              {rows.length > 8 && (
                <button
                  className="text-indigo-300"
                  onClick={() => setAll((v) => !v)}
                >
                  {all ? "Show fewer" : "Show all instruments"}
                </button>
              )}
            </div>
          </>
        )}
      </section>
      <div className="grid gap-6 xl:grid-cols-[minmax(0,2fr)_minmax(0,1fr)]">
        <BacktestPreview />
        <section className="card" aria-labelledby="forward-title">
          <p className="eyebrow">Independent of historical simulation</p>
          <h2 id="forward-title" className="mt-2 text-lg font-semibold">
            Forward evaluation
          </h2>
          <p className="mt-3 text-sm leading-relaxed text-gray-400">
            New forecasts are recorded before their outcomes. Results become
            available after five trading sessions and depend on the number of
            resolved forecasts.
          </p>
          <Link to="/track-record" className="btn-secondary inline-block mt-5">
            Check forecast outcomes →
          </Link>
          <hr className="border-gray-800 my-5" />
          <h3 className="text-sm font-medium">Model research</h3>
          <p className="mt-2 text-xs leading-relaxed text-gray-400">
            Compare models with baselines, calibration and explanation
            diagnostics.
          </p>
          <Link
            to="/compare"
            className="inline-block mt-3 text-sm text-indigo-300"
          >
            Open comparison →
          </Link>
        </section>
      </div>
      <section className="card" aria-labelledby="archive-title">
        <h2 id="archive-title">
          <button
            className="w-full flex justify-between gap-4 text-left text-sm font-medium"
            aria-expanded={archive}
            aria-controls="archive-content"
            onClick={() => setArchive((v) => !v)}
          >
            Saved news archive
            <span aria-hidden="true">{archive ? "−" : "+"}</span>
          </button>
        </h2>
        <p className="mt-2 text-xs text-gray-400">
          Optional context with publication dates and VADER language scores.
        </p>
        <div id="archive-content">{archive && <NewsArchive />}</div>
      </section>
    </div>
  );
}
