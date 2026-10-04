const pct = value => `${(value * 100).toFixed(2)}%`;
const points = value => `${value > 0 ? "+" : ""}${(value * 100).toFixed(2)} pp`;
const tone = value => value > 1e-12 ? "text-emerald-300" : value < -1e-12 ? "text-rose-300" : "text-gray-300";

function PeriodTable({ rows, name, strategyName }) {
  return <div className="overflow-x-auto focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400"
    role="region" aria-label={name} tabIndex={0}>
    <table className="data-table">
      <caption className="sr-only">{strategyName} net returns and benchmarks on matching observed dates. Differences are percentage points.</caption>
      <thead><tr><th scope="col">Period</th><th scope="col">Observed dates</th><th scope="col">Sessions</th>
        <th scope="col">Selected net return</th><th scope="col">SPY</th><th scope="col">Equal-weight universe</th><th scope="col">Return minus SPY</th></tr></thead>
      <tbody>{rows.map(row => <tr key={row.period}>
        <th scope="row" className="font-medium text-left whitespace-nowrap">{row.period}</th>
        <td className="whitespace-nowrap text-gray-400">{row.start} to {row.end}</td><td>{row.sessions}</td>
        <td className={`tabular-nums ${tone(row.net_return)}`}>{pct(row.net_return)}</td>
        <td className="tabular-nums">{pct(row.spy_return)}</td><td className="tabular-nums">{pct(row.universe_return)}</td>
        <td className={`tabular-nums whitespace-nowrap ${tone(row.difference_vs_spy)}`}>{points(row.difference_vs_spy)}</td>
      </tr>)}</tbody>
    </table>
  </div>;
}

export default function PeriodPerformance({ analysis, strategy }) {
  const periods = analysis?.available && analysis.report.strategies[strategy.key];
  if (!periods) return <section className="card" aria-label="Performance over time">
    <h2 className="text-lg font-semibold">Performance over time</h2>
    <p className="mt-2 text-sm text-gray-400">The year and month breakdown is unavailable for this saved run. The full-period results remain above.</p>
  </section>;
  const summary = periods.summary;
  const best = periods.months.find(row => row.period === summary.best_month);
  const worst = periods.months.find(row => row.period === summary.worst_month);
  return <section className="card !p-0 overflow-hidden" aria-label="Performance over time">
    <div className="p-5">
      <h2 className="text-lg font-semibold">Performance over time</h2>
      <p className="mt-2 text-sm text-gray-300">{strategy.name}: check whether the overall result holds across different periods.</p>
      <p className="mt-2 text-xs leading-relaxed text-gray-400">Returns cover the dates shown, including partial years and months. They include trading costs and are not annualized. Holdings continue across period boundaries.</p>
      <dl className="mt-5 grid sm:grid-cols-3 gap-4">
        <div><dt className="text-xs text-gray-400">Months beating SPY</dt><dd className="mt-1 text-lg tabular-nums">{summary.months_beating_spy} / {summary.months}</dd>
          <p className="mt-1 text-xs text-gray-400">{summary.months_tied_spy} tied; descriptive consistency, not prediction accuracy.</p></div>
        <div><dt className="text-xs text-gray-400">Best observed month</dt><dd className="mt-1 text-lg tabular-nums">{best.period}: {pct(best.net_return)}</dd></div>
        <div><dt className="text-xs text-gray-400">Worst observed month</dt><dd className="mt-1 text-lg tabular-nums">{worst.period}: {pct(worst.net_return)}</dd></div>
      </dl>
      <h3 className="mt-5 font-medium text-gray-200">Year breakdown · observed study windows</h3>
    </div>
    <PeriodTable rows={periods.years} name="Year performance comparison" strategyName={strategy.name} />
    <details className="p-5 border-t border-gray-800">
      <summary className="cursor-pointer font-medium text-gray-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-indigo-400">Monthly comparison · {summary.months} observed months</summary>
      <div className="mt-4"><PeriodTable rows={periods.months} name="Monthly performance comparison" strategyName={strategy.name} /></div>
    </details>
    <p className="px-5 pb-5 text-xs leading-relaxed text-gray-400">Periods compound daily returns, so they do not add up arithmetically. This historical breakdown does not establish statistical significance or performance on fresh data.</p>
  </section>;
}
