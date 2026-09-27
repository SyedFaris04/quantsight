import { useState } from 'react';
import { useApi } from '../hooks/useApi';

const percent = value => `${(value * 100).toFixed(2)}%`;
const score = value => value.toFixed(6);

export default function ResearchComparison() {
  const { data, loading, error, refetch } = useApi('/research/news-comparison');
  const [family, setFamily] = useState('logistic');
  if (loading) return <div className="card h-48 animate-pulse" role="status" aria-label="Loading controlled research results" />;
  if (error || !data?.available) return (
    <section className="card" aria-label="Controlled news research">
      <h2 className="font-semibold text-white">Controlled news experiment</h2>
      <p className="mt-2 text-sm text-gray-400">The research report is currently unavailable.</p>
      <button className="btn-secondary mt-3 text-sm" onClick={refetch}>Retry</button>
    </section>
  );
  const r = data.report;
  const rows = r.models.filter(row => row.family === family);
  const paired = r.comparisons[family].versus_vader;
  const [low, high] = paired.interval_95;
  const interpretation = low > 0 ? 'Lower Brier loss than VADER in this exploratory comparison.'
    : high < 0 ? 'Higher Brier loss than VADER in this exploratory comparison.'
      : 'No clear improvement over VADER: the interval includes zero.';
  return (
    <section className="overflow-hidden rounded-2xl border border-indigo-500/30 bg-gray-900" aria-labelledby="news-study-title">
      <div className="border-b border-gray-800 bg-gradient-to-r from-indigo-950/60 to-gray-900 p-5 sm:p-6">
        <div className="flex flex-wrap items-center gap-3">
          <span className="rounded-full border border-amber-400/30 bg-amber-400/10 px-3 py-1 text-xs font-medium text-amber-200">Exploratory study</span>
          <span className="text-xs text-gray-400">Five-session direction · 2023 evaluation</span>
        </div>
        <h2 id="news-study-title" className="mt-3 text-xl font-semibold text-white">Does financial news improve prediction?</h2>
        <p className="mt-2 max-w-3xl text-sm leading-relaxed text-gray-300">
          Finance, news coverage, VADER and FinBERT compared on the same observations.
          The constant-probability baseline stays visible so model complexity has to earn its place.
        </p>
        <dl className="mt-5 grid grid-cols-1 gap-4 sm:grid-cols-3">
          <div><dt className="text-xs text-gray-400">Evaluation observations</dt><dd className="mt-1 text-xl font-semibold text-white">{r.segments.evaluation.rows.toLocaleString()}</dd></div>
          <div><dt className="text-xs text-gray-400">Observations with news</dt><dd className="mt-1 text-xl font-semibold text-white">{r.segments.evaluation.news_observed_rows.toLocaleString()}</dd></div>
          <div><dt className="text-xs text-gray-400">Instruments covered by news</dt><dd className="mt-1 text-xl font-semibold text-white">{r.covered_tickers} / {r.universe_tickers}</dd></div>
        </dl>
      </div>
      <div className="p-5 sm:p-6">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <label className="text-sm text-gray-300" htmlFor="research-model-family">Prediction model</label>
          <select id="research-model-family" value={family} onChange={event => setFamily(event.target.value)}
            className="rounded-lg border border-gray-600 bg-gray-950 px-3 py-2 text-sm text-white focus:outline-none focus:ring-2 focus:ring-indigo-400">
            <option value="logistic">Logistic regression</option>
            <option value="hist_gradient_boosting">Gradient boosting</option>
          </select>
        </div>
        <div className="mt-4 overflow-x-auto" role="region" aria-label="Research metric comparison" tabIndex={0}>
          <table className="w-full min-w-[560px] text-left text-sm">
            <caption className="sr-only">All variants share the same 2023 evaluation rows. Lower Brier loss is better.</caption>
            <thead className="border-b border-gray-700 text-xs text-gray-400"><tr>
              <th scope="col" className="py-3 pr-4">Inputs</th><th scope="col" className="px-3 py-3 text-right">Accuracy</th>
              <th scope="col" className="px-3 py-3 text-right">Balanced accuracy</th><th scope="col" className="pl-3 py-3 text-right">Brier loss ↓</th>
            </tr></thead>
            <tbody>
              <tr className="border-b border-gray-800 bg-gray-800/40 text-gray-200">
                <th scope="row" className="py-3 pl-2 pr-4 font-medium">Constant training prior</th>
                <td className="px-3 py-3 text-right tabular-nums">{percent(r.baseline.accuracy)}</td>
                <td className="px-3 py-3 text-right tabular-nums">{percent(r.baseline.balanced_accuracy)}</td>
                <td className="pl-3 py-3 text-right tabular-nums">{score(r.baseline.brier_score)}</td>
              </tr>
              {rows.map(row => <tr key={row.key} className="border-b border-gray-800 text-gray-300">
                <th scope="row" className="py-3 pl-2 pr-4 font-medium">{row.inputs}</th>
                <td className="px-3 py-3 text-right tabular-nums">{percent(row.metrics.accuracy)}</td>
                <td className="px-3 py-3 text-right tabular-nums">{percent(row.metrics.balanced_accuracy)}</td>
                <td className="pl-3 py-3 text-right tabular-nums">{score(row.metrics.brier_score)}</td>
              </tr>)}
            </tbody>
          </table>
        </div>
        <p className="mt-3 text-xs leading-relaxed text-gray-400">The constant baseline uses the training-period UP frequency. Always predicting UP scores {percent(r.always_up_accuracy)} accuracy. Balanced accuracy weighs UP and DOWN equally.</p>
        <div className="mt-5 rounded-xl border border-gray-700 bg-gray-950/60 p-4">
          <h3 className="text-sm font-semibold text-white">FinBERT versus VADER</h3>
          <p className="mt-2 text-sm text-gray-300">{interpretation}</p>
          <p className="mt-2 text-xs leading-relaxed text-gray-400">Brier improvement: {score(paired.brier_improvement)} · 95% block-bootstrap interval: [{score(low)}, {score(high)}]. Positive values favor FinBERT. These intervals are exploratory and do not correct for multiple comparisons.</p>
        </div>
        <details className="mt-5 text-sm text-gray-300">
          <summary className="cursor-pointer rounded py-1 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-400">Method and limitations</summary>
          <p className="mt-3 leading-relaxed">Fit: 2017–2021. Separate calibration: 2022. Evaluation: 2023. Five-session labels are purged at segment boundaries. Two new FinBERT variants use the previous study's frozen controls; no hyperparameter search.</p>
          <ul className="mt-3 list-disc space-y-2 pl-5 text-gray-400">{r.limitations.map(item => <li key={item}>{item}</li>)}</ul>
          <p className="mt-3 break-all text-xs text-gray-500">Experiment {r.run_id} · Parent {r.parent_run_id}</p>
          <div className="mt-3 flex flex-wrap gap-4">{r.sources.map(source => <a key={source.url} href={source.url} target="_blank" rel="noreferrer" className="text-indigo-300 underline hover:text-indigo-200">{source.label}</a>)}</div>
        </details>
      </div>
    </section>
  );
}
