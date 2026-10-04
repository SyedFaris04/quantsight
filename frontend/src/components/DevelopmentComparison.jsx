import { useState } from 'react';
import api, { useApi } from '../hooks/useApi';
import ConfidenceAudit from './ConfidenceAudit';

const names = {
  fit_prior: 'Training-prior baseline', calibration_prior: 'Recent-prior baseline', always_up: 'Always UP (P=100%)',
  logistic_raw: 'Logistic · raw', logistic_sigmoid: 'Logistic · calibrated',
  hist_boost_raw: 'Histogram boosting · raw', hist_boost_sigmoid: 'Histogram boosting · calibrated',
  xgboost_raw: 'XGBoost · raw', xgboost_sigmoid: 'XGBoost · calibrated',
};
const pct = value => `${(value * 100).toFixed(2)}%`;
const score = value => value.toFixed(6);

export default function DevelopmentComparison() {
  const { data, loading, error, refetch } = useApi('/research/development-models');
  const [period, setPeriod] = useState('mean');
  const [candidate, setCandidate] = useState('logistic_raw');
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState('');
  async function download() {
    setExporting(true); setExportError('');
    try {
      const response = await api.get('/research/development-models/export', { responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement('a'); link.href = url; link.download = 'quantsight-development-models.json';
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch { setExportError('The report could not be downloaded. Please retry.'); }
    finally { setExporting(false); }
  }
  if (loading) return <div className="card text-sm text-gray-400" role="status">Loading development model comparison…</div>;
  if (error || !data?.available) return <section className="card" aria-label="Finance development comparison">
    <h2 className="font-semibold text-white">Finance model development</h2>
    <p className="text-sm text-gray-400 mt-2">The comparison report is currently unavailable.</p>
    <button onClick={refetch} className="btn-secondary mt-3">Retry model comparison</button>
  </section>;
  const report = data.report;
  const rows = period === 'mean' ? report.summaries.map(row => ({ ...row, metrics: row.mean_metrics }))
    : report.results.filter(row => row.fold === period);
  const detail = report.results.find(row => row.candidate === candidate && row.fold === (period === 'mean' ? 'validate_2024' : period));
  const summary = report.summaries.find(row => row.candidate === candidate);
  const baseline = report.summaries.find(row => row.candidate === 'fit_prior');
  const modelWin = report.summaries.some(row => !['fit_prior', 'calibration_prior', 'always_up'].includes(row.candidate)
    && row.mean_metrics.brier_score < baseline.mean_metrics.brier_score);
  return <section className="rounded-2xl border border-cyan-500/30 bg-gray-900 overflow-hidden" aria-label="Finance development comparison">
    <div className="p-5 sm:p-6 border-b border-gray-800 bg-gradient-to-r from-cyan-950/40 to-gray-900">
      <div className="flex flex-wrap gap-2 text-xs"><span className="rounded-full bg-amber-400/10 text-amber-200 px-3 py-1">Development only · 2022–2024</span>
        <span className="text-gray-400 py-1">2025 candidate test remains unopened</span></div>
      <h2 className="text-xl font-semibold text-white mt-3">Do the models beat a simple baseline?</h2>
      <p className="text-sm text-gray-300 mt-2 max-w-3xl">Three fixed finance models, separate calibration years and matching observations.
        Accuracy is shown alongside probability quality and the tendency to predict UP.</p>
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mt-5">
        <div><p className="text-xs text-gray-400">Validation observations</p><p className="text-xl font-semibold text-white">{report.validation_rows.toLocaleString()}</p></div>
        <div><p className="text-xs text-gray-400">Instruments / fixed inputs</p><p className="text-xl font-semibold text-white">{report.tickers} / 13</p></div>
        <div><p className="text-xs text-gray-400">Predictor + calibration fits</p><p className="text-xl font-semibold text-white">{report.fits.predictors} + {report.fits.calibrators}</p></div>
      </div>
    </div>
    <div className="p-5 sm:p-6 space-y-5">
      <div className="rounded-lg border border-amber-400/20 bg-amber-400/5 p-4 text-sm text-amber-100 leading-relaxed">
        <strong>{modelWin ? 'Development comparison completed.' : 'No tested predictor beat the training-prior baseline on mean Brier loss.'}</strong>
        <p className="mt-1">The fixed selection rule chose {names[report.selected_candidate]}. This is evidence from reused development years, not a final test or a reason to replace the live model.</p>
      </div>
      <div className="flex flex-wrap justify-between items-end gap-3">
        <label className="text-sm text-gray-300 flex flex-col gap-1">Evaluation period
          <select value={period} onChange={event => setPeriod(event.target.value)} className="bg-gray-950 border border-gray-600 rounded-lg p-2">
            <option value="mean">2022–2024 · equal-year mean</option>
            {report.folds.map(fold => <option key={fold.name} value={fold.name}>{fold.name.slice(-4)}</option>)}
          </select>
        </label>
        <button onClick={download} disabled={exporting} className="btn-secondary text-sm">{exporting ? 'Downloading…' : 'Download study JSON'}</button>
      </div>
      {exportError && <p role="alert" className="text-sm text-red-300">{exportError}</p>}
      <p className="text-xs text-gray-500 sm:hidden">Swipe the table sideways to see all metrics.</p>
      <div tabIndex={0} role="region" aria-label="Model scores, scroll horizontally for all metrics" className="overflow-x-auto focus:outline-none focus:ring-2 focus:ring-cyan-500 rounded-lg">
        <table className="w-full text-sm whitespace-nowrap text-left">
          <thead className="text-xs text-gray-400"><tr>{['Candidate', 'Accuracy ↑', 'Balanced acc. ↑', 'MCC ↑', 'ROC-AUC ↑', 'Brier loss ↓', 'Predicts UP'].map(title => <th key={title} className="py-3 px-3">{title}</th>)}</tr></thead>
          <tbody>{rows.map(row => <tr key={row.candidate} className={`border-t border-gray-800 ${row.candidate === 'fit_prior' ? 'bg-cyan-900/15' : ''}`}>
            <th scope="row" className="px-3 py-3 font-medium text-gray-200">{names[row.candidate]}</th>
            <td className="px-3 text-gray-300">{pct(row.metrics.accuracy)}</td><td className="px-3 text-gray-300">{pct(row.metrics.balanced_accuracy)}</td>
            <td className="px-3 text-gray-300">{row.metrics.mcc.toFixed(3)}</td><td className="px-3 text-gray-300">{row.metrics.roc_auc.toFixed(3)}</td>
            <td className="px-3 text-cyan-200 font-mono">{score(row.metrics.brier_score)}</td><td className="px-3 text-gray-300">{pct(row.metrics.buy_fraction)}</td>
          </tr>)}</tbody>
        </table>
      </div>
      <p className="text-xs text-gray-400 leading-relaxed">Lower Brier loss means better probability forecasts overall, not calibration alone.
        Balanced accuracy weights UP and DOWN equally; MCC measures agreement beyond chance (−1 to 1).
        The training-prior baseline gives every instrument the UP frequency from the fit period; the recent-prior baseline uses the calibration year.
        Always UP uses P(UP)=100%, so its probability loss can be poor even when its directional accuracy is reasonable.</p>
      <details className="text-sm text-gray-300">
        <summary className="cursor-pointer font-medium text-white">Probability reliability, uncertainty and method</summary>
        <div className="mt-4 space-y-4">
          <label className="flex flex-col gap-1 text-xs text-gray-400">Inspect candidate
            <select value={candidate} onChange={event => setCandidate(event.target.value)} className="bg-gray-950 border border-gray-600 rounded-lg p-2 text-sm text-white w-full sm:w-auto">
              {Object.entries(names).map(([key, name]) => <option key={key} value={key}>{name}</option>)}
            </select>
          </label>
          <p className="text-xs text-gray-400 leading-relaxed">Equal-year Brier improvement versus training prior: <strong className="text-gray-200">{score(summary.brier_gain_vs_fit_prior)}</strong>.
            {' '}Exploratory 95% interval: [{score(summary.interval_95[0])}, {score(summary.interval_95[1])}]. Positive means lower loss.
            {summary.interval_95[0] <= 0 && summary.interval_95[1] >= 0 ? ' The interval includes zero.' : ''}
            {' '}Uses 1,000 paired resamples of 20-session blocks within each year, keeping tickers together. These intervals are not adjusted for choosing among multiple candidates.</p>
          <p className="text-xs text-gray-400">Reliability for {detail.year}{period === 'mean' ? ' (select a year above to inspect another period)' : ''}: compare predicted P(UP) with the actual UP rate in each occupied bin.</p>
          <p className="text-xs text-gray-500 sm:hidden">Swipe the reliability table sideways for observed outcomes.</p>
          <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Probability reliability table">
            <table className="w-full text-xs text-left whitespace-nowrap"><thead><tr>{['Probability bin', 'Rows', 'Mean predicted UP', 'Observed UP'].map(title => <th key={title} className="p-2 text-gray-400">{title}</th>)}</tr></thead>
              <tbody>{detail.reliability.filter(bin => bin.n > 0).map(bin => <tr key={bin.lower} className="border-t border-gray-800"><td className="p-2">{pct(bin.lower)}–{pct(bin.upper)}</td>
                <td className="p-2">{bin.n.toLocaleString()}</td><td className="p-2">{pct(bin.mean_probability)}</td><td className="p-2">{pct(bin.observed_up_rate)}</td></tr>)}</tbody>
            </table>
          </div>
          <ul className="list-disc pl-5 space-y-1 text-xs text-gray-400 leading-relaxed">
            <li>Five-session adjusted-close direction. Whole dates remain together; outcomes crossing segment boundaries are purged.</li>
            <li>Fixed 13 finance inputs; scaling is fitted on training data. Raw and separately calibrated outputs use their own 50% decision threshold.</li>
            <li>The lowest equal-year Brier loss selects among all registered outputs and baselines. No extra tuning followed these results.</li>
            <li>Fixed survivor universe and retrospectively adjusted prices limit interpretation. No sentiment or after-cost trading benefit is established here.</li>
          </ul>
          <p className="text-xs text-gray-500 break-all">Run {report.run_id} · Protocol SHA-256 {report.registration.protocol_sha256}</p>
        </div>
      </details>
      <ConfidenceAudit names={names} parentRunId={report.run_id} />
    </div>
  </section>;
}
