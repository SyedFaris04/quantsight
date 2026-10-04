import { useState } from 'react';
import api, { useApi } from '../hooks/useApi';

const pct = value => value == null ? '—' : `${(value * 100).toFixed(2)}%`;
const score = value => value == null ? '—' : value.toFixed(6);

export default function ConfidenceAudit({ names, parentRunId }) {
  const [requested, setRequested] = useState(false);
  const { data, loading, error, refetch } = useApi(requested ? '/research/development-confidence' : null);
  const [candidate, setCandidate] = useState('xgboost_raw');
  const [period, setPeriod] = useState('pooled');
  const [exporting, setExporting] = useState(false);
  const [exportError, setExportError] = useState('');
  const report = data?.report;
  const available = data?.available && report.parent_run_id === parentRunId;
  const selected = available ? report.periods[period] : null;
  async function download() {
    setExporting(true); setExportError('');
    try {
      const response = await api.get('/research/development-confidence/export', { responseType: 'blob' });
      const url = URL.createObjectURL(response.data);
      const link = document.createElement('a'); link.href = url; link.download = 'quantsight-development-confidence.json';
      document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch { setExportError('The confidence report could not be downloaded. Please retry.'); }
    finally { setExporting(false); }
  }
  return <details className="rounded-xl border border-gray-700 bg-gray-950/40 p-4 sm:p-5"
    onToggle={event => { if (event.currentTarget.open) setRequested(true); }}>
    <summary className="cursor-pointer font-medium text-cyan-200 focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400">Does stronger confidence improve accuracy?</summary>
    <div className="mt-4 space-y-4" role="region" aria-label="Confidence and coverage audit">
      <p className="text-sm leading-relaxed text-gray-300">See what happens when only predictions with stronger model confidence are retained. Higher confidence can mean fewer predictions; it does not guarantee fewer mistakes.</p>
      {loading && <p className="text-sm text-gray-400" role="status">Loading confidence audit…</p>}
      {!loading && requested && (error || !available) && <div>
        <p className="text-sm text-gray-400">The confidence audit is currently unavailable. The main development results remain above.</p>
        <button className="btn-secondary mt-3" onClick={refetch}>Retry confidence audit</button>
      </div>}
      {selected && <>
        <div className="flex flex-wrap items-end gap-4">
          <label className="text-sm text-gray-300">Confidence candidate
            <select aria-label="Confidence candidate" value={candidate} onChange={event => setCandidate(event.target.value)} className="block mt-1 w-full bg-gray-950 border border-gray-600 rounded-lg p-2">
              {report.protocol.candidates.map(key => <option key={key} value={key}>{names[key]}</option>)}
            </select>
          </label>
          <label className="text-sm text-gray-300">Confidence period
            <select aria-label="Confidence period" value={period} onChange={event => setPeriod(event.target.value)} className="block mt-1 w-full bg-gray-950 border border-gray-600 rounded-lg p-2">
              <option value="pooled">2022–2024 · pooled rows</option>
              {report.protocol.years.map(year => <option key={year} value={String(year)}>{year}</option>)}
            </select>
          </label>
          <button onClick={download} disabled={exporting} className="btn-secondary text-sm">{exporting ? 'Downloading…' : 'Download confidence JSON'}</button>
        </div>
        {exportError && <p role="alert" className="text-sm text-red-300">{exportError}</p>}
        <p className="text-xs leading-relaxed text-gray-400">{selected.total_rows.toLocaleString()} eligible ticker-days. Confidence is the larger of P(UP) and P(DOWN). The training-prior baseline is evaluated on the exact same retained rows as the selected candidate.</p>
        <div className="overflow-x-auto rounded-lg focus-visible:outline focus-visible:outline-2 focus-visible:outline-cyan-400"
          role="region" aria-label="Confidence threshold comparison" tabIndex={0}>
          <table className="w-full whitespace-nowrap text-sm text-left">
            <caption className="sr-only">{names[candidate]} confidence thresholds, coverage and conditional prediction quality. Empty selections have undefined scores. All-years results are weighted by retained observations.</caption>
            <thead className="text-xs text-gray-400"><tr>{['Minimum confidence', 'Retained rows', 'Coverage', 'Signal dates', 'Mean confidence', 'Accuracy', 'Baseline accuracy', 'Brier loss ↓', 'Baseline Brier ↓', 'Predicts UP'].map(label => <th scope="col" key={label} className="py-3 px-3">{label}</th>)}</tr></thead>
            <tbody>{selected.candidates[candidate].map(row => <tr key={row.threshold} className="border-t border-gray-800">
              <th scope="row" className="px-3 py-3 font-medium text-gray-200">≥ {pct(row.threshold)}</th>
              <td className="px-3 text-gray-300">{row.retained_rows.toLocaleString()}{row.small_sample && <span className="block text-xs text-amber-200">Small sample</span>}</td>
              <td className="px-3 text-gray-300">{pct(row.coverage)}</td><td className="px-3 text-gray-300">{row.retained_dates.toLocaleString()}</td>
              <td className="px-3 text-gray-300">{pct(row.mean_confidence)}</td><td className="px-3 text-gray-300">{pct(row.accuracy)}</td>
              <td className="px-3 text-gray-300">{pct(row.baseline_accuracy)}</td><td className="px-3 font-mono text-cyan-200">{score(row.brier_score)}</td>
              <td className="px-3 font-mono text-gray-300">{score(row.baseline_brier)}</td><td className="px-3 text-gray-300">{pct(row.predicts_up)}</td>
            </tr>)}</tbody>
          </table>
        </div>
        <p className="text-xs leading-relaxed text-amber-100/90">All thresholds are fixed diagnostics. No threshold is recommended or applied to live signals. Small sample marks fewer than 200 rows or 20 signal dates; a larger sample still does not establish reliability. Empty selections show undefined scores.</p>
        <p className="text-xs leading-relaxed text-gray-400">Pooled scores weight each retained observation, unlike the equal-year means above. Samples differ across candidates and thresholds, so compare each with its matching baseline. These previously studied years, overlapping outcomes and dependent stocks do not provide a fresh test or significance guarantee.</p>
      </>}
    </div>
  </details>;
}
