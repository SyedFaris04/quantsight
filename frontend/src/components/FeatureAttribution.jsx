import { useState } from "react";
import { useApi } from "../hooks/useApi";

const labels = {
  rsi: "Relative strength (RSI)", return_5: "5-session return", return_10: "10-session return",
  bb_width: "Bollinger band width", volatility_20: "20-session volatility", atr: "Average true range",
  volume_change: "Volume change", volume_sma_10: "10-session average volume",
  clv: "Close location value", cmf_20: "Chaikin money flow", svi_10: "Signed volume index",
  mfi_14: "Money flow index", rsi_rel_market: "RSI relative to market",
  return5_rel_market: "Return relative to market", return5_rel_spy: "Return relative to SPY",
  volchg_rel_market: "Volume change relative to market", market_volatility: "Market volatility",
  ret5_rel_sector: "Return relative to sector", wsb_sentiment: "WSB sentiment", wsb_count: "WSB post count",
  gdelt_compound: "News sentiment", gdelt_pos: "Positive news sentiment", gdelt_neg: "Negative news sentiment",
  gdelt_article_count: "News article count",
};
const label = (name) => labels[name] || name.replace(/_/g, " ").replace(/norm$/, "(normalised)");
const signed = (value) => `${value >= 0 ? "+" : ""}${value.toFixed(4)}`;
const percent = (value) => `${(value * 100).toFixed(2)}%`;

export default function FeatureAttribution({ ticker }) {
  const [model, setModel] = useState("xgb_finance");
  const { data, loading, error, refetch } = useApi(
    `/feature-attribution/${encodeURIComponent(ticker)}?model=${model}`,
  );
  // Do not show an earlier ticker/model response while a new request starts.
  const ready = data?.ticker === ticker.toUpperCase() && data?.model === model && !loading && !error;
  const ranked = ready ? [...data.features].sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)) : [];
  const top = ranked.slice(0, 8);
  const rest = ranked.slice(8).reduce((sum, item) => sum + item.contribution, 0);
  const bars = ranked.length > 8 ? [...top, { feature: `Other ${ranked.length - 8} inputs (sum)`, contribution: rest }] : top;
  const max = Math.max(...bars.map((item) => Math.abs(item.contribution)), 0.001);
  return (
    <section className="card space-y-5" aria-label="XGBoost feature attribution">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold text-gray-100">What moved this model’s prediction?</h2>
          <p className="text-xs text-gray-400 mt-1">TreeSHAP · latest saved historical prediction · XGBoost only</p>
        </div>
        <label className="text-xs text-gray-400 flex flex-col gap-1">
          Explanation model
          <select value={model} onChange={(event) => setModel(event.target.value)}
            className="bg-gray-900 border border-gray-700 rounded-lg p-2 text-sm text-gray-100">
            <option value="xgb_finance">XGBoost Finance</option>
            <option value="xgb_sentiment">XGBoost + Sentiment</option>
          </select>
        </label>
      </div>
      {loading ? <p role="status" className="text-sm text-gray-400 py-6">Loading verified model contributions…</p>
        : error ? <div role="alert" className="text-sm text-amber-300">
          <p>Could not load a verified explanation. {String(error)}</p>
          <button onClick={refetch} className="mt-2 underline">Retry explanation</button>
        </div> : ready ? <>
          <div className="flex flex-wrap gap-x-4 gap-y-2 text-xs text-gray-400">
            <span>Saved input: <strong className="text-gray-200">{data.date}</strong></span>
            <span>Direction: <strong className={data.direction === "BUY" ? "text-emerald-400" : "text-red-400"}>{data.direction}</strong></span>
            <span className="text-emerald-400">✓ Output reconstruction verified</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[["Raw model P(UP)", percent(data.raw_probability_up)],
              ["Calibrated P(UP)", percent(data.calibrated_probability_up)],
              ["Explained raw score", signed(data.raw_margin)]].map(([title, value]) =>
              <div key={title} className="rounded-lg bg-gray-900/60 border border-gray-800 p-3">
                <p className="text-xs text-gray-400">{title}</p><p className="text-xl font-semibold text-gray-100 mt-1">{value}</p>
              </div>)}
          </div>
          <p className="text-sm text-gray-400 leading-relaxed">
            Positive values push the raw score toward UP; negative values push toward DOWN.
            These are log-odds contributions, not percentage-point changes or accuracy scores.
            BUY/SELL uses raw P(UP) at 50%. Separate calibration can cross 50% without changing that saved direction.
          </p>
          <div className="space-y-3" aria-label="Signed feature contributions">
            <div className="flex justify-between text-xs text-gray-500"><span>← Toward DOWN</span><span>Toward UP →</span></div>
            {bars.map((item) => <div key={item.feature}>
              <div className="flex justify-between gap-2 text-xs mb-1">
                <span className="text-gray-300 break-words">{label(item.feature)}</span>
                <span className={`font-mono shrink-0 ${item.contribution >= 0 ? "text-emerald-400" : "text-red-400"}`}>{signed(item.contribution)}</span>
              </div>
              <div className="relative h-3 rounded bg-gray-900 overflow-hidden" aria-hidden="true">
                <div className="absolute left-1/2 h-full w-px bg-gray-600" />
                <div className={`absolute h-full rounded ${item.contribution >= 0 ? "bg-emerald-500" : "bg-red-400"}`}
                  style={{ width: `${Math.abs(item.contribution) / max * 50}%`,
                    left: item.contribution >= 0 ? "50%" : `${50 - Math.abs(item.contribution) / max * 50}%` }} />
              </div>
            </div>)}
          </div>
          <div className="text-xs text-gray-400 bg-gray-900/50 rounded-lg p-3 leading-relaxed">
            Baseline {signed(data.baseline)} + all contributions {signed(data.contribution_sum)} = raw score {signed(data.reconstructed_margin)}.
            <br />The baseline is the tree model’s reference score, not a market forecast. These associations explain this model output, not causes of price changes.
            Verification checks consistency with the saved prediction; it does not prove the prediction is correct.
          </div>
          <details className="text-xs text-gray-400">
            <summary className="cursor-pointer text-gray-200">All {ranked.length} inputs and verification details</summary>
            <div className="overflow-x-auto mt-3">
              <table className="w-full text-left whitespace-nowrap">
                <thead className="text-gray-300"><tr><th className="p-2">Feature</th><th className="p-2">Saved value</th><th className="p-2">Scaled input</th><th className="p-2">Contribution</th></tr></thead>
                <tbody>{ranked.map((item) => <tr key={item.feature} className="border-t border-gray-800">
                  <td className="p-2">{item.feature}</td><td className="p-2">{item.value.toPrecision(6)}</td>
                  <td className="p-2">{item.scaled_value.toPrecision(6)}</td><td className="p-2">{signed(item.contribution)}</td>
                </tr>)}</tbody>
              </table>
            </div>
            <div className="mt-3 space-y-1 break-all">
              <p>Model SHA-256: {data.model_sha256}</p><p>Input SHA-256: {data.input_sha256}</p>
              <p>Reconstruction error: {data.reconstruction_error.toExponential(2)} · {data.trees_used} trees · XGBoost {data.xgboost_version}</p>
              <p>Calibration: {data.calibration_method}. Exact TreeSHAP from the saved trees; no approximate contribution mode.</p>
            </div>
          </details>
        </> : <p className="text-sm text-gray-400">Waiting for the selected prediction.</p>}
    </section>
  );
}
