import { Link } from "react-router-dom";
import { useApi } from "../hooks/useApi";
import { companyName } from "../data/companyNames";

const display = value => value == null ? "—" : value;
const percent = value => value == null ? "—" : `${value}%`;
const focus = "focus-visible:outline focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-indigo-400";

function Metric({ label, value, note }) {
  return <div className="rounded-xl border border-gray-800 bg-gray-900 p-5">
    <dt className="text-xs font-medium text-gray-400">{label}</dt>
    <dd className="mt-2 text-3xl font-semibold tabular-nums text-white">{value}</dd>
    <p className="mt-2 text-xs leading-relaxed text-gray-400">{note}</p>
  </div>;
}

function Unavailable({ message, retry }) {
  return <div role="alert" className="rounded-xl border border-gray-700 bg-gray-900 p-5">
    <p className="text-sm text-gray-300">{message}</p>
    <button onClick={retry} className={`btn-secondary mt-3 text-sm ${focus}`}>Try again</button>
  </div>;
}

function NewsSentiment() {
  const { data, loading, error, refetch } = useApi("/market-sentiment");
  return <section className="card" aria-labelledby="news-sentiment-title">
    <p className="text-xs font-medium uppercase tracking-wider text-indigo-300">Language analysis</p>
    <h2 id="news-sentiment-title" className="mt-2 text-lg font-semibold text-white">Saved news sentiment</h2>
    <p className="mt-2 text-sm leading-relaxed text-gray-400">VADER scores headline language. These scores are separate from the historical prediction snapshot.</p>
    {loading ? <p role="status" className="mt-5 text-sm text-gray-400">Loading news analysis...</p>
      : error ? <div className="mt-4"><Unavailable message="News analysis could not be loaded." retry={refetch} /></div>
      : !data?.article_count ? <p className="mt-5 text-sm text-gray-400">No dated headlines are available to score.</p>
      : <>
        <div className="mt-4 rounded-lg border border-gray-700 bg-gray-950/50 p-3 text-xs leading-relaxed text-gray-300">
          <p>Archive window: {data.window_start || "Unknown"} to {data.window_end || "Unknown"}</p>
          <p className="mt-1">Latest saved article: {data.age_days == null ? "age unavailable" : `${data.age_days} days old`}. This is not a live news feed.</p>
        </div>
        <dl className="mt-5 grid grid-cols-3 gap-3 text-center">
          {[["Positive", "positive_pct", "text-emerald-300"], ["Neutral", "neutral_pct", "text-gray-200"], ["Negative", "negative_pct", "text-rose-300"]].map(([label, key, color]) =>
            <div key={key}><dt className="text-xs text-gray-400">{label}</dt><dd className={`mt-2 text-xl font-semibold tabular-nums ${color}`}>{percent(data[key])}</dd></div>)}
        </dl>
        <p className="mt-4 text-xs text-gray-400">{data.article_count.toLocaleString()} unique headlines in a {data.days}-day archive window.</p>
        <details className="mt-4 text-xs text-gray-400"><summary className={`cursor-pointer text-gray-300 ${focus}`}>Daily sentiment values</summary>
          <div className="mt-3 max-h-48 overflow-auto" role="region" aria-label="Daily saved-news sentiment" tabIndex={0}>
            <table className="w-full text-left"><caption className="sr-only">Mean VADER compound score by article date, from minus one to plus one.</caption>
              <thead><tr><th scope="col" className="py-2">Article date</th><th scope="col" className="py-2 text-right">Mean score</th></tr></thead>
              <tbody>{data.trend?.map(row => <tr key={row.date} className="border-t border-gray-800"><td className="py-2">{row.date}</td><td className="py-2 text-right tabular-nums">{row.avg_compound.toFixed(3)}</td></tr>)}</tbody>
            </table>
          </div>
        </details>
      </>}
  </section>;
}

const pathways = [
  { to: "/backtesting", label: "Backtesting", tag: "Historical simulation", text: "Compare net returns, Sharpe, drawdown and trading costs against benchmarks." },
  { to: "/compare", label: "AI Compare", tag: "Controlled research", text: "Inspect finance, VADER and FinBERT results alongside simple baselines." },
  { to: "/track-record", label: "Track Record", tag: "Forward evaluation", text: "Follow recorded forecasts and their outcomes after five trading sessions." },
];

function articleUrl(value) {
  try { const url = new URL(value); return ["https:", "http:"].includes(url.protocol) ? url.href : null; }
  catch { return null; }
}

export default function Dashboard() {
  const { data, loading, error, refetch } = useApi("/dashboard");
  const kpis = data?.kpis;
  const snapshot = data?.snapshot;
  const dates = snapshot?.latest_signal_date
    ? snapshot.earliest_signal_date === snapshot.latest_signal_date ? snapshot.latest_signal_date
      : `${snapshot.earliest_signal_date} to ${snapshot.latest_signal_date}`
    : "Date unavailable";
  const ready = !loading && !error && kpis;
  return <div className="mx-auto max-w-7xl space-y-6">
    <section className="rounded-2xl border border-indigo-500/25 bg-gradient-to-br from-indigo-950/60 via-gray-900 to-gray-900 p-6 sm:p-8">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <p className="text-xs font-semibold uppercase tracking-widest text-indigo-300">QuantSight research workspace</p>
        <span className="rounded-full border border-indigo-400/30 bg-indigo-400/10 px-3 py-1 text-xs text-indigo-200">Five-session prediction horizon</span>
      </div>
      <h1 className="mt-4 text-3xl font-semibold tracking-tight text-white sm:text-4xl">Understand the signal. Check the evidence.</h1>
      <p className="mt-3 max-w-2xl text-sm leading-relaxed text-gray-300">Explore model predictions, compare their historical performance, and follow outcomes as new forecasts mature.</p>
      <div className="mt-6 flex flex-wrap gap-3">
        <Link to="/backtesting" className={`btn-primary text-sm ${focus}`}>Explore backtesting</Link>
        <Link to="/market" className={`btn-secondary text-sm ${focus}`}>Open market explorer</Link>
      </div>
    </section>

    <section aria-labelledby="snapshot-title" className="space-y-4">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div><h2 id="snapshot-title" className="text-lg font-semibold text-white">Historical model snapshot</h2>
          <p className="mt-1 text-sm text-gray-400">Saved model outputs for research; these are not current trading signals.</p></div>
        <p className="rounded-lg border border-gray-700 px-3 py-2 text-xs text-gray-300">Signal dates: {loading ? "Loading..." : error ? "Unavailable" : dates}</p>
      </div>
      {loading ? <div role="status" className="grid grid-cols-2 gap-4 lg:grid-cols-4"><span className="sr-only">Loading historical snapshot</span>{[1,2,3,4].map(i => <div key={i} className="h-32 animate-pulse rounded-xl bg-gray-900" />)}</div>
        : error || !kpis ? <Unavailable message="The historical snapshot could not be loaded." retry={refetch} />
        : <>
          {(snapshot?.mixed_signal_dates || snapshot?.undated_signals > 0) && <p className="rounded-lg border border-amber-700/50 bg-amber-950/20 p-3 text-xs text-amber-100">Some saved signals have different or missing dates. This summary does not represent one synchronized market observation.</p>}
          <dl className="grid grid-cols-2 gap-4 lg:grid-cols-4">
            <Metric label="Instruments covered" value={display(kpis.total_tickers)} note="Stocks and ETFs in the saved snapshot" />
            <Metric label="Historical BUY votes" value={display(kpis.buy_signals)} note={`${display(kpis.sell_signals)} SELL / ${display(kpis.hold_signals)} HOLD`} />
            <Metric label="Unanimous model votes" value={display(kpis.strong_agreement)} note="All available models agree on direction" />
            <Metric label="Mean model confidence" value={percent(kpis.avg_confidence)} note="Average model score; not ensemble accuracy" />
          </dl>
        </>}
    </section>

    <nav aria-label="Explore the evidence" className="grid gap-4 md:grid-cols-3">
      {pathways.map(item => <Link key={item.to} to={item.to} className={`group rounded-xl border border-gray-800 bg-gray-900 p-5 transition-colors hover:border-indigo-500/60 ${focus}`}>
        <p className="text-xs font-medium text-indigo-300">{item.tag}</p><h2 className="mt-2 text-lg font-semibold text-white">{item.label} <span aria-hidden="true" className="text-indigo-400">→</span></h2>
        <p className="mt-2 text-sm leading-relaxed text-gray-400">{item.text}</p>
      </Link>)}
    </nav>

    <div className="grid grid-cols-1 items-start gap-6 lg:grid-cols-2">
      <section className="card" aria-labelledby="historical-candidates-title">
        <p className="text-xs font-medium uppercase tracking-wider text-indigo-300">Saved consensus</p>
        <h2 id="historical-candidates-title" className="mt-2 text-lg font-semibold text-white">Historical BUY candidates</h2>
        <p className="mt-2 text-sm leading-relaxed text-gray-400">Ranked by model agreement, then confidence. Open an instrument to inspect its saved analysis.</p>
        {!ready ? <p className="mt-5 text-sm text-gray-400">{loading ? "Loading saved candidates..." : "Candidates are unavailable until the snapshot loads."}</p>
          : !data.top_opportunities?.length ? <p className="mt-5 text-sm text-gray-400">No BUY candidates in this saved snapshot.</p>
          : <ol className="mt-5 space-y-2">{data.top_opportunities.map((item, index) => <li key={item.ticker}>
            <Link to={`/detail/${item.ticker}`} className={`flex items-center gap-3 rounded-xl border border-gray-800 bg-gray-950/40 p-3 transition-colors hover:border-gray-600 ${focus}`}>
              <span aria-hidden="true" className="w-5 text-sm text-gray-500">{index + 1}</span>
              <div className="min-w-0 flex-1"><p className="text-sm font-semibold text-white">{item.ticker}</p><p className="truncate text-xs text-gray-400">{companyName(item.ticker)}</p><p className="mt-1 text-xs text-gray-400">{item.agreement_level} agreement</p></div>
              <div className="text-right"><p className="text-sm font-medium text-emerald-300">BUY</p><p className="mt-1 text-xs text-gray-300">{percent(item.confidence)}</p></div>
            </Link>
          </li>)}</ol>}
        <p className="mt-4 text-xs leading-relaxed text-gray-500">Confidence and agreement are model outputs, not a measured probability of investment success.</p>
      </section>
      <NewsSentiment />
    </div>

    <section className="card" aria-labelledby="saved-headlines-title">
      <div className="flex flex-wrap items-end justify-between gap-3"><div><h2 id="saved-headlines-title" className="text-lg font-semibold text-white">Saved news headlines</h2><p className="mt-1 text-sm text-gray-400">Latest entries in the GDELT archive, with their publication dates.</p></div><span className="text-xs text-gray-400">Separate from model signal dates</span></div>
      {!ready ? <p className="mt-5 text-sm text-gray-400">{loading ? "Loading saved headlines..." : "Headlines could not be loaded."}</p>
        : !data.news?.length ? <p className="mt-5 text-sm text-gray-400">No saved headlines are available.</p>
        : <ul className="mt-5 grid gap-4 md:grid-cols-2">{data.news.map((article, index) => {
          const url = articleUrl(article.url);
          const title = <><p className="text-sm font-medium leading-relaxed text-gray-200">{article.title}</p><p className="mt-2 break-words text-xs text-gray-400">{article.ticker || "Market"} / {article.date ? String(article.date).slice(0, 10) : "Date unavailable"}{article.source ? ` / ${article.source}` : ""}</p></>;
          return <li key={`${article.url}-${index}`} className="rounded-xl border border-gray-800 bg-gray-950/30">{url ? <a href={url} target="_blank" rel="noopener noreferrer" className={`block rounded-xl p-4 hover:bg-gray-800/30 ${focus}`}>{title}<span className="sr-only">Opens in a new tab</span></a> : <div className="p-4">{title}</div>}</li>;
        })}</ul>}
    </section>
    <p className="text-center text-xs text-gray-500">Educational decision support. Historical simulations and new live outcomes are evaluated separately.</p>
  </div>;
}
