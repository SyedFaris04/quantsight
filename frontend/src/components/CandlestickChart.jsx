/**
 * frontend/src/components/CandlestickChart.jsx
 * Real OHLCV candlestick chart. Recharts has no built-in candlestick type,
 * so this draws one using the well-known "range Bar + custom shape" recipe:
 * a Bar's dataKey returns [Low, High] (so Recharts maps that range onto the
 * y-axis for us), and the custom shape function linearly interpolates the
 * Open/Close pixel positions within that same [y, y+height] span to draw
 * the candle body on top of the high-low wick. A Volume bar panel sits
 * underneath sharing the same x-axis. No new dependency, no fabricated data
 * — every candle comes directly from the real Open/High/Low/Close/Volume
 * columns already in features_finance.csv.
 */

import {
  ComposedChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, BarChart, Cell,
} from "recharts";

const UP_COLOR   = "#22c55e";
const DOWN_COLOR = "#ef4444";

function CandleShape(props) {
  const { x, y, width, height, payload } = props;
  const { Open, Close, High, Low } = payload;
  if ([Open, Close, High, Low].some(v => v == null)) return null;

  const isUp = Close >= Open;
  const color = isUp ? UP_COLOR : DOWN_COLOR;
  const range = High - Low || 1;

  // Linear interpolation within this bar's own drawn pixel span [y, y+height],
  // which Recharts has already mapped from the [Low, High] data range.
  const pxFor = (v) => y + height * (High - v) / range;

  const bodyTop    = pxFor(Math.max(Open, Close));
  const bodyBottom = pxFor(Math.min(Open, Close));
  const bodyHeight = Math.max(bodyBottom - bodyTop, 1);
  const wickX      = x + width / 2;
  const bodyWidth  = Math.max(width * 0.6, 2);
  const bodyX      = x + (width - bodyWidth) / 2;

  return (
    <g>
      {/* High-low wick */}
      <line x1={wickX} y1={y} x2={wickX} y2={y + height} stroke={color} strokeWidth={1} />
      {/* Open-close body */}
      <rect x={bodyX} y={bodyTop} width={bodyWidth} height={bodyHeight} fill={color} />
    </g>
  );
}

function CandleTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload;
  if (!d) return null;
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-3 text-xs shadow-xl min-w-36">
      <p className="text-gray-400 mb-2 font-medium">{label}</p>
      <p className="text-gray-300">Open:  <span className="text-white font-mono">${d.Open?.toFixed(2)}</span></p>
      <p className="text-gray-300">High:  <span className="text-white font-mono">${d.High?.toFixed(2)}</span></p>
      <p className="text-gray-300">Low:   <span className="text-white font-mono">${d.Low?.toFixed(2)}</span></p>
      <p className="text-gray-300">Close: <span className="text-white font-mono">${d.Close?.toFixed(2)}</span></p>
      {d.Volume != null && (
        <p className="text-gray-300 mt-1">Vol: <span className="text-white font-mono">{Number(d.Volume).toLocaleString()}</span></p>
      )}
    </div>
  );
}

export default function CandlestickChart({ data, yDomain }) {
  return (
    <div>
      <ResponsiveContainer width="100%" height={220}>
        <ComposedChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="date"
            tick={{ fill: "#6b7280", fontSize: 10 }}
            tickLine={false}
            axisLine={{ stroke: "#374151" }}
            interval={Math.max(Math.floor((data?.length || 1) / 6), 0)}
            tickFormatter={d => d?.slice(5)}
          />
          <YAxis
            domain={yDomain}
            tick={{ fill: "#6b7280", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
            tickFormatter={v => `$${v.toFixed(0)}`}
            width={55}
          />
          <Tooltip content={<CandleTooltip />} />
          <Bar dataKey={(d) => [d.Low, d.High]} shape={<CandleShape />} isAnimationActive={false} />
        </ComposedChart>
      </ResponsiveContainer>

      {/* Volume panel — same x-domain, real Volume column */}
      <ResponsiveContainer width="100%" height={56}>
        <BarChart data={data} margin={{ top: 0, right: 10, left: 0, bottom: 0 }}>
          <XAxis dataKey="date" hide />
          <YAxis hide />
          <Bar dataKey="Volume" isAnimationActive={false}>
            {data?.map((d, i) => (
              <Cell key={i} fill={d.Close >= d.Open ? UP_COLOR : DOWN_COLOR} fillOpacity={0.5} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
