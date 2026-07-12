/**
 * frontend/src/components/RadialProgress.jsx
 * SVG circular progress ring — shows a real confidence/score percentage
 * as a ring around a center label. No dependency, pure SVG stroke-dasharray.
 */

export default function RadialProgress({ value, size = 88, strokeWidth = 7, color = "#6366f1", label }) {
  const pct = Math.min(Math.max(value ?? 0, 0), 100);
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (pct / 100) * circumference;

  return (
    <div className="relative inline-flex items-center justify-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke="#1f2937" strokeWidth={strokeWidth}
        />
        <circle
          cx={size / 2} cy={size / 2} r={radius}
          fill="none" stroke={color} strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          style={{ transition: "stroke-dashoffset 0.6s ease" }}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-lg font-bold text-white leading-none">{pct.toFixed(0)}%</span>
        {label && <span className="text-[10px] text-gray-500 mt-1">{label}</span>}
      </div>
    </div>
  );
}
