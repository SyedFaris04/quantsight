// Quote change is relative to previous close, not current value.
export function dailyQuoteProfit(currentValue, changePercent, isLive) {
  if (
    !isLive ||
    !Number.isFinite(currentValue) ||
    !Number.isFinite(changePercent) ||
    changePercent <= -100
  )
    return null;
  const ratio = changePercent / 100;
  return (currentValue * ratio) / (1 + ratio);
}
