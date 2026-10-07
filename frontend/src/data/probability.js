// Legacy API "confidence" is already calibrated P(UP), also for SELL.
export function savedProbabilityUp(model) {
  const value = model?.confidence;
  return typeof value === "number" &&
    Number.isFinite(value) &&
    value >= 0 &&
    value <= 100
    ? value
    : null;
}
export function probabilityDifference(finance, sentiment) {
  const a = savedProbabilityUp(finance),
    b = savedProbabilityUp(sentiment);
  return a == null || b == null ? null : b - a;
}
