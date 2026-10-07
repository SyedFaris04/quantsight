import test from "node:test";
import assert from "node:assert/strict";
import {
  savedProbabilityUp,
  probabilityDifference,
} from "../src/data/probability.js";
test("SELL probability is not complemented; calibration can cross the raw vote", () => {
  assert.equal(savedProbabilityUp({ signal: 0, confidence: 55.19 }), 55.19);
  assert.equal(
    probabilityDifference(
      { signal: 0, confidence: 55.19 },
      { signal: 1, confidence: 61.2 },
    ),
    61.2 - 55.19,
  );
});
test("unknown or invalid probabilities remain unknown", () => {
  for (const confidence of [undefined, null, NaN, Infinity, -1, 101, "50"])
    assert.equal(savedProbabilityUp({ confidence }), null);
  assert.equal(probabilityDifference({}, { confidence: 60 }), null);
  assert.equal(savedProbabilityUp({ confidence: 0 }), 0);
});
