import test from "node:test";
import assert from "node:assert/strict";
import { parseTimeline, TIMELINES } from "../src/lib/timeline.js";

test("a shared link keeps a valid timeline", () => {
  assert.equal(parseTimeline("?timeline=2005"), "2005");
  assert.equal(parseTimeline("?timeline=2015"), "2015");
});
test("unknown, empty or malformed values fall back to 2015", () => {
  for (const search of ["", "?timeline=1999", "?timeline=", "?other=1", "?timeline=2005x"])
    assert.equal(parseTimeline(search), "2015");
});
test("the switch offers both timelines, 2015 first", () => {
  assert.deepEqual(TIMELINES, ["2015", "2005"]);
});
