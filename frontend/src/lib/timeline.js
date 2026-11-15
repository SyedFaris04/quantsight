// Training-history versions of the served models; the dashboard switch shows one.
export const TIMELINES = ["2015", "2005"];

export function parseTimeline(search) {
  const value = new URLSearchParams(search).get("timeline");
  return TIMELINES.includes(value) ? value : TIMELINES[0];
}
