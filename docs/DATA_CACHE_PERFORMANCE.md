# Dashboard and explanation caches — 4 October 2026

- The API builds a small latest-prediction index once during startup. Dashboard
  no longer filters and sorts each full prediction table for all 44 stocks.
- Copilot keeps one latest feature row per stock, with all its original columns.
  Finance rows reuse the table already loaded and validated by the API. Original
  CSV date strings are preserved. Sentiment's full table is released after its
  latest rows are selected.
- Historical label joins retain only ticker, date and signal. Timeline reasons
  retain eight relevant indicator columns, including ticker/date, and reuse the
  data for subsequent stocks instead of reading the CSV again.
- A lock prevents concurrent requests from loading the same copilot cache twice.
  Returned latest rows are copies, so a caller cannot alter the cached result.
- Caches reset on application startup. Saved files stay fixed during a process;
  changing those files requires a restart. Live Yahoo signals keep their existing
  separate cache behavior.

## Verification

- All 44 complete stock explanations match their pre-change response hashes.
- Dashboard, overview, comparison, two timelines and two accuracy track records
  also match their pre-change response hashes.
- Four new regression tests cover concurrent reads, preserved latest rows,
  narrow history columns, original date strings and cache reset.
- Total backend regression suite: 113 passing tests.
- The Linux serving smoke check now includes timeline/accuracy endpoints and
  checks all 88 compact finance/sentiment latest rows against full CSV lookups,
  with exact values. Both native XGBoost explanations still reproduce predictions.

## Local diagnostic samples

The same sequence loads Dashboard, AAPL explanation/comparison, AAPL and MSFT
timelines, AAPL and NVDA accuracy, then explanations for all 44 stocks.

| Measure | Before | After |
|---|---:|---:|
| Copilot dataframe caches | 98.55 MiB | 16.81 MiB |
| Process resident memory after this sequence | 375.3 MiB | 303.4 MiB |
| First AAPL explanation | 1.245 s | 0.815 s |
| First AAPL timeline | 0.924 s | 0.491 s |
| Subsequent MSFT timeline | 1.138 s | 0.006 s |

These are single local Windows samples, not controlled benchmarks or promises
of equivalent Render speed. Cache contents shrink by about 83%; that percentage
does not describe total server memory. Model libraries, saved news, attention
JSONs and the API's main finance panel still occupy memory.

JSON measurements and pre/post hashes are stored locally in
`backend/data/research/cache_performance/` (excluded from Git).

## Deployment

No new keys, dependencies or Supabase migrations are needed. GitHub's existing
deployment connections publish this backend change. The frontend API contract
and saved prediction values are preserved.

First sentiment access still parses its CSV once, and native TreeSHAP still
loads model libraries and exact historical inputs on its first request. A next
step is to prepare and validate compact serving artifacts for these fixed inputs,
with clear invalidation when models or data change.

The subsequent [compact serving snapshot release](SERVING_SNAPSHOTS.md) removes
the initial full sentiment CSV parse for latest-feature lookups. TreeSHAP's
initial input scan remains follow-up work.
