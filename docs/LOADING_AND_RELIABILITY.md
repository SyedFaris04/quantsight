# Loading and request reliability — 4 October 2026

This release reduces the initial frontend download and improves recovery from
slow or failed requests. It does not change trained models or prediction results.

## Measured production-build change

Same local build configuration, before and after route splitting:

| Measure | Before | After | Reduction |
|---|---:|---:|---:|
| Initial JavaScript bytes | 1,063,230 | 455,013 | 57.2% |
| Initial JavaScript, gzip level 9 | 293,513 | 135,341 | 53.9% |

- These are file-size measurements, not a claim of 57% faster real-world page loads.
  CDN compression and deployment configuration can produce slightly different sizes.
- A fresh browser visit to the built homepage requests one JavaScript entry file;
  chart and other page modules are deferred until required.
- Dashboard, navigation, authentication and chat remain in the initial application.
  Ten other pages use React lazy loading. Shared chart code is cached across pages.
- The largest generated chunk is below 500 kB; Vite's previous large-chunk warning
  is no longer emitted.
- Total JavaScript across all files is 1,068,564 bytes. Opening every page eventually
  loads additional chunks; the improvement comes from avoiding unnecessary initial work.

## Recovery behavior

- A page download displays a loading indicator while navigation remains available.
- If a page module fails to download or render, an error boundary offers an explicit
  reload. Navigation, authentication and chat are outside that boundary.
- Reload is deliberate: an old deployment's removed module or a cached rejected import
  may need a fresh document. There is no automatic reload loop.
- Chat stays mounted during navigation; an unsent draft survives changing pages.
- Every shared-hook GET, including a manual retry, cancels and supersedes the previous
  request. A generation check prevents late completion/error/finally handlers from
  replacing newer data or clearing newer loading indicators.
- Endpoint changes hide the previous ticker's data immediately. Disabled endpoints
  clear their state. Old page callbacks cannot restart requests after their effect
  has been cleaned up.
- The existing GET deadline remains 30 seconds. Dashboard shows a slow-request notice
  after eight seconds; timeout, network, service-unavailable and rate-limit errors
  have readable messages. Manual retry remains available.
- No automatic retries were added. In particular, the client does not repeatedly
  retry rate limits or automatically replay POST requests.
- Browser cancellation does not guarantee that the backend stops work already started.

## Verification

- Seven frontend regression tests cover late responses, manual retry races, cancellation,
  disabled endpoints, slow timers, bounded request counts and error formatting.
- `npm test` now runs in the existing GitHub frontend job alongside the production build.
  No new JavaScript dependencies were added.
- Chromium checked every route in the production build, including direct Stock Detail
  and Login URLs, with real data from a local backend.
- The homepage did not request chart/page chunks before navigation.
- Browser fault checks verified a delayed page download, a failed download, continued
  navigation, explicit reload recovery, an actual 30-second Axios timeout and successful retry.
- Chat draft persistence, 390/320-pixel layouts and the absence of uncaught errors during
  normal navigation were checked. Error and mobile screenshots were inspected.
- An additional warm local API sample returned Dashboard in 0.036 seconds, Market's
  `/overview` in 0.021 seconds and the development report in 0.033 seconds. These single
  observations are diagnostics, not production latency benchmarks.

Local measurements and screenshots are in `backend/data/research/loading_performance/`
(excluded from Git). The browser check forwards only GET requests to the local backend;
it does not submit chat messages, portfolio changes or quiz answers.

To reproduce with the existing developer API origin `http://127.0.0.1:8001`:

1. Start the local backend on port 8000.
2. In `frontend`, run `npm test`, `npm run build`, then
   `node node_modules/vite/bin/vite.js preview --host 127.0.0.1 --port 4173 --strictPort`.
3. In `backend`, run `.venv/Scripts/python.exe -m research.verify_loading_ui`.
   The verification browser forwards the build's port-8001 GETs to port 8000.
   Playwright and a compatible Chromium installation are required for this manual QA.

## Remaining backend work

The subsequent [backend reliability release](BACKEND_RELIABILITY.md) adds serving
dependency separation, startup measurements and accurate saved-data readiness.
Steady-state memory and production hosting diagnostics remain follow-up work.

The frontend improvements do not eliminate hosting startup, external-provider or
network delays. Separate backend work should measure startup and representative
production latency, audit which training dependencies are actually needed for serving,
and add readiness/release diagnostics before changing deployment configuration.

No new API keys, environment variables or Supabase migration are required for this release.

Implementation references: [React lazy loading](https://react.dev/reference/react/lazy)
and [Axios request cancellation](https://axios-http.com/docs/cancellation).
