# Supervisor demonstration checks — 27 September 2026

- Purpose: check the deployed demonstration and a local backup for Monday, 28 September.
- Application revision checked: `82c74b4`; accompanying changes are documentation only.
- Checked in the evening, Malaysia time. Timings are observations from this laptop/network, not performance guarantees.

## Deployed checks

- Vercel homepage: HTTP 200, about 1.32 seconds.
- Render `/dashboard`: HTTP 200, about 23.58 seconds; historical snapshot dated 20 December 2024, four models.
- Render `/backtest`: HTTP 200, about 16.09 seconds; saved report available, run `20260926T102301904516Z`.
- Render `/research/news-comparison`: HTTP 200, about 0.87 seconds; report available, run `0024c8e43e438b37`.
- Render `/live-track-record`: HTTP 200, about 12.77 seconds; v2 available, 0 logged, 0 pending and 0 resolved forecasts.
- These were read-only HTTP checks. Prior releases also have desktop/mobile browser verification documented in the project.

## Local checks

- Started FastAPI at `http://127.0.0.1:8001` and Vite at `http://127.0.0.1:3000`.
- Explicitly set the frontend process's `VITE_API_URL` to the local backend.
- Backend loaded four historical prediction files, 104,076 finance feature rows and the saved news file; startup completed with 44 instruments.
- Root API, Dashboard, Backtesting and research API: all HTTP 200; observed responses about 0.6–2.3 seconds.
- Chromium browser check: blocked all browser HTTP requests except localhost/127.0.0.1.
- Dashboard: actual historical snapshot and date displayed.
- Backtesting: ensemble strategy loaded and the full report JSON downloaded successfully.
- AI Compare: research section and constant training-prior baseline displayed.
- No uncaught JavaScript page errors in these checks.
- The external font request was blocked; the pages remained usable.
- Browser page checks, including loading/rendering/screenshot work: initial Dashboard about 23.7 seconds; Backtesting 2.58 seconds; AI Compare 1.08 seconds. Start and visit the local pages before the meeting too.
- This browser restriction did not disable the backend machine's internet. The checked core endpoints serve saved artifacts; this is not an end-to-end offline certification of every feature.
- Live prices, authentication/account synchronization, Supabase tracking and Groq chat remain internet-dependent.
- Local screenshots, the downloaded report and machine-readable check results are saved in the ignored directory `backend/data/research/supervisor_demo_20260928/`.

## Scope and repeatability

- No forecasts, trades, account changes or model training were triggered by these checks.
- No new database migration or dependency change is required for these notes.
- Existing test-suite results are reported separately in the progress document; a documentation update does not rerun or improve predictive performance.
- Use [the exact startup commands and meeting checklist](../SUPERVISOR_PROGRESS.md) to repeat the local demonstration.
