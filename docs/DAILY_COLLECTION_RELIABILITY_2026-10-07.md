# Daily collection reliability — 7 October 2026

## Why this work matters

- Forward evaluation needs real forecasts saved before the next market open.
- Missing daily forecasts weaken the evidence, even when historical backtests work.
- This update improves collection reliability and reporting. It does not improve model accuracy by itself.

## What was observed before the update

- The [7 October scheduled run](https://github.com/SyedFaris04/quantsight/actions/runs/37556701339) failed with curl exit code 22: an HTTP error response.
- The [6 October run](https://github.com/SyedFaris04/quantsight/actions/runs/37402735606) was marked successful.
- Full historical job logs require repository administrator access through GitHub's API. The exact HTTP status and root cause of the failed run could not be retrieved.
- Render readiness was healthy at release `89979e8`.
- A read-only live AAPL request returned a real signal dated 6 October, taking about 68 seconds.
- One track-record response was unavailable; two subsequent reads succeeded with 88 forecasts, 44 resolved and 44 pending.
- A real-time admin request returned HTTP 200 with zero inserts and resolutions. It correctly refused new recording after the next open. This checks the locally configured key against Render; it does not verify GitHub's stored secret.
- These observations support retrying temporary failures. They do not prove which failure caused the earlier scheduled run.

## Changes

### Market-data cache

- Concurrent requests wait for one shared download. They do not receive an empty placeholder while another request is fetching data.
- Failed-download caching now starts when the download finishes. Previously, a download lasting over 60 seconds could exhaust the failure timeout before returning and cause repeated downloads inside one job.
- Successful cache entries refresh when the latest completed NYSE session changes.
- Missing sector quotes produce a failed snapshot with a short retry timeout, rather than retaining missing model features for 15 minutes.
- Downloads use eight workers and a 15-second provider request timeout. This timeout is per request, not a total job time limit.
- Complete trained-universe history, sector data, finite inputs and completed-session checks remain required. No missing prices or features are invented.

### Database collection

- Read existing forecasts before inference. Recovery attempts skip forecasts already saved for that session and protocol.
- Existing forecasts are never replaced.
- Resolve due outcomes independently of whether new inference works.
- Read coverage again after the job. A successful insert counter alone does not prove a complete forecast cohort.
- Return forecast date, recording deadline, saved coverage, missing tickers and safe error categories.
- Distinguish resolution, inference, recording and coverage-read failures.
- Reject an empty configured ticker universe.
- Recheck the recording window after inference. Crossing the next open cannot produce a late forecast.

### GitHub scheduler

- Primary attempt: 22:17 UTC, Monday–Friday.
- Recovery attempt: 04:17 UTC, Tuesday–Saturday, for the same completed session.
- In Malaysia these are 06:17 and 12:17 the following day. They are scheduled times, not guaranteed execution times.
- Moving away from the top of the hour follows [GitHub's advice about scheduled workflow load](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule).
- Check backend readiness before calling the collector: up to four checks, separated by 15 seconds.
- Use up to three collection attempts, separated by 75 seconds. This exceeds the backend's 60-second failed-cache timeout.
- Retry network failures and selected temporary HTTP failures. Authentication failures do not retry.
- Require verified complete database coverage and no reported outcome-resolution errors before reporting success.
- An expired window with missing forecasts is a failed job, even if the HTTP response is 200.
- A recovery run with all forecasts already saved can pass without rerunning inference.
- Add an Actions summary with backend release, attempt statuses, forecast date, coverage and safe error codes.
- Never print credentials or raw backend/provider response text. Admin headers do not follow HTTP redirects.
- Keep the workflow's 20-minute job limit. The runner needs only Python's standard library.

## Setup and model impact

- Uses the existing `RENDER_BACKEND_URL` and `ADMIN_JOB_SECRET` GitHub secrets.
- No new API key, database table, SQL migration or manual setup is needed.
- Saved models, calibrators, feature formulas and decision thresholds are unchanged.
- Editing live inference code changes the existing feature-code hash. Future forecasts will therefore have a new implementation version, even though the feature formulas are unchanged. Forward reports retain this distinction rather than mixing versions silently.
- Missed dates remain missing. No retrospective forecasts are created.

## Verification

- 189 backend tests passed in the serving environment and in the separate clean CI environment.
- New tests cover cold starts, retry exhaustion, authentication failures, malformed responses, safe reporting, incomplete coverage, late recording, database verification failures, missing sectors, cache session changes and concurrent downloads.
- A fixture with a 90-second failed fetch verifies that a second request does not trigger another download immediately.
- Runtime verification passed, including saved API responses and verified XAI evidence. No heavy training dependencies were added to serving.
- The restarted local backend is ready with 44 tickers. Its real-time admin response verifies zero saved forecasts for 6 October and skips all 44 after the 7 October opening deadline; it does not backfill that missed date.
- Hosted observations above precede this release. Check the deployed release and the next eligible scheduled collection separately.

## What still needs real observation

- Confirm the next eligible scheduled run actually saves all 44 forecasts before the next open.
- Check its GitHub Actions summary and the Track Record page for the same forecast date and complete coverage.
- Check exact fifth-session resolutions as their target dates arrive.
- More reliable collection does not establish predictive skill. The current resolved evidence is still only one forecast date.
- GitHub scheduling and the external price provider can still fail or be delayed. The recovery run reduces risk; it cannot guarantee collection.

The provider settings follow the [official yfinance download reference](https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html).
