# Dashboard dates and interface - 27 September 2026

The previous Dashboard described saved predictions as "today" and "Live signals",
and a loaded news file as real-time news. The redesigned page shows those sources
as historical research inputs, with direct paths to Backtesting, AI Compare and
forward Track Record. It adds a consistent layout, instrument names, accessible
daily sentiment values, independent retries and explicit missing-data states.

`/dashboard` retains its existing keys and adds `snapshot` metadata: earliest/latest
signal date, mixed dates, undated signals, model count and five-session horizon.
The current saved prediction snapshot is dated 20 December 2024. Consensus and
confidence are labelled as model outputs; they do not measure future profitability.

`/market-sentiment` retains percentage/count/trend keys and adds its saved-archive
source, scoring-window dates, latest article timestamp, age in elapsed UTC days,
and excluded-row count. The scoring window remains anchored to the latest usable
saved article so the archive can be studied. It is explicitly not a rolling live
news service. Current archive end: 23 April 2026. Invalid/future dates and empty
titles are excluded; repeated titles are counted once at their latest date.
Headline validity, language coverage, ticker relevance and VADER accuracy require
separate evaluation; date provenance does not establish those properties.

The sidebar no longer displays a permanent green "Live" status. Chatbot tool
descriptions and source-date instructions match the historical inputs. The data
tools supply dates, but generated chatbot wording has not been evaluated in this change.

Validation: 63 Python tests, including seven new date/filtering tests; production
frontend build; local Chromium checks of actual API dates/counts, candidate/evidence
links, daily values, 390px/320px layout, independent request failures and retries,
old-backend metadata absence and empty news. No uncaught page errors. Desktop and
mobile screenshots inspected. The existing frontend bundle-size warning remains.

Optional browser reproduction from `backend` (Playwright plus Chromium required):

```powershell
.\.venv\Scripts\python.exe -m research.verify_dashboard_ui --browser C:/path/to/chrome.exe
```

Add `--base-url https://quantsight.vercel.app` to verify deployment. Screenshots and
timestamped results stay in ignored `data/research/dashboard_ui`. No model retraining,
database migration or external account mutation is needed for this change.
