# Backtesting interface verification - 27 September 2026

The interface now labels return differences in percentage points (`pp`), includes
SPY reference values beside risk metrics, explains the main metrics, and stacks
the headline cards on narrow screens. Strategy tables support keyboard scrolling
and expose the selected strategy through pressed-state accessibility metadata.
No backtest accounting, model predictions or published numerical results changed.

The optional browser check uses the actual backend report and real downloads:

- All 15 strategy selections display the API's return difference with the correct
  unit. Table selection updates the selector; benchmark selection hides model-only
  classification results.
- All four downloads work. Metrics include all strategies, daily values contain
  462 sessions and end at the reported equity, trades match the execution count,
  and the JSON export identifies the displayed run.
- Methodology and metric explanations open correctly. A keyboard can scroll the
  comparison table. Screens at 390px and 320px have no page-wide horizontal overflow.
- Browser-only intercepted responses exercise unavailable data, recovery through
  Retry, an unpublished report, stale-source warnings and download failure.
- No uncaught page errors. Desktop/mobile screenshots were visually inspected.
- The production frontend build passes. Its existing large bundle warning remains.

Run from `backend` with the backend/frontend running and Playwright/Chromium installed:

```powershell
.\.venv\Scripts\python.exe -m research.verify_backtesting_ui --browser C:/path/to/chrome.exe
```

For deployment verification add `--base-url https://quantsight.vercel.app` and a
separate `--output data/research/production_backtest_ui` directory. These optional
tools are not runtime dependencies. Downloaded evidence, screenshots and timestamped
verification JSON are saved under ignored `backend/data/research/`.

These checks establish interface behavior and consistency with the saved report;
they do not establish future profitability or remove historical selection bias.
