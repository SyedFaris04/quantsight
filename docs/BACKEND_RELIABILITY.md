# Backend startup and readiness — 4 October 2026

## What changed

- Render's existing `pip install -r requirements.txt` command now installs the
  serving packages, with direct versions matching the tested Python 3.12 environment.
- PyTorch, Transformers, datasets, SHAP and GDELT collection packages are offline
  tools. The API serves saved LSTM predictions and attention; it runs XGBoost for
  live predictions and native TreeSHAP. Serving does not require those offline packages.
- Offline developers can install `requirements-training.txt`, then install the
  appropriate PyTorch build separately. The pinned research environments remain available.
- XGBoost and sklearn training imports are deferred until needed. The first live
  prediction or TreeSHAP request still pays the cost of loading its model libraries.
- Startup logs report module-import time and separate prediction, feature and news
  loading times. These measurements exclude Render scheduling, builds and wakeup.
- Required prediction and price datasets are checked before startup succeeds.
  Missing models, invalid probabilities, duplicate keys and invalid price data
  prevent a misleading healthy deployment. LSTM exports can omit actual labels;
  the existing track-record code derives those labels from feature data.
- Missing or corrupt optional news yields an empty news archive and a server log;
  saved stock data remains available.

## Health endpoints

| Endpoint | Meaning |
|---|---|
| `/` and `/health/ready` | HTTP 200 after required saved data loads and validates; otherwise 503 if reachable |
| `/health/live` | The API process can respond; it does not validate datasets |

Readiness includes loaded row counts, startup durations and the deployed commit
from Render's `RENDER_GIT_COMMIT`. Local runs report `unknown` without that variable.
No credentials or local paths are included. Responses use `Cache-Control: no-store`.
External Yahoo, Groq and Supabase availability is explicitly marked `not_checked`;
an outage in those services should not restart the saved-data API.

Uvicorn completes startup before serving requests. Invalid required datasets cause
startup to fail, so production may refuse connections rather than return HTTP 503.
Readiness describes the data cached at startup, not later disk changes.

The root URL now provides the same readiness check, so an existing Render HTTP
check on `/` benefits automatically. If the service uses Render's default TCP
check, an operator can set **Settings → Health Checks → Path** to `/health/ready`.
This dashboard setting has not been changed from this workspace.
[Render documents how HTTP checks gate deployments](https://render.com/docs/health-checks).

## Verification and measurements

- Backend regression suite: 109 tests, including six readiness contract tests.
- `python -m research.verify_runtime` checks the actual FastAPI startup, Dashboard,
  Market, saved AAPL history, explanations, comparison, research report and both
  XGBoost attribution variants. It also checks unstarted readiness, failed required
  data, optional-news failure and cache headers. It makes no external-provider
  requests or database writes.
- Six saved endpoint payloads had identical SHA-256 hashes before and after this
  change in the same local environment.
- A clean Linux GitHub Actions job installs only serving dependencies, runs
  `pip check`, and executes that same real-asset smoke check.
- Local Windows samples: previous warm import 2.791 s and initial resident memory
  274.6 MiB; updated import 1.880 s and initial resident memory 210.0 MiB.
  One earlier import took 25.298 s, demonstrating variation. These are diagnostic
  samples, not a controlled cold-start benchmark or a claim about Render speed.
- After explanations and both XAI models loaded, resident memory was about
  565 MiB before and after. Deferring imports reduces initial work; it does not
  eliminate the full application's later memory needs.

## Deployment and local setup

No new API keys, Supabase migrations or frontend configuration are required.
Keep Render's build command `pip install -r requirements.txt` and start command
`uvicorn main:app --host 0.0.0.0 --port $PORT` with root directory `backend`.
Use Python 3.12, as used in the clean serving check.

For a local API run from `backend`:

```powershell
.venv/Scripts/python.exe -m pip install -r requirements.txt
.venv/Scripts/python.exe -m uvicorn main:app --port 8000
```

A reduced requirements file does not uninstall old extras from an existing venv
or cached deployment environment. A fresh build is what verifies the smaller package set.

If the service is on Render's Free plan, idle spin-down still introduces wakeup
delay; this release cannot remove that platform behavior.
[Render's Free-plan behavior](https://render.com/docs/free).

## Next improvement

The copilot maintains separate full CSV and attention caches. Measure and reduce
their steady-state memory next, while preserving historical explanations and
prediction results. Render's actual memory/CPU graphs and build logs are needed
to determine whether its instance size or build environment contributes to delays.
