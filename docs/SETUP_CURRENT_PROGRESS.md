# Set up the current QuantSight version

Prepared for this existing project on 27 September 2026. Use your existing
Supabase, GitHub, Render and Vercel projects; there is no need to create replacements.

## What is already done and what remains

| Component | Current status | Required next action |
|---|---|---|
| Local Python environment, frontend packages, model files | Present | Start the local servers when needed |
| Corrected historical backtest and exports | Generated locally | Include the saved report and exports in the release |
| New live tracker and interface code | Implemented locally | Publish the reviewed code and deploy |
| New Supabase table | User applied migration 002; server/public reads pass | Verify hosted writes with genuine forecasts |
| Render/Vercel services | Existing site works, but serves older code | Verify settings and deploy the new commit |
| Local environment variables | Required keys are populated | Verify corresponding hosted values; no new keys needed if valid |
| GitHub scheduler | Updated workflow is local | Verify secrets, publish workflow, then test |
| Model/news research | Archive and pilot results are saved locally | No dataset download or retraining needed for setup |

Local checks passed: 47 Python tests and 11 PostgreSQL migration tests; frontend
production build passed previously. These do not establish hosted deployment success.
Private dashboard settings have not been inspected. A populated local key does not
prove the hosted service uses that same value.

## 1. Supabase: create the new live tracking table

1. Open [your project's SQL Editor](https://supabase.com/dashboard/project/wlsphzugowaaqljqjuvz/sql/new)
   and sign in.
2. Open `backend/supabase/002_live_predictions_v2.sql` in VS Code or a text editor.
   Its full local path is
   `C:\Users\syeda\OneDrive\Desktop\quantsightv2\backend\supabase\002_live_predictions_v2.sql`.
3. Copy the entire file, from `begin;` through `commit;` (the opening comments can
   be copied too), into a new SQL query. Give it a name such as `QuantSight live tracker v2`.
4. Click **Run**. A successful schema operation may say **Success. No rows returned**;
   it does not need to display prediction records.
5. In **Table Editor**, refresh and find `public.live_predictions_v2`. It should
   initially be empty. The existing `live_predictions` table should still exist.

This creates storage, constraints, access rules and triggers. It does not train
models, deploy code or copy legacy predictions into the new evaluation.
[Supabase documents SQL Editor as a database management method](https://supabase.com/docs/guides/database/overview).

Tell the assistant whether it succeeds. For an error, share the error text rather
than API keys. The available workspace has no signed-in dashboard or database DDL
connection, which is why this account step needs your involvement.

## 2. Check Render's existing backend configuration

Open [Render Dashboard](https://dashboard.render.com), select the service whose
address is `https://quantsight-rikc.onrender.com`, then inspect **Settings**.

Expected configuration for this repository:

| Setting | Expected value |
|---|---|
| Repository | `SyedFaris04/quantsight` |
| Branch | `main` |
| Root Directory | `backend` |
| Build Command | `pip install -r requirements.txt` |
| Start Command | `uvicorn main:app --host 0.0.0.0 --port $PORT` |

If the existing service instead uses the repository root with commands beginning
`cd backend`, that can be equivalent; do not combine both directory conventions.
The local tested Python version is 3.12.9. Check the deployed Python version before
changing it; environment locking and model-library compatibility remain release checks.
These commands follow [Render's FastAPI setup](https://render.com/docs/deploy-fastapi).

Under **Environment**, verify the following names. Matching values already exist
in your local `backend/.env`, so reuse valid existing values instead of generating
different ones for each service.

| Variable | Value/source |
|---|---|
| `SUPABASE_URL` | `https://wlsphzugowaaqljqjuvz.supabase.co` |
| `SUPABASE_SERVICE_ROLE_KEY` | Existing backend service-role key |
| `ADMIN_JOB_SECRET` | Existing backend secret; must match GitHub Actions |
| `GROQ_API_KEY` | Existing Groq key, needed for AI chat |

Use **Save only** if you are configuring values before publishing the new code.
The next deployment will load them. Do not paste secret values into chat or put
the service-role key in Vercel's frontend variables.
[Render environment-variable instructions](https://render.com/docs/configure-environment-variables).

## 3. Check Vercel's existing frontend configuration

Open [Vercel Dashboard](https://vercel.com/dashboard), select the project serving
`quantsight.vercel.app`, then inspect its project settings.

Expected build configuration: repository `SyedFaris04/quantsight`, production
branch `main`, root directory `frontend`, framework **Vite**, build command
`npm run build`, output directory `dist`. The repository's `frontend/vercel.json`
already provides the single-page application route fallback.

Under **Settings > Environment Variables**, check the **Production** environment:

| Variable | Value/source |
|---|---|
| `VITE_API_URL` | `https://quantsight-rikc.onrender.com` |
| `VITE_SUPABASE_URL` | `https://wlsphzugowaaqljqjuvz.supabase.co` |
| `VITE_SUPABASE_ANON_KEY` | Existing public anon key from `frontend/.env` |

The local API URL is `http://127.0.0.1:8001`; keep that for local development.
Vercel must use the Render URL. If you use preview deployments, configure their
variables separately too. Changes require a new build/deployment before the
browser bundle uses them. [Vercel environment-variable documentation](https://vercel.com/docs/environment-variables).

## 4. Check GitHub's scheduler secrets

Open [repository Actions secrets](https://github.com/SyedFaris04/quantsight/settings/secrets/actions),
or navigate **Settings > Secrets and variables > Actions**.
Under repository secrets, add or update:

| Secret | Value |
|---|---|
| `RENDER_BACKEND_URL` | `https://quantsight-rikc.onrender.com` |
| `ADMIN_JOB_SECRET` | Exactly the same value as Render's `ADMIN_JOB_SECRET` |

These are secrets, not ordinary repository variables. GitHub hides saved secret
values; if you cannot confirm a match, set the existing local value in both
locations. [GitHub secret setup](https://docs.github.com/en/actions/how-tos/write-workflows/choose-what-workflows-do/use-secrets).

## 5. Publish and deploy the reviewed changes

Saving files locally does not send them to GitHub or update the website. The
workspace has many modified/untracked files, including earlier model work.
The next release should be assembled and checked as one coherent commit.

The assistant can prepare the local release. It needs the actual application
changes, migration, workflows, serving model/data artifacts and saved backtest
report with its referenced export directory. Research article downloads and
`backend/.env` stay outside the release. Local frontend configuration must not
replace Vercel's production API URL.

If reviewing in VS Code, open **Source Control**, inspect the files/diffs selected
for the release, then commit the reviewed selection. Push that commit to the
configured deployment branch. Do not delete existing user changes to simplify this.
Check the **Evaluation and frontend checks** workflow for Python, database and
frontend build results. A push may also trigger deployment if auto-deploy is enabled.

On Render, verify the deployed commit ID and build log. If auto-deploy is disabled,
use **Manual Deploy > Deploy latest commit** after the commit is on GitHub.
On Vercel, verify the production deployment uses that commit and the intended
environment variables. Redeploy the correct commit if necessary; redeploying an
old commit cannot include local changes.

The updated backend must be available before relying on the new frontend's endpoints.
No new subscription, dataset service or retraining job is required by these changes.
Actual resource/build issues, if any, must be diagnosed from deployment logs.

## 6. Verify the deployed version before running the scheduler

Open these addresses:

- `https://quantsight-rikc.onrender.com/` - API health should respond.
- `https://quantsight-rikc.onrender.com/backtest` - should return `available: true`
  and a report whose `freshness.matches_current_files` is true.
- `https://quantsight-rikc.onrender.com/live-track-record` - should return
  `available: true`, `protocol_version: "nyse-close-5-v2"`, `horizon_sessions: 5`
  and `legacy_excluded: true`. Empty counts are normal before new records exist.
- `https://quantsight.vercel.app/backtesting` - should display the new historical report.

The assistant can run the read-only checker for you. To run it yourself, open
PowerShell in the project directory:

```powershell
Set-Location 'C:\Users\syeda\OneDrive\Desktop\quantsightv2'
.\backend\.venv\Scripts\python.exe backend/check_live_rollout.py --backend-url https://quantsight-rikc.onrender.com
```

Expect `all_requested_checks_passed: true` once all checked components are deployed.
This tests read access and response contracts; hosted write protections and an actual
scheduled forecast/outcome still require verification. If only the migration is done,
the backend checks can still fail until the code deployment is complete.

Then open [GitHub Actions](https://github.com/SyedFaris04/quantsight/actions), choose
**Daily live prediction tracker**, select **Run workflow**, choose `main` and run it.
The workflow must exist on the default branch for manual dispatch.
[GitHub manual-run instructions](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/manually-run-a-workflow).

This is a real recording job, not a simulation: during an eligible recording window
it stores current forecasts. It may validly skip outside that window. A weekend can
still be inside Friday-close-to-Monday-open eligibility. Repeated runs should count
duplicates rather than replacing previous forecasts. No accuracy is available until
the fifth-session outcomes mature.

The schedule is 22:00 UTC Monday-Friday, which is **06:00 Malaysia time Tuesday-Saturday**.
Inspect the run log for success, skips or errors; a green workflow alone does not
prove that forecasts were inserted.

## Local demonstration for Monday

No reinstall, dataset download or retraining is necessary on this computer.
The prepared local frontend uses API port **8001**. Open two PowerShell terminals.

Backend terminal:

```powershell
Set-Location 'C:\Users\syeda\OneDrive\Desktop\quantsightv2\backend'
.\.venv\Scripts\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001
```

Frontend terminal:

```powershell
Set-Location 'C:\Users\syeda\OneDrive\Desktop\quantsightv2\frontend'
npm run dev
```

Keep both running. Open `http://localhost:3000/backtesting`; API documentation is
at `http://127.0.0.1:8001/docs`. If a port is already occupied, inspect the existing
server rather than launching another copy. The historical backtest works without
the new Supabase table, but new live tracking needs that migration.

## Common setup results

| Result | Likely next check |
|---|---|
| SQL succeeds but new table is empty | Expected until the updated tracker records genuine forecasts |
| `PGRST205` for v2 | Confirm correct project/table and API schema refresh |
| `/backtest` returns 404 | Render still serves old code or wrong service |
| Tracker response lacks `protocol_version` | Old backend is still deployed |
| Daily job returns 403 | GitHub and Render admin secrets differ or are missing |
| Daily job returns 503 | Inspect migration, database connection or per-ticker errors |
| Website calls `127.0.0.1` | Correct Vercel's production `VITE_API_URL` and rebuild |
| Frontend loads but API fails locally | Confirm backend runs on 8001 and restart Vite after env edits |

Existing login/portfolio/game database setup is separate. It has not been recreated
by these improvements; broader account isolation and game correctness work remains
on the roadmap. Likewise, fresh final model evaluation, FinBERT experiments and
interface refinement are development tasks, not setup steps you have missed.
