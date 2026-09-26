# Live rollout status - 27 September 2026

## Verified connections

- Git remote: `https://github.com/SyedFaris04/quantsight.git`.
- Public frontend: `https://quantsight.vercel.app` returned HTTP 200.
- The deployed frontend bundle points to `https://quantsight-rikc.onrender.com`.
- Render health returned HTTP 200 after an initial cold-start timeout.
- Both local frontend and backend reference Supabase project `wlsphzugowaaqljqjuvz`.
- Supabase migration was applied by the user on 27 September. Follow-up
  service-role and anonymous reads of all expected v2 columns now return HTTP 200.
  The earlier 404 / `PGRST205` condition is resolved. Hosted write/trigger checks
  remain separate from this read-only confirmation.
- Render `/backtest` returns 404. `/live-track-record` returns the legacy response,
  without the v2 protocol or five-session horizon. The hosted backend is not current.

These observations establish the frontend-to-backend connection and configured
database identity. They do not verify GitHub webhook settings, deployment secrets,
or the hosted database's write protections. See [read-only check output](live_rollout_check.json).

## Completed rollout preparation

The actual migration passed **11 tests** in isolated
[PGlite PostgreSQL WASM](https://pglite.dev/docs/about), with local equivalents of
Supabase's anonymous, authenticated and service roles. This executes SQL, constraints,
triggers and row-level security rather than mocking database responses.

Checks cover repeatable migration, retained legacy rows, server timestamps,
duplicate inserts, client read/write privileges, RLS when privileges are broadened,
immutable forecasts, recording windows, invalid values, premature/wrong-date
resolution and immutable resolved outcomes. Calendar-date correctness remains the
Python calendar/tracker tests' responsibility. PGlite does not reproduce Supabase's
gateway, authentication service or concurrent network requests.

The database test job is included in `.github/workflows/quality.yml`. Run locally:

```powershell
Push-Location backend/supabase/tests
npm ci --ignore-scripts
npm test
Pop-Location
```

The daily scheduler now validates required configuration, passes secrets through
environment variables, retries transient failures and limits overlapping workflow
runs using [GitHub concurrency](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/control-workflow-concurrency).
It does not cancel a running job when another is queued. Database insert/update
idempotence remains necessary because network timeouts do not guarantee the server
stopped processing. These workflow changes are local and have not run on GitHub yet.

## Remaining rollout steps

1. In the project's [Supabase SQL Editor](https://supabase.com/dashboard/project/wlsphzugowaaqljqjuvz/sql/new),
   run [002_live_predictions_v2.sql](../backend/supabase/002_live_predictions_v2.sql).
   It creates a separate table and retains `live_predictions`. The available local
   service key is for the data API, not a database administration connection;
   no signed-in browser or DDL connection is available to this workspace.
2. Re-run the read-only checker below. Database reads should pass even while Render
   still uses the old backend. A nonzero exit status is expected until all requested
   checks pass.
3. Release the reviewed backend/frontend changes through the existing repository
   and Render/Vercel setup. The workspace contains substantial earlier changes and
   model artifacts; a release needs a concrete reviewed commit, not a blind push.
4. Verify hosted permissions/triggers against the migration and the five-session
   API response. Confirm the two GitHub scheduler secrets match Render's settings.
5. Observe a genuine scheduled after-close forecast and its fifth-session outcome.
   Do not insert synthetic test predictions into production or backfill missed ones.

Read-only diagnostic, run from the repository root:

```powershell
.\backend\.venv\Scripts\python.exe backend/check_live_rollout.py --backend-url https://quantsight-rikc.onrender.com --output docs/live_rollout_check.json
```

The checker reads credentials from local environment configuration, never prints
them, requests no prediction rows from the database, and never calls the admin job.
It verifies all expected table columns, public/service reads, report availability
and freshness, and the live tracker's protocol. It does not certify write permissions
or scheduler execution. The output records UTC time; the document date is Malaysia time.

The user applied the migration. This diagnostic only performs reads; it does not
commit, push, deploy or create forecast rows.
