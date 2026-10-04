# Compact latest features — 4 October 2026

The dashboard now loads the latest sentiment feature rows from a small Parquet
file instead of parsing the full historical CSV. It contains one row for each
of the 44 stocks, with every original feature column and exact stored values.
This is a serving optimization; it does not train a model or change predictions.

## Data and integrity

- Source: `backend/data/processed/features_sentiment.csv` — 104,076 rows,
  102,386,428 bytes with the repository's LF line endings.
- Serving view: `backend/data/serving/features_sentiment_latest.parquet` — 44 rows,
  54,737 bytes.
- Manifest: `features_sentiment_latest.json` records the source and artifact
  SHA-256 hashes, selection rule, format version, row counts, columns and types.
- `.gitattributes` keeps this source CSV in LF line endings on Windows and Linux
  so its byte hash remains identical across Git checkouts.
- Runtime reads the source bytes once to verify their hash. It does not parse
  those bytes into the full dataframe. It then checks the artifact hash and
  schema before loading the latest rows into the existing process cache.
- Missing, stale or corrupt artifacts trigger a logged CSV fallback. The app
  therefore uses current original data rather than a stale serving view.
- Saved files are fixed for the process lifetime. Restart after changing them.
- Finance's latest rows continue to reuse its already validated startup data.
  Timelines and track records retain their full historical coverage.

The manifest is an integrity check for repository-owned files, not a signature
from an external authority. CI also checks semantic correctness: every serving
row must exactly equal the original full CSV's latest lookup. Both hash checks
and semantic checks matter; matching hashes alone cannot prove the right rows
were selected.

## Regeneration

After changing `features_sentiment.csv`, run these commands from `backend` using
the serving environment, then commit the source, Parquet file and manifest together:

```powershell
.venv/Scripts/python.exe -m research.build_serving_snapshots
.venv/Scripts/python.exe -m research.build_serving_snapshots --check
```

The builder verifies that source content remains unchanged during generation and
writes the manifest last. A mixed artifact/manifest generation fails validation.
The check command reports an error and exits unsuccessfully if the files are
stale or any row differs. GitHub's clean Linux serving job runs it before the
API smoke check. The smoke check also rejects a full sentiment CSV parse while
serving the dashboard and explanation requests, so it verifies the fast path.

## Validation and local measurement

- 120 passing backend tests, including seven snapshot tests for precision/nulls,
  stale source bytes, corrupt files, wrong row selection, duplicates and fallback.
- All 44 complete explanations and eight representative endpoint responses have
  the same pre/post hashes.
- Local first-dashboard sample: 1.018 seconds before, 0.134 seconds after.
  This is one Windows diagnostic sample, not a controlled latency benchmark or
  a promise of the same reduction on Render.
- Measurements remain in `backend/data/research/serving_snapshots/`, excluded
  from Git. Source hashing still reads the historical file, and server startup,
  first model-library load and external-provider delays remain separate costs.

No new API keys, serving dependencies, Supabase migration or manual deployment
setting is required. PyArrow was already part of serving; it is now explicitly
included in the evaluation environment for the snapshot regression tests.

Next work can apply the same validated-artifact approach to the exact saved
inputs used by TreeSHAP, reducing its initial historical CSV scan.

The subsequent [XAI serving release](XAI_SERVING_PERFORMANCE.md) prepares and
verifies all 88 native TreeSHAP results, removing both the historical input scan
and the initial model-library import from their normal serving path.
