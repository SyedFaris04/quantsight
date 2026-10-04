# Prepared native TreeSHAP explanations — 4 October 2026

The API now serves prepared native TreeSHAP results for the latest saved
predictions of both XGBoost models. The public endpoint and explanation fields
are unchanged. These are the existing model's exact explanations, computed once
from its saved trees, scaler and calibrator rather than calculated again on the
first web request. This changes serving performance, not prediction accuracy.

## Evidence and integrity

- 44 explanations per model, 88 total. Each record includes the exact historical
  feature values, scaled values, contributions, baseline, input hash and model hash.
- The selected feature date is the saved prediction's date. A newer available
  feature row is not substituted. Current saved prediction date: 20 December 2024.
- Preparation uses `get_native_attribution`, which retains exact TreeSHAP,
  best-iteration handling, margin reconstruction and saved-output reproduction
  checks. No models are fitted and no prospective holdout data is opened.
- Each manifest binds the model bundle, complete feature CSV and prediction CSV
  by SHA-256. Model bundles include their scaler, calibrator and feature order.
- The manifest also binds normalized source text of `feature_attribution.py` and
  `cv_utils.py`, the artifact bytes, XGBoost version and ticker set.
- Runtime verifies these hashes before caching the prepared results. File
  signatures detect changed generations; saved assets are otherwise fixed during
  a process. Returned records are copies so callers cannot mutate cached evidence.
- Missing, stale or corrupt prepared data triggers native computation with the
  original reproduction checks. If current inputs cannot reproduce the saved
  prediction, the existing endpoint returns a safe 503 error.
- Unknown tickers remain 404. LSTM attribution remains unsupported by this endpoint.

Integrity hashes cover repository-owned assets; they do not independently prove
the explanation math. The separate CI check recomputes all 88 explanations from
the full original inputs and requires matching complete result dictionaries.

## Verification and measurement

- 127 passing backend tests, including seven additional artifact checks: no model
  or CSV load on the prepared path, exact historical dates, cached-copy isolation,
  stale model/input/prediction bytes, corrupt files, changed implementation,
  inconsistent records and altered contributions despite updated file hashes.
- All 88 explanation response hashes match the pre-change native results.
- Clean serving verification requires both attribution endpoints to work without
  importing XGBoost or sklearn. Live prediction routes can still import their
  model libraries when needed.
- Prepared finance JSON: 196,435 bytes; sentiment JSON: 249,869 bytes, plus two
  small manifests. Full data and models remain available for native fallback.
- Local fresh-process samples: finance first request 3.884 s before / 0.124 s
  after; sentiment 1.012 s before / 0.119 s after. Finance paid the initial model
  import in the earlier sample; sentiment reused those libraries. These are
  diagnostic samples, not controlled benchmarks or a production speed guarantee.
- Local serving smoke resident memory after Dashboard, history, explanation and
  both prepared XAI routes was about 303 MiB. Model libraries are loaded later
  if a live prediction or native fallback needs them.

Measurements and pre/post hashes remain in
`backend/data/research/xai_performance/`, excluded from Git.

## Regeneration and deployment

After changing a bound model, input dataset, prediction file or explanation code,
run these commands in the backend serving environment and commit both JSON
artifacts and manifests with the source change:

```powershell
.venv/Scripts/python.exe -m research.build_attribution_snapshots
.venv/Scripts/python.exe -m research.build_attribution_snapshots --check
```

The builder checks source stability, then publishes each manifest last. GitHub's
clean Linux serving job runs the recomputation check before its API smoke check.
`.gitattributes` preserves LF line endings for hashed CSV/JSON assets across
Windows and Linux; code hashes normalize text line endings.

No new API keys, dependencies, Supabase migration or deployment setting is needed.
This prepared path applies to historical saved XGBoost predictions. It does not
claim to explain changing live Yahoo inputs or the LSTM models.
