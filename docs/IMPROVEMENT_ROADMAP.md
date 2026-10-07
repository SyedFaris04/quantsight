# QuantSight improvement sequence

This roadmap follows the current implementation, using the proposal as a research
reference. Existing user changes and trained models are preserved. A negative
result is reported rather than tuned away on the final test period.

## Stage 1 - Historical evaluation foundation (implemented locally)

- [x] Define the existing five-session direction target in one shared module.
- [x] Correct fallback labels; leave unknown future tail outcomes missing.
- [x] Replace weekly look-ahead with prior-close decisions and next-open execution.
- [x] Account for shares, cash, overnight gaps, actual turnover and final liquidation.
- [x] Compare twelve model variants and the ensemble on common dates against SPY
      and an equal-weight universe.
- [x] Compute daily net return, CAGR, Sharpe, Sortino, Calmar, drawdown/duration,
      exposure, costs, tail loss, benchmark-relative statistics and classification baselines.
- [x] Add reproducible random portfolios, cost sensitivity and block-bootstrap intervals.
- [x] Publish an auditable report, daily values and execution CSVs with source hashes.
- [x] Add a Backtesting page, exports, freshness checks and methodology disclosures.
- [x] Add hand-computable accounting tests and API tests; add CI checks.
- [x] Complete generated-report and API verification: 18 automated tests, frontend
      production build, report/four exports/legacy read endpoints, and independent
      replay of every exported execution across 15 portfolios and 462 sessions.

The first published run uses 1,000 random portfolios and 1,000 bootstrap samples.
Independent trade-ledger reconstruction agrees with the reported equity to less
than USD 0.000001. The report is now deployed and its hosted API freshness check passes.
Backtesting browser QA now covers all 15 strategy controls, API-matched percentage-point
units, four real downloads, export-to-report reconciliation, keyboard table scrolling,
390px/320px layouts and error/retry/empty/stale-report states. Desktop/mobile screenshots
were inspected. The page includes metric explanations and visible SPY reference values.

## Stage 2 - Data and live evaluation consistency (local implementation complete)

- [x] Audit source coverage: 2026 news has no overlap with prices ending in 2024;
      WSB ends in August 2021. Save reproducible source hashes and coverage report.
- [x] Preserve new collector timestamps/history; add missingness flags to future builds.
      Legacy text joins remain explicitly timing-unverified.
- [x] Validate observed session gaps; document unknown historical adjustment basis
      and lack of point-in-time universe membership.
- [x] Verify all 30 live model inputs against saved features after universe-wide
      warm-up; share indicator primitives and apply the saved calibrator.
- [x] Implement five-session live tracking with immutable forecast metadata,
      exact target-close resolution, missed-job handling and retained legacy rows.
- [x] Fix summary pagination, inserted/duplicate/skipped counts and scheduler failures.
- [x] Separate live XGBoost from historical consensus/risk in Market; correct Track Record.
- [x] Pass 39 tests, frontend build, API smoke checks and a real live inference check.
- [x] User applied the Supabase migration; server/public reads of expected columns pass.
      Render and Vercel deployed the first release; hosted backtest and v2 tracker checks pass.
      Eleven isolated PostgreSQL migration tests pass. See [rollout status](LIVE_ROLLOUT.md).
- [ ] Verify actual hosted recording and fifth-session resolution through the scheduler;
      read-only readiness checks do not establish hosted write/trigger behavior.
      The 7 October reliability update adds coverage verification, safe stage reports,
      cache fixes and a recovery schedule. The next eligible scheduled collection still
      needs observation. See [daily collection reliability](DAILY_COLLECTION_RELIABILITY_2026-10-07.md).
- [x] Acquire and audit an overlapping historical news dataset; run a versioned
      timestamp-aware exploratory pilot. Coverage is limited to 26 of 44 tickers.
- [ ] Acquire/freeze a genuinely unused evaluation window before further tuning.
      The new historical archive does not make the previously examined period fresh.
      A separate 2025 candidate price snapshot is now frozen and passes structural
      checks for all 44 instruments plus six sector ETFs. Prior external use remains
      unconfirmed; no final-test scores were computed. See [fresh validation](research/FRESH_VALIDATION.md).

Details and rollout order: [Data and live evaluation](DATA_AND_LIVE_EVALUATION.md).
Stage 2's local engineering and initial historical data acquisition are complete;
hosted write/outcome verification and fresh-window acquisition remain open.
AI Compare has desktop/mobile browser verification on the deployed site. Backtesting
also has desktop/mobile checks; other pages still need visual QA.

## Stage 3 - Controlled model research (exploratory pilot complete)

- [x] Review primary papers, repositories and dataset cards; retain a source registry.
- [x] Audit 70,974 news source rows, excluding supplied future-price and NLP fields.
- [x] Complete six fixed model fits with purged fit/calibration/evaluation segments.
- [x] Compare finance, coverage-only and VADER sentiment on matching rows; report
      baselines and paired block-bootstrap uncertainty. No convincing sentiment gain;
      serving models unchanged.
- [x] Complete two frozen FinBERT extension fits and publish paired comparisons in AI Compare.
      No directional-accuracy improvement; a small boosting Brier gain does not beat the
      constant baseline. Full Python suite now contains 56 passing tests.
      See [FinBERT study](research/FINBERT_STUDY.md).

Results and next choices: [Research recommendations](research/RESEARCH_RECOMMENDATIONS.md).
Meeting evidence: [28 September supervisor brief](SUPERVISOR_PROGRESS_2026-09-28.md).
Eight recent fits are recorded across the VADER pilot and FinBERT extension,
but do not reconstruct older unrecorded trials.
Still required:

- Freeze a fresh final test window before further experiments.
- [x] Build tested calendar-exact labels and purged fit/calibration/validation segments
      on the new development snapshot; reject candidate-holdout rows in the loader.
      Three development folds are prepared; 72 Python tests pass. Model selection and
      a confirmed unused final window remain outstanding.
- Use shared chronological folds, label-overlap purging and training-only feature selection.
- [x] Run the first fixed finance-development comparison on 2022/2023/2024 with
      13 predeclared features, three model families, separate sigmoid calibration,
      nine predictor fits and nine calibrator fits. All six model outputs failed
      to beat the training-prior baseline on mean Brier loss; no model was promoted.
      AI Compare publishes the year results, reliability bins and uncertainty.
      See [finance development study](research/FINANCE_DEVELOPMENT_STUDY.md).
- Compare finance, news, social polarity and emotion ablations on matching samples.
- Tune model/threshold/calibrator on appropriate separate validation windows.
- Test deadband, excess-return and ranking targets as explicitly separate experiments.
- Compare multiple seeds, regimes/sectors and error patterns; measure consensus correlation.
- Version every trial and model artifact; refresh explanations/calibration/evaluation together.
- Add abstention/coverage curves and risk/position-sizing strategy ablations.
- [x] Add a fixed confidence/coverage audit for nine development candidates across
  seven cutoffs and four period views, with matching baseline scores and small-sample
  disclosure. No reliable raw-XGBoost improvement or live cutoff change is established;
  risk/position-sizing ablations remain separate work. See
  [confidence audit](research/CONFIDENCE_COVERAGE_AUDIT.md).

Gate: performance relative to baselines, with uncertainty, on untouched data; no required
accuracy percentage and no promise that complexity must outperform simpler models.

## Stage 4 - Explanations and decision support

- [x] Add verified per-prediction native TreeSHAP for both saved XGBoost variants;
      reconstruct the raw score and match the saved direction/calibrated probability.
      Distinguish existing rule-based commentary and attention weights.
- Verify counterfactuals against model outputs and realistic feature constraints.
- Evaluate explanation fidelity/stability and calibrate confidence terminology.
- Show missing/stale data, horizon, version, and No Reliable Signal states clearly.

## Stage 5 - Interface and product correctness

- Extend Backtesting's spacing, typography and evidence-first layout across the app.
- Redesign Dashboard, Market and Detail; group research vs learning navigation.
- [x] Redesign Dashboard with dated historical snapshots, saved-news window/age,
      research links, separate retry states and mobile checks. Remove the static
      sidebar "Live" label and update chatbot tool descriptions to respect source dates.
      Seven provenance tests bring the Python suite to 63; Market/Detail redesign remains.
- Fix portfolio daily P&L and account/guest-state separation.
- Fix game replay scoring; distinguish model-interpretation from outcome prediction.
- Add keyboard/focus, chart alternatives, responsive and loading/error-state checks.
- [x] Split route bundles: initial JavaScript is approximately 57% smaller, with
      page-level loading/error recovery and seven request-race regression tests.
      Production-build browser checks cover every route, actual timeout/retry,
      chat persistence and mobile layouts. See [loading and reliability](LOADING_AND_RELIABILITY.md).
- Continue splitting oversized page components where it improves maintainability.

## Stage 6 - Database, deployment and maintainability

- Add explicit numbered Supabase migrations, display_name and private/public data separation.
- Make competitive scores server-authoritative; test RLS and account isolation.
- Refactor backend routers/services/repositories and add a shared model registry.
- Separate serving/training dependencies, pin environments and version deployment artifacts.
- Add readiness, latency/error monitoring, scheduler heartbeat and data/model drift reports.
- Verify GitHub -> Render/Vercel release and Supabase settings on the actual deployments.

## Stage 7 - FYP evidence and final validation

- Present historical simulation and forward tracking separately.
- [x] Add date-level forward evidence, paired baselines, guarded uncertainty,
  version separation, collection-gap warnings and an aggregate export. See
  [forward evidence](FORWARD_EVIDENCE_2026-10-07.md). Daily collection has recent
  failures; reporting improvements do not resolve that operational issue.
- [x] Add year/month backtest consistency for all 15 portfolios, paired benchmark
  returns, exact partial-period dates and checked exports. The ensemble beats SPY
  in 9 of 23 observed months; its advantage differs across the two study windows.
  This is descriptive historical evidence, not fresh-window validation.
  See [period consistency](BACKTEST_PERIOD_CONSISTENCY.md).
- Add fresh-window robustness, start-date/regime/sector sensitivity, and hypothesis-aware
  statistical comparisons. Compute deflated Sharpe only with a defensible trial history.
- Conduct novice/expert task and comprehension evaluation under university requirements.
- Reconcile proposal/thesis objectives with implemented methods and honest limitations.
- Produce a reproducible methodology, experiment tables, demo script and final report.

No stage requires an artificial improvement to a headline number to be considered complete.
