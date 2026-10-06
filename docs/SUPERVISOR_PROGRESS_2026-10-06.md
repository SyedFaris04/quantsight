# Supervisor progress update, 6 October 2026

Presentation: [Updated PowerPoint](presentations/QuantSight_Supervisor_Progress_2026-10-06_Updated.pptx).

This update compares the previous presentation at commit `23df560` with project
work through `168fcc9`. The slides include source references and explanations in
their speaker notes. Screenshots show the real local application on 5 October.

## New work since the previous presentation

- Added exact native TreeSHAP explanations for both saved XGBoost variants.
  All 88 latest historical outputs reproduce their saved results.
- Repaired the chatbot's retired-provider-model failure and added clearer recovery.
- Completed a fixed finance study with three chronological validation years,
  13 inputs and 32,472 validation ticker-days.
- Added probability reliability and confidence/coverage diagnostics.
- Added yearly and monthly backtest comparisons with matching benchmarks.
- Reduced initial JavaScript bytes by 57.2% in the measured production build.
- Improved request cancellation, serving readiness and prepared explanation data.
- Created the 72-page illustrated system guide.

## Results to explain carefully

- The training-prior baseline wins the finance study's Brier-based selection rule.
  No candidate replaced live inference.
- At 60% confidence, raw development XGBoost retains 12.91% of rows and achieves
  51.51% accuracy, compared with 51.49% for its baseline on those same rows.
- The 70% cutoff retains only 32 rows. Its 62.50% accuracy equals the baseline.
- The historical ensemble beats SPY in 9 of 23 observed months. It gains 29.40%
  versus SPY's 20.60% in the partial 2023 window, and 25.18% versus 25.78% in the
  partial 2024 window. These are simulations from reused development history.
- Model probability, observed accuracy and net trading returns are distinct.

## Verification and current limits

- Last documented application release verification, 4 October: 144 backend tests,
  seven frontend tests, production build and serving/evidence checks passed.
- Last verified forward-tracker snapshot, 5 October: 44 logged forecasts,
  44 pending and zero resolved. Its target session has now passed.
- Hosted readiness and tracker reads timed out on 6 October. Current outcome
  resolution and hosted availability remain unconfirmed.
- The 2025 candidate final-test period stays unopened pending confirmation that
  it has never guided model choices, including outside this project folder.
- User understanding, chatbot factual accuracy and prospective performance still
  require evaluation.

## Demo and next milestones

- Show Stock Detail attribution, the chatbot, AI Compare diagnostics and the
  Backtesting period breakdown.
- Start the local backup before presenting, following [HOW_TO_RUN](../HOW_TO_RUN.md).
  Local external-provider features still require internet access.
- Agree on an eligible final evaluation period and measurable user-testing criteria.
- Register the next experiment and trading procedure before fitting or final scoring.

The presentation is a documentation update. It performs no model training,
holdout evaluation, forecast insertion or database migration.
