# News data quality: 7 October 2026

## Decision

- Keep the serving models unchanged. This audit measures data quality, not improved prediction accuracy.
- The existing Drinkall archive is useful for a limited historical study, but it does not supply a complete 2022–2024 news panel.
- Do not repeat the completed VADER/FinBERT experiments to search for a favorable result.
- Register any new cleaning, availability-lag or prediction experiment separately. Preserve the old inputs and reported results.

## What we checked

- Registered [audit protocol](../../backend/research/protocols/news_quality_v1.json) in commit `2fa964a` before measurement.
- Verified the normalized archive against its previously recorded SHA-256 checksum.
- Read only ticker/date keys from the existing finance CSV. No price values, outcome labels or reserved 2025 candidate files were read.
- Checked unique ticker/article keys, consistent article identities, timestamp order and assignment to the first eligible NYSE close.
- Checked exact repeated headlines, URL versions, overlap between chronological article segments and coverage by year/ticker.
- Measured coverage after zero, one and two additional trading sessions of availability delay. No classifiers were trained or scored.
- Saved only aggregate counts and provenance hashes. Raw headlines and URLs remain in the ignored local research directory.

## Main findings

| Check | Finding | Meaning |
|---|---:|---|
| Integrity checks | Passed | Stored keys and availability/session rules are internally consistent |
| Unique article IDs | 50,059 | Distinct URL/headline identities in the retained archive |
| Ticker/article links | 67,434 | One article can link to several stocks |
| Covered instruments | 26 of 44 | Eighteen instruments have no linked articles |
| Unique casefolded headlines | 48,633 | Exact case/whitespace normalization; no semantic matching |
| Headlines attached to multiple article IDs | 1,052 | Some headlines recur at different URLs |
| Excess article IDs for identical headlines | 1,426 | Additional identities beyond one per exact headline |
| Repeated stock/session/headline groups | 1,083 | Repetition may increase daily article count and weight |
| Excess links within those groups | 1,084 | Candidate for a separately registered deduplication test |
| URLs with multiple normalized headlines | 0 | No exact URL/headline version conflict detected in retained records |
| Date-only publication timestamps | 567 articles | Existing rules delay these by one UTC day |
| Availability more than one day after publication | 574 articles | Publication time alone would overstate availability |

- Zero detected URL conflicts does **not** verify the original historical article version or body.
- Repeated headlines can be legitimate recurring stories. These counts are not proof of outcome leakage.
- Median availability delay is about 50 minutes; 95th percentile is about 2.57 hours. These figures count each retained article once. The older source audit counts ticker/article links, so its percentiles differ.

## Coverage with the correct denominator

Each row below divides matched ticker/session keys by **all finance keys in that year**, including instruments without news. These are not the boundary-purged pilot evaluation rows.

| Year | News-covered keys | Finance keys | Coverage |
|---|---:|---:|---:|
| 2017 | 473 | 10,420 | 4.54% |
| 2018 | 355 | 10,542 | 3.37% |
| 2019 | 856 | 10,851 | 7.89% |
| 2020 | 683 | 11,132 | 6.14% |
| 2021 | 2,717 | 11,088 | 24.50% |
| 2022 | 3,171 | 11,044 | 28.71% |
| 2023 | 3,698 | 11,000 | 33.62% |
| 2024 | 19 | 10,824 | 0.18% |

- The previously reported 11.503% coverage uses the entire 2015–2024 panel; it is not 2023 coverage.
- Missing coverage changes substantially over time. A model must have explicit missing-news controls.
- Extra delay of one or two NYSE sessions leaves the overall matched-key count at 11,972, but changes which dates receive news and some annual counts. Equal counts do not imply identical samples or identical predictions.

## Exact overlap between article segments

| Pair | Shared normalized headlines | Shared URLs | Shared article IDs |
|---|---:|---:|---:|
| Fit 2017–2021 / calibration 2022 | 34 | 0 | 0 |
| Fit 2017–2021 / evaluation 2023 | 17 | 0 | 0 |
| Calibration 2022 / evaluation 2023 | 50 | 0 | 0 |

- Segments use article availability dates before label-boundary purging; they do not replace the original model split checks.
- NLP pretraining overlap, near duplicates, ticker-link correctness and original article versions remain unverified.

## External dataset review

- [Drinkall's source README](https://github.com/FelixDrinkall/financial-news-dataset) documents UTC publication/archive timestamps, a 2017–2023 archive, and a CC-BY-NC-SA-4.0 declaration. Its price-enriched and supplied sentiment fields remain excluded from our inputs.
- [FNSPID's Hugging Face page](https://huggingface.co/datasets/Zihan1004/FNSPID) currently reports a dataset-generation/type-parsing failure. A partial viewer or the earlier 2 MiB prefix cannot establish full 44-instrument coverage.
- [FNSPID's repository](https://github.com/Zdong104/FNSPID_Financial_News_Dataset) advertises data through 2023. It therefore does not solve our missing 2024 news by itself. Its July 2025 use announcement and older commercial restrictions remain inconsistent.
- Do not download tens of gigabytes or integrate another text dataset solely because a paper reports a good score on a different prediction task.

## Live system check

Observed on 7 October 2026:

- Local frontend: `http://localhost:3000`; local backend readiness and dashboard/XAI APIs respond successfully.
- GitHub quality workflow passed for `abd2d28`.
- Render readiness reports release `abd2d28`, 44 instruments and no required-data problems.
- Vercel reports a successful deployment. Its served comparison chunk includes the XAI endpoint, Integrated Gradients and LIME.
- Forward-results endpoint: 88 logged, 44 resolved, 44 pending; direction accuracy 52.3%, always-UP accuracy 59.1%, Brier loss 0.2444.
- This is a small, correlated market sample. It does not establish generalization or trading profitability; continue collection and show the baseline.
- These observations are timestamped deployment checks, not guarantees of future provider availability.

## Next sequence

1. Continue immutable forward collection; report resolved sample sizes, date coverage and baselines.
2. Use a short user-comprehension pilot to check whether users understand old signals, P(UP), drawdown and explanations.
3. For another news experiment, first choose a source with verified coverage for the intended development years and historical availability. Keep coverage-only and finance-only controls.
4. Register fixed chronological folds, training-only feature choices, separate calibration, a fit budget and economic evaluation rules before measuring performance.
5. Require improvement against simple baselines across development periods before considering serving-model promotion. Keep final evaluation data sealed until the procedure is fixed.

## Reproduction and validation

From `backend`, using the already acquired, checksum-matching local archive:

```powershell
.\.venv\Scripts\python.exe -m research.news_quality
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_news_quality.py -v
```

- [Machine-readable audit](news_quality_audit_2026-10-07.json) includes per-year/per-ticker coverage, lag counts, protocol and hashes.
- Existing output is preserved. Replay requires the same byte-identical source files, protocol and audit code; the script refuses to overwrite a different result. Line-ending conversion can change byte hashes.
- Five new regression tests cover exact overlap, local coverage denominators, syndication/version distinctions, multi-ticker links, invalid timestamps/identities, reserved-period rejection and holiday/early-close handling.
- All 157 backend tests passed; the frontend production build passed. The completed audit replay matched its saved aggregate result.
- No database migration or new cloud environment variables are needed for this audit.
