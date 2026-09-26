# QuantSight research recommendations

Prepared 27 September 2026 for the 28 September supervisor progress meeting.
Primary sources were reviewed on 26 September. Recommendations below are research
choices, not promises of improved accuracy or investment returns.

## Decision

Prioritize timestamped data, chronological evaluation and strong simple baselines.
Keep the existing application and backtesting engine. Do not replace a live model
because it has the highest score on a previously examined historical period.

We acquired a manageable news archive, checked its actual coverage, and completed
six fixed exploratory model fits. The first sentiment experiment did **not** provide
convincing evidence of improvement. That is a useful FYP result: the experiment can
be reproduced and distinguishes an attractive idea from demonstrated performance.

## What the new data actually contains

The selected [Drinkall financial news archive](https://github.com/FelixDrinkall/financial-news-dataset)
was pinned to revision `3d87f0a8140d6517619d56ad9014f7f6c68abdcb`.
Its seven compressed files total 105,919,076 bytes. Our local audit found **70,974
source rows**, differing from the README's advertised 51,272 articles.

- 67,434 retained ticker/article links; one article can mention several companies.
- Coverage for **26 of QuantSight's 44 instruments**, with 18 missing entirely.
- 11,972 matching ticker/session rows, or 11.503% of the complete 2015-2024 finance
  panel. This denominator includes years outside the archive; it is not 2023 coverage.
- 3,658 of the pilot's 10,780 eligible 2023 rows have news after label-boundary purging.
- 61 source rows rejected because archive download preceded stated publication.
- Availability assigned conservatively using the latest publication, archive and
  modification time, followed by the first eligible NYSE close. Midnight publication
  dates are delayed by a day. Archive timing is still a proxy requiring source review.
- Supplied future/current/past price fields, related-company expansion and precomputed
  sentiment/emotion were excluded. Headlines were scored independently.

Raw articles stay in the ignored local research directory. The declared license is
CC-BY-NC-SA-4.0; attribution and the source license are retained. This acquisition
does not establish unrestricted rights to republish underlying articles in the app.
See the [machine-readable audit](news_candidate_audit.json) for hashes and coverage.

## Fixed pilot: method and outcome

The [protocol](../../backend/research/monday_pilot_protocol.json) was saved before
running the experiment. Run ID: `4e1d6bff50369ed5`.

| Segment | Forecast dates after purging | Rows | Purpose |
|---|---|---:|---|
| Fit | 2017-01-03 to 2021-12-23 | 53,813 | Fit scaler and classifier |
| Calibration | 2022-01-03 to 2022-12-22 | 10,824 | Fit a separate fixed sigmoid |
| Evaluation | 2023-01-03 to 2023-12-21 | 10,780 | Report predictions once |

Each row's five-session outcome must occur before the next segment begins.
Two model families, regularized logistic regression and histogram gradient boosting,
each received three input variants: finance only; finance plus news availability/count;
and those inputs plus VADER headline polarity. All variants use identical evaluation
rows. Missing text has an explicit observation flag. There was no hyperparameter search.

| Model/input | Direction accuracy | Balanced accuracy | Brier loss (lower is better) |
|---|---:|---:|---:|
| Constant training-prior probability | 55.23% | 50.00% | 0.247623 |
| Logistic: finance | 47.43% | 51.41% | 0.255580 |
| Logistic: finance + coverage | 47.43% | 51.37% | 0.255928 |
| Logistic: finance + coverage + sentiment | 47.49% | 51.41% | 0.255890 |
| Gradient boosting: each of the three variants | 44.90% | 49.45% | 0.258412 |

Always predicting UP achieves 55.23% accuracy on these rows. It is a separate
classification baseline; the Brier value above uses the training-period UP frequency,
not a probability of one. All three boosting variants produced identical predictions.

The paired logistic sentiment improvement over coverage alone is only **0.000038**
in Brier loss. Its exploratory 95% block-bootstrap interval is **[-0.000063, 0.000138]**,
which includes zero. The bootstrap resamples 20-session blocks, retaining all stocks
together within each date, for 1,000 repetitions. It does not correct for the project's
entire history of model selection.

Conclusion: this fixed VADER experiment supplies no convincing incremental sentiment
benefit, and both model families underperform simple baselines on headline metrics.
It does not test FinBERT or establish that all sentiment methods are ineffective.
Inspect regime shifts and calibration in later training-only experiments; attributing
this result to a particular cause without another controlled test would be speculation.

This is **exploratory**, because 2023 was previously examined and the existing finance
feature list was influenced by historical results. No pilot trading strategy was run;
these classification scores imply no Sharpe ratio. Serving models were not replaced.
Full metrics, software versions and hashes: [pilot results](news_pilot_results.json).

## Dataset shortlist

| Source | Best use for QuantSight | Status and limitation |
|---|---|---|
| [Drinkall archive](https://github.com/FelixDrinkall/financial-news-dataset) | Immediate historical news ablation | Downloaded, audited and piloted; partial ticker coverage and reused period |
| [FNSPID](https://huggingface.co/datasets/Zihan1004/FNSPID) | Potential broader ticker-linked historical news | Only a bounded 2 MiB prefix inspected; full coverage unverified |
| [StockNet](https://github.com/yumoxu/stocknet-dataset) | Separate replication using social text | Older 2014-2016 experiment, different universe/target; recompute five-session labels |
| [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) | Sentiment-language study | No timestamped return panel; used to fine-tune ProsusAI FinBERT, so not its independent test |
| [Twitter financial news sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | External bullish/bearish/neutral text evaluation | Check files and duplicates first; card counts disagree; sentiment accuracy is not return accuracy |
| [SEC EDGAR](https://www.sec.gov/search-filings/edgar-application-programming-interfaces) | Filing events and fundamental features | Not ingested; use acceptance/release time and historically available facts |
| [ALFRED/FRED vintages](https://fred.stlouisfed.org/docs/api/fred/realtime_period.html) | Macro regime context | Not ingested; use values known then rather than subsequently revised history |
| [FinGPT Forecaster](https://huggingface.co/datasets/FinGPT/fingpt-forecaster-dow30-202305-202405) | Explanation-task reference | Not selected as a market panel; future-derived answers must never become features |

FNSPID's main news CSV is approximately **23.23 GB**. Our nonrandom prefix contains
296 complete records, all for ticker A; 295 timestamps are midnight. This cannot
estimate full-universe coverage. Use conservative availability rules until timestamp
precision is verified. Its [repository](https://github.com/Zdong104/FNSPID_Financial_News_Dataset)
also has conflicting usage statements between its license and README announcements.
The [paper](https://arxiv.org/abs/2402.06698) reports price-regression experiments:
R-squared values must not be presented as directional accuracy or trading success.
See [prefix probe](fnspid_probe.json) and [source registry](dataset_registry.json).

## Research that changes our implementation choices

1. **Use a finance-specific language model as the next controlled comparison.**
   [FinBERT](https://arxiv.org/abs/1908.10063) and the
   [ProsusAI model card](https://huggingface.co/ProsusAI/finbert) support financial
   sentiment classification, not a guaranteed stock-return advantage. Pin the model,
   cache headline scores, and compare VADER versus FinBERT on the same samples.
   Keep coverage-only controls so news frequency is not mistaken for sentiment value.

2. **Keep strong simple models in every experiment.**
   [Grinsztajn et al.](https://arxiv.org/abs/2207.08815) found strong tree performance
   across tabular benchmarks. That is a reason to include regularized linear and
   boosted-tree baselines, not proof that a tree will win this financial task.
   Retain the existing GRU as a comparator before adding another architecture.

3. **Prefer compact text features before large embeddings.**
   [Drinkall et al.](https://arxiv.org/abs/2502.02199) study embedding compression
   for noisy regression. Start with sentiment aggregates. If testing embeddings,
   fit compression only on training data and compare a small predeclared dimension
   set; their regression findings do not establish QuantSight direction accuracy.

4. **Strengthen evaluation across regimes and execution assumptions.**
   [FinTSB](https://arxiv.org/abs/2502.18834) motivates standardized financial
   time-series evaluation and realistic costs. Borrow that evaluation discipline.
   [Microsoft Qlib](https://github.com/microsoft/qlib) offers useful examples of
   reproducible workflows, ranking metrics and cost-aware backtests. Neither requires
   replacing the working FastAPI/React application or copying another market's costs.

5. **Treat pretrained-model knowledge as another possible information leak.**
   [Glasserman and Lin](https://arxiv.org/abs/2309.17322) examine look-ahead bias in
   GPT sentiment-based return prediction. Record model revision, training-data limits
   when known, and text dates. Entity masking is a sensitivity check, not a guarantee
   that historical evaluation is uncontaminated.

6. **Report probability quality and uncertainty, not accuracy alone.**
   Follow [scikit-learn's calibration guidance](https://scikit-learn.org/stable/modules/calibration.html)
   with separate chronological calibration data. Report Brier loss, log loss and
   reliability plots together; Brier also reflects discrimination and is not a pure
   calibration measure. For trading, [deflated Sharpe](https://www.davidhbailey.com/dhbpapers/deflated-sharpe.pdf)
   addresses selection and non-normality, but requires defensible trial information.
   The six newly logged fits do not reconstruct all earlier project experiments.

## Best sequence from here

1. **Monday evidence:** demonstrate the corrected historical backtest, explain the
   data gap, show this fixed pilot and present its negative finding. Use the
   [supervisor brief](../SUPERVISOR_PROGRESS_2026-09-28.md). A few days of prospective
   logging cannot establish a reliable five-session live accuracy estimate.
2. **Reliable collection:** apply and integration-test the prepared Supabase migration,
   then verify scheduler, backend/frontend deployment and immutable live logging.
   These cloud checks remain outstanding, not completed by this research work.
3. **Freeze the next experiment:** select a genuinely unused final period before
   tuning; later prices are candidates requiring acquisition and availability review.
   Select features inside training folds. Use purged chronological walk-forward
   validation for model, calibration and threshold choices; keep final outcomes sealed.
4. **Paired FinBERT study:** compare finance, coverage, VADER and FinBERT with fixed
   samples and a recorded trial budget. Evaluate availability lags, news-covered
   subsets and sector/regime stability. Inspect probability reliability before promotion.
5. **Economic validation:** feed eligible out-of-sample predictions into the existing
   engine. Compare identical dates, costs and allocation rules against SPY, equal weight
   and random portfolios. Include drawdown, turnover and cost sensitivity.
6. **Interface and final evidence:** improve Dashboard/Market/Detail consistency,
   show horizon, model version and missing data clearly; complete browser accessibility
   and responsive checks. Finish attribution quality, account isolation and deployment
   validation before claiming the complete product is ready.

For prediction success use accuracy versus always-UP, balanced accuracy, MCC, ROC-AUC,
Brier/log loss and uncertainty. For investment simulation use net return/CAGR, Sharpe,
Sortino, maximum drawdown, Calmar, turnover/costs and benchmark-relative results.
For system success use reproducibility, data freshness, scheduler reliability and
user task completion. No single number establishes that the project succeeds.

## Reproduction

From `backend` in PowerShell:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-research.txt
.\.venv\Scripts\python.exe -m research.news_dataset
.\.venv\Scripts\python.exe -m research.news_pilot
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The existing pilot refuses to overwrite its run directory. Read the saved result
instead of deleting it; use a new versioned protocol for a genuinely new experiment.
The research extra pins VADER; exact observed library versions are in the result.
The broader environment is not yet a fully locked reproducible environment.
