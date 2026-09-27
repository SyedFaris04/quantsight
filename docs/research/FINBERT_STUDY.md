# FinBERT controlled comparison - 27 September 2026

Run `0024c8e43e438b37` extends the fixed VADER pilot `4e1d6bff50369ed5`.
The live XGBoost model and published trading backtest were not replaced.

## Result

FinBERT did not improve directional accuracy in either tested model. Gradient
boosting showed a small Brier-loss improvement over VADER; logistic regression
became worse on the point estimate. Both remain worse than the constant training-prior
baseline on accuracy and Brier loss. This experiment does not justify promoting a
new serving model, and it does not establish that all uses of financial sentiment fail.

| Prediction model | Inputs | Accuracy | Balanced accuracy | Brier loss (lower is better) |
|---|---|---:|---:|---:|
| Constant training prior | Training-period UP frequency | 55.23% | 50.00% | 0.247623 |
| Logistic regression | Finance | 47.43% | 51.41% | 0.255580 |
| Logistic regression | Finance + coverage | 47.43% | 51.37% | 0.255928 |
| Logistic regression | Finance + VADER | 47.49% | 51.41% | 0.255890 |
| Logistic regression | Finance + FinBERT | 47.33% | 51.24% | 0.256194 |
| Gradient boosting | Finance | 44.90% | 49.45% | 0.258412 |
| Gradient boosting | Finance + coverage | 44.90% | 49.45% | 0.258412 |
| Gradient boosting | Finance + VADER | 44.90% | 49.45% | 0.258412 |
| Gradient boosting | Finance + FinBERT | 44.88% | 49.43% | 0.258354 |

All rows use the same 10,780 evaluation observations. Coverage and sentiment
variants include the news observation flag and article count. This table describes
new research classifiers, not a before/after change to the live XGBoost model.

Paired Brier improvement is baseline loss minus FinBERT loss; positive favors FinBERT:

| Model | Comparator | Improvement | Exploratory 95% block-bootstrap interval |
|---|---|---:|---|
| Logistic regression | VADER | -0.000304 | [-0.000706, 0.000046] |
| Logistic regression | Coverage only | -0.000266 | [-0.000660, 0.000056] |
| Gradient boosting | VADER | 0.000058 | [0.000005, 0.000119] |
| Gradient boosting | Coverage only | 0.000058 | [0.000005, 0.000119] |

The boosting interval excludes zero in this particular unadjusted comparison,
but the gain is very small and does not overcome the simple baseline. These
intervals do not correct for multiple comparisons or earlier model-selection trials.
On the secondary news-covered subset (3,658 rows), FinBERT Brier loss is 0.258516
for logistic regression and 0.260812 for boosting; neither improves on its corresponding
VADER subset score. No trading returns or Sharpe ratios were calculated for these
new predictions.

## What was controlled

- The protocol was saved before this run, with two additional classifier fits and
  no hyperparameter or threshold search. The previous six control fits were reused.
- Training: 2017-2021; separate sigmoid calibration: 2022; evaluation: 2023.
  Boundary rows are purged using each ticker's actual fifth-session outcome date.
- The parent finance, news and feature-list artifact hashes must match. Parent
  prediction rows are joined by ticker/date, checked for duplicate/missing keys,
  and checked for matching labels and coverage. Their published metrics are replayed.
- FinBERT scores **48,655 unique headlines** from the same timestamped archive.
  Scores are mapped back to the original 67,434 ticker/article links. Availability
  dates and eligible evaluation rows are unchanged.
- Frozen model: `ProsusAI/finbert`, revision
  `4556d13015211d73dccd3fdd39d39232506f3e43`. Model files, software versions, source
  code and inputs are hashed in the result. No remote model code is executed.
- CPU float32 inference, batch size 32, maximum 128 tokens, four CPU threads.
  **Zero headlines were truncated.** Similar-length batches reduce padding.
- Polarity is P(positive) minus P(negative), aggregated to daily mean, standard
  deviation and shares above +0.05/below -0.05, matching the VADER feature layout.
- A local SQLite cache preserves completed headline batches. No raw article text,
  downloaded model weights or score cache is sent to the web backend or GitHub.

[FinBERT's paper](https://arxiv.org/abs/1908.10063) and
[model card](https://huggingface.co/ProsusAI/finbert) establish its financial
sentiment task and positive/negative/neutral outputs. They do not establish that
those outputs improve stock returns. Financial PhraseBank was used for fine-tuning,
so evaluating that same model on PhraseBank is not an independent NLP test.

## Limits and next decision

The market period was already examined, existing finance feature selection was
influenced by historical results, and the ticker universe is fixed rather than
point-in-time membership. Archive timestamps remain proxies, while pretrained
model training-corpus overlap is not fully auditable. This is an exploratory
comparison, not a newly untouched final test or proof of historical deployability.

Keep the live model unchanged. The next model-development step is a genuinely
unused evaluation window with training-only feature selection and purged walk-forward
validation. Investigate regime and calibration stability within development data.
Do not keep tuning this 2023 table until a favorable number appears.

## Application and validation

AI Compare now shows the controlled study separately from legacy model variants:
the constant baseline, both model families, probability quality, paired uncertainty,
coverage and limitations. Older labels describing variant differences as a sentiment
"lift" were corrected. The API serves only a small precomputed aggregate report;
web requests never load FinBERT or launch an experiment.

Validation: 56 Python tests passed, including six scorer/alignment tests and three
research API tests. The frontend production build passed. Chromium checks verified
desktop results, both model selectors against actual API values, method disclosure,
390px mobile layout without page overflow, and failure/retry recovery. No uncaught
page errors occurred. Desktop and mobile screenshots were visually inspected.
The existing large frontend bundle warning remains.

Artifacts: [full experiment result](finbert_pilot_results.json),
[protocol](../../backend/research/finbert_protocol.json),
[published aggregate](../../backend/data/research_reports/news_comparison.json).

## Reproduction

From `backend`, with the original audited data and parent predictions available:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-finbert-research.txt
.\.venv\Scripts\python.exe -m research.finbert_pilot
.\.venv\Scripts\python.exe -m research.publish_comparison
```

A completed run refuses to overwrite its results. Interrupted headline scoring
can reuse its cache. An interrupted classifier fit should be logged as another
trial before retrying. Input checks use exact file hashes; another snapshot or
changed file line endings can require a new versioned parent/extension protocol.
The full environment is not yet a portable, completely locked artifact bundle.
Published aggregate results are included in the repository for the website.

Optional browser checks, with backend/frontend running, require Playwright plus
Chromium and use `python -m research.verify_compare_ui`. A custom Chromium path
can be supplied with `--browser`. These tools are for local validation, not Render.
