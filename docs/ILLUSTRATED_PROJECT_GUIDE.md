# QuantSight illustrated project guide

Documentation: 5 October 2026. Code baseline: `8bd515a`.

This companion contains the feature explanations used in the PDF. Screenshots are real local UI captures from an isolated guest session.

## System flow

Prices / news / social posts -> checked daily feature panels -> chronological training and calibration -> classification, controlled studies and backtests -> versioned serving assets -> GitHub checks -> Vercel frontend / Render API; Supabase schema is configured separately.

## Dashboard: the saved model snapshot

- **What it is:** The home page gives a quick view of the research system and links to its evidence.
- **Data:** The API reads the latest saved prediction row for each of 44 stocks and ETFs. The displayed signal date is 20 December 2024.
- **Calculation:** The four core models vote on direction. More BUY votes give BUY; more SELL votes give SELL; a tie gives HOLD. The cards count these outcomes.
- **Interpretation:** Mean confidence is an average model score. It is not the measured accuracy of the ensemble or a current market signal.
- **Code / evidence:** `frontend/src/pages/Dashboard.jsx`

## Dashboard: historical BUY candidates

- **What it is:** This list helps the user find instruments to inspect more closely.
- **Data:** The same dated, four-model prediction snapshot supplies the candidates and their individual scores.
- **Calculation:** The page keeps overall BUY results, then ranks them by model agreement and confidence. Selecting a ticker opens Stock Detail.
- **Interpretation:** Agreement means the models chose the same direction. Models can agree and still be wrong. These are saved research outputs.
- **Code / evidence:** `backend/copilot_engine.py`

## Dashboard: saved news sentiment

- **What it is:** Positive, neutral and negative shares describe the language in the saved headlines.
- **Data:** The GDELT archive is scored using VADER. This view covers 1,032 unique headlines in its recent saved 14-day window, ending 23 April 2026.
- **Calculation:** The server deduplicates archived headlines, groups their polarity labels and calculates each group as a share of the total. Daily values are expandable.
- **Interpretation:** This is an archive, not a live feed. Its 2026 news is separate from the models' 2024 signal date. Language sentiment is not return accuracy.
- **Code / evidence:** `backend/dashboard_evidence.py`

## Dashboard: archived headline links

- **What it is:** The news list lets the user inspect the underlying headline, ticker, date and publisher.
- **Data:** These are stored GDELT entries returned by the market-news API. The page links to the original publishers.
- **Calculation:** The API sorts saved entries by date and returns a bounded list. Clicking a headline opens its publisher in a new tab.
- **Interpretation:** A ticker match can be noisy, and publishers may change their pages. Saved headlines do not become inputs to a past prediction simply by appearing here.
- **Code / evidence:** `backend/main.py`

## Market: scan and filter instruments

- **What it is:** The market table compares the four core model signals for each instrument.
- **Data:** Historical mode reads the saved prediction CSVs and the explanation API. The default rows are from the December 2024 snapshot.
- **Calculation:** Search by ticker; filter by BUY, SELL or HOLD and agreement; sort the results. Overall direction comes from model votes. Risk uses indicator rules.
- **Interpretation:** Historical confidence columns contain calibrated P(UP), including on SELL rows. The displayed risk categories are rules, not estimated loss probabilities.
- **Code / evidence:** `frontend/src/pages/Overview.jsx`

## Market: current daily inference

- **What it is:** Live mode requests a new finance-only XGBoost prediction using a completed daily market session.
- **Data:** Yahoo Finance supplies adjusted daily prices for the required instrument and sector universe. The backend applies the saved model bundle.
- **Calculation:** It checks the NYSE calendar, excludes unfinished bars, recreates the 30 finance inputs, scales them and applies the saved calibrator. The target is five sessions later.
- **Interpretation:** This is daily inference, not an intraday forecast. Missing or stale required inputs produce an unavailable result. Historical four-model votes are not used in live mode.
- **Code / evidence:** `backend/live_signals.py`

## Stock Detail: signal and navigation

- **What it is:** Stock Detail brings together the signal, charts, explanations, news, history and learning tools for one ticker.
- **Data:** The stock and explanation APIs read the saved price features and four core model outputs for AAPL in this example.
- **Calculation:** The overview combines model votes and their average P(UP). Tabs switch between Overview, Technical, Sentiment & Emotion, History and Learn.
- **Interpretation:** The overall vote can differ from one individual model. The headline confidence is an average score, not the probability that buying the stock will make money.
- **Code / evidence:** `frontend/src/pages/Detail.jsx`

## Price chart: candles and volume

- **What it is:** Each candle shows one session's open, high, low and close. The bars below show volume.
- **Data:** The stock API reads the saved OHLCV price panel. Period controls request different lengths of this daily history.
- **Calculation:** The candle body runs from open to close; the wick shows the high and low. Colour shows whether that day closed above or below its open.
- **Interpretation:** Green candles are observed price moves, not BUY predictions. This historical chart ends in December 2024 and does not show intraday candles.
- **Code / evidence:** `frontend/src/components/CandlestickChart.jsx`

## RSI chart and saved signal markers

- **What it is:** The RSI view compares recent upward and downward price moves on a scale from 0 to 100.
- **Data:** RSI values come from the saved feature panel. The chosen model's saved BUY/SELL dates supply the optional markers.
- **Calculation:** RSI uses a 14-session lookback. Guide lines at 30 and 70 help identify low and high momentum readings; controls choose the period and model overlay.
- **Interpretation:** An RSI threshold is a rule of thumb. A high or low value does not guarantee a reversal, and markers are model decisions rather than executed trades.
- **Code / evidence:** `frontend/src/pages/Detail.jsx`

## MACD chart and model overlay

- **What it is:** MACD shows the difference between shorter and longer exponential moving averages.
- **Data:** The saved price features provide MACD. Signal markers are read from the selected core model's historical predictions.
- **Calculation:** MACD compares the 12- and 26-session averages. Its 9-session signal line is used by the indicator rules; the chart displays MACD over the selected period.
- **Interpretation:** A bullish indicator reading does not force a model to predict BUY. The model considers several inputs, while the chart shows one indicator.
- **Code / evidence:** `backend/technical_indicators.py`

## Indicator scores and model votes

- **What it is:** The summary separates technical, sentiment and emotion assessments from the four saved model signals.
- **Data:** The explanation API reads the latest historical feature row and each core model's output.
- **Calculation:** Indicator rules produce the summary scores. The model section shows XGBoost Finance, XGBoost +Sentiment, LSTM Finance and LSTM +Sentiment.
- **Interpretation:** These indicator scores are not SHAP contributions. Historical text values are zero in the current test period. Differences between variants do not prove a news benefit.
- **Code / evidence:** `backend/copilot_engine.py`

## Explainable AI: the exact saved prediction

- **What it is:** The attribution panel explains one saved XGBoost output. Finance and Finance +Sentiment can be selected.
- **Data:** The backend loads the exact saved model bundle and ticker/date input row, then verifies the prediction against the saved CSV.
- **Calculation:** The panel separates raw P(UP), calibrated P(UP) and the raw score. For AAPL Finance, raw P(UP) is 46.49%, calibrated P(UP) is 55.19%, and direction remains SELL.
- **Interpretation:** Direction uses the raw 50% threshold. Calibration can cross 50% without changing that saved label. This panel explains a historical XGBoost output, not the live model or consensus.
- **Code / evidence:** `backend/feature_attribution.py`

## Explainable AI: feature contribution bars

- **What it is:** The bars show which inputs pushed the tree model's raw score toward UP or DOWN.
- **Data:** Native XGBoost TreeSHAP calculates contributions from the saved trees after applying the bundle's feature order and scaler.
- **Calculation:** The chart shows the eight largest absolute contributions and combines the remaining inputs. Baseline plus all contributions reconstructs the raw log-odds score.
- **Interpretation:** Contributions are log-odds, not percentage points or causal effects on stock prices. A zero-valued input can still have a contribution in a tree model.
- **Code / evidence:** `backend/feature_attribution.py`

## Explainable AI: inputs and verification

- **What it is:** The expanded table lets the user inspect the actual input values, scaled values and contributions.
- **Data:** The table is built from the exact row used for this saved prediction. It contains all 30 finance inputs; the screenshot shows its first section.
- **Calculation:** The backend checks raw-score reconstruction, raw probability, calibrated probability and saved direction. Model and input hashes identify the evidence version.
- **Interpretation:** A verification failure produces an unavailable explanation. Matching the saved prediction proves consistency, not that the prediction is correct or that the model has a fresh trading edge.
- **Code / evidence:** `backend/feature_attribution.py`

## Bull case and bear case

- **What it is:** This panel presents evidence for both directions and scenarios that could change an indicator assessment or model vote count.
- **Data:** The explanation engine combines saved RSI, MACD, price trend, historical accuracy and the core models' votes.
- **Calculation:** Rules describe supportive and opposing indicators. Reliability uses historical performance. The scenario section shows relevant indicator thresholds or vote changes.
- **Interpretation:** These are rule-based summaries. Scenarios do not rerun the trained predictor or guarantee that its forecast would change. Historical reliability may not carry into a new period.
- **Code / evidence:** `backend/copilot_engine.py`

## Indicator and vote scenarios

- **What it is:** Scenarios describe a threshold or vote change that would alter an indicator assessment or the overall model consensus.
- **Data:** The explanation engine uses the saved indicator values and the four models' direction votes.
- **Calculation:** The displayed conditions identify a change to RSI, momentum, trend or the voting balance. They give the user a simple way to inspect sensitivity.
- **Interpretation:** These conditions do not rerun the trained predictor. They are not verified counterfactual predictions and do not guarantee a change in its raw or calibrated score.
- **Code / evidence:** `backend/copilot_engine.py`

## Risk assessment

- **What it is:** The risk card gives a simple warning level for the current saved stock analysis.
- **Data:** It uses saved price indicators, model disagreement and confidence from the explanation engine.
- **Calculation:** A fixed rule-based score maps the evidence to Low, Medium or High and lists contributing factors. It helps the user inspect uncertainty.
- **Interpretation:** This is not portfolio VaR, a probability of losing money or a personalised suitability check. Backtesting has separate measured volatility and drawdown metrics.
- **Code / evidence:** `backend/copilot_engine.py`

## Technical tab: indicator explanations

- **What it is:** The technical cards describe RSI, MACD, Bollinger Bands and the moving-average trend in simple terms.
- **Data:** The saved finance feature row supplies each numeric value. The explanation engine supplies the accompanying text.
- **Calculation:** Rules compare RSI with thresholds, MACD with its signal line, price with its bands, and price/SMA10 with SMA50.
- **Interpretation:** The words Bullish and Bearish describe indicator rules. They are not independent trained models and do not guarantee the stock's next move.
- **Code / evidence:** `backend/copilot_engine.py`

## Technical tab: LSTM attention

- **What it is:** Attention weights show how the LSTM layer distributes weight across its ten input sessions.
- **Data:** These weights are saved with the historical LSTM prediction and returned by the stock-analysis data.
- **Calculation:** The ten weights add to approximately 100%. The chart scales bars relative to one another so small differences are visible.
- **Interpretation:** For AAPL, all weights are near the uniform 10% level. Attention is internal weighting, not exact reasoning, causal importance or TreeSHAP.
- **Code / evidence:** `frontend/src/pages/Detail.jsx`

## Sentiment tab: language scores and variants

- **What it is:** This view separates saved news and social sentiment from the differences between finance-only and text-feature models.
- **Data:** GDELT and Reddit/WSB supply the original text sources. VADER scores language polarity, and daily aggregates are joined to the feature panel.
- **Calculation:** The page displays available polarity scores and compares the saved model variants' P(UP) values. A difference is a descriptive model-output difference.
- **Interpretation:** Current historical text inputs are zero on the latest prediction date. Labels such as improvement or impact in this older view should not be interpreted as proven sentiment benefit.
- **Code / evidence:** `frontend/src/pages/Detail.jsx`

## Sentiment tab: emotion features

- **What it is:** Emotion features describe feelings expressed in social posts, beyond positive or negative sentiment.
- **Data:** A pretrained GoEmotions-based RoBERTa model scores WSB posts. The project aggregates six groups: fear, optimism, anger, excitement, confusion and disappointment.
- **Calculation:** Post scores are aggregated by ticker and day for the text-feature variants. The interface shows no emotion chart when the latest date has no WSB coverage.
- **Interpretation:** AAPL has no matching WSB posts on its latest saved prediction date. Missing coverage is not evidence that traders felt neutral, and emotion-model F1 is not stock prediction accuracy.
- **Code / evidence:** `backend/build_features.py`

## Sentiment tab: ticker-specific news

- **What it is:** Stock Detail also lists saved headlines associated with the selected ticker.
- **Data:** The news API reads the GDELT archive and returns a limited set of ticker matches, dates and publisher links.
- **Calculation:** Entries are filtered by ticker and ordered by saved date. The user can open a publisher to check the full context.
- **Interpretation:** The AAPL headlines shown here are from April 2026. They cannot explain a December 2024 forecast, and multilingual or ambiguous matches require care.
- **Code / evidence:** `backend/main.py`

## History tab: per-stock accuracy

- **What it is:** This shows how often each core model matched the historical five-session outcome for this particular stock.
- **Data:** The accuracy-history API compares the saved prediction labels with the stored historical target labels for that ticker.
- **Calculation:** Correct divided by evaluated predictions gives accuracy. The page shows a recent 20-prediction sample as well as each model's longer saved history.
- **Interpretation:** A small recent sample can look unusually strong. This reused historical window is separate from the forward Live Track Record, and overlapping targets are dependent.
- **Code / evidence:** `backend/copilot_engine.py`

## History tab: decision timeline

- **What it is:** The timeline shows how the saved signal changed over the last seven available sessions.
- **Data:** The history API reads dated core-model outputs and the corresponding historical indicator rows.
- **Calculation:** It displays one dated signal card per session and a short indicator summary. The user can compare changes in direction and scores.
- **Interpretation:** This is a history of model decisions, not a record of real trades. Its confidence labels are historical P(UP), including on SELL decisions.
- **Code / evidence:** `backend/copilot_engine.py`

## Learn tab: the indicator glossary

- **What it is:** The glossary helps a new user understand the terms used across Stock Detail.
- **Data:** Definitions are written in the frontend. This feature does not request a price forecast or use a language-model call.
- **Calculation:** Open the glossary and select a term to reveal its plain-language explanation. The example expands the decision timeline.
- **Interpretation:** Some older definitions say tomorrow or describe confidence loosely. The actual system target is five sessions, and historical confidence is P(UP). Use the current method explanation in this guide.
- **Code / evidence:** `frontend/src/pages/Detail.jsx`

## Finance development: compare with a baseline

- **What it is:** This study asks whether three fixed finance models outperform a simple training-frequency forecast.
- **Data:** A frozen Yahoo price snapshot supplies 44 instruments and 13 fixed finance inputs. Validation uses 2022, 2023 and 2024: 32,472 ticker-days.
- **Calculation:** Logistic regression, histogram boosting and XGBoost use chronological fit, separate calibration and validation periods. There are nine predictor fits and nine calibration fits.
- **Interpretation:** These are development years. No tested predictor beat the training-prior baseline on equal-year mean Brier loss, so none was promoted to the live system.
- **Code / evidence:** `backend/data/research_reports/development_models.json`

## Finance development: probability quality

- **What it is:** The table compares direction accuracy and the quality of the probability forecasts on matching observations.
- **Data:** The study JSON contains raw and calibrated versions of the three predictors plus training-prior, recent-prior and always-UP baselines.
- **Calculation:** Choose an individual validation year or the equal-year mean. Brier is the mean squared probability error; smaller is better. Other columns show balance, MCC, AUC and UP rate.
- **Interpretation:** The training-prior baseline has mean Brier 0.250877, versus 0.251999 for raw XGBoost. A similar accuracy score does not imply a better probability forecast.
- **Code / evidence:** `backend/data/research_reports/development_models.json`

## Probability reliability and uncertainty

- **What it is:** The expanded panel checks whether predicted probabilities match observed outcomes and shows uncertainty in the study comparison.
- **Data:** It uses the same fixed finance-development predictions. Select a candidate and year to inspect occupied probability bins.
- **Calculation:** For the displayed raw logistic model in 2024, the large 50-60% bin predicts 55.02% UP and observes 54.76%. Paired, 20-session block resamples estimate uncertainty in the Brier difference.
- **Interpretation:** A close match in one bin does not prove overall superiority. Some bins are small. The displayed improvement interval includes zero and does not correct for selecting among several candidates.
- **Code / evidence:** `frontend/src/components/DevelopmentComparison.jsx`

## Confidence audit: accuracy versus coverage

- **What it is:** This diagnostic asks what is lost or gained when lower-confidence predictions are removed.
- **Data:** It reuses the fixed development predictions. It compares nine outputs, seven confidence thresholds and four period views, without fitting new models.
- **Calculation:** Confidence is max(P(UP), P(DOWN)). For every retained set, the training-prior baseline is evaluated on exactly the same rows. Coverage is retained rows divided by eligible rows.
- **Interpretation:** At the raw XGBoost 60% threshold, only 12.91% of rows remain and accuracy is 51.51%. Higher confidence did not guarantee better results. No cutoff is recommended or applied live.
- **Code / evidence:** `backend/data/research_reports/development_confidence.json`

## Controlled news study

- **What it is:** This separate experiment compares finance, news coverage, VADER and FinBERT on the same historical observations.
- **Data:** A timestamped financial-news archive supplies historical text. The 2023 comparison has 10,780 evaluated ticker-days, 3,658 with news and coverage for 26 of 44 instruments.
- **Calculation:** The study adds coverage and sentiment inputs under a fixed time-availability rule. The model selector switches between logistic regression and histogram boosting.
- **Interpretation:** The screenshot uses logistic regression. Its constant training-prior baseline is stronger on Brier than the tested news variants. This study is exploratory and does not update serving models.
- **Code / evidence:** `docs/research/FINBERT_STUDY.md`

## FinBERT versus VADER: inspect uncertainty

- **What it is:** This panel checks whether the finance-language sentiment model gives a clear improvement over the simpler VADER scorer.
- **Data:** Both variants use the same research rows and news timing. The report stores their Brier difference and a block-bootstrap interval.
- **Calculation:** A positive Brier improvement favours FinBERT. Blocks preserve some time dependence when resampling the observed period.
- **Interpretation:** For the displayed logistic comparison, the interval includes zero, so there is no clear improvement. Results depend on the selected predictor and do not prove better live accuracy.
- **Code / evidence:** `frontend/src/components/ResearchComparison.jsx`

## Historical core models: table and bar chart

- **What it is:** The historical comparison displays accuracy, F1, precision, recall and AUC for the four core variants.
- **Data:** Saved model_metrics.json supplies these figures. The adjacent bar chart plots the same measurements side by side.
- **Calculation:** A star marks the largest saved score in each column. The LSTM +Sentiment variant has 54.25% saved accuracy; finance-only LSTM has 53.75%.
- **Interpretation:** These are older reused-period results, not the controlled development study above. Legacy LSTM+Transformer labels refer to LSTM with attention; the separate Transformer Encoder is another model.
- **Code / evidence:** `backend/data/predictions/model_metrics.json`

## Historical profiles: bars, differences and radar

- **What it is:** The charts make trade-offs between the four saved core variants easier to see.
- **Data:** All bars, variant differences and radar axes are derived from the same historical metrics file.
- **Calculation:** Bars compare each metric directly. Variant differences subtract finance-only scores from text-feature scores. The radar view plots all five metrics as a profile.
- **Interpretation:** Chart size is not evidence of robustness. A positive variant difference cannot isolate the effect of sentiment when the evaluated historical text inputs are zero.
- **Code / evidence:** `frontend/src/pages/Compare.jsx`

## Historical variant differences

- **What it is:** This section shows the numerical differences between the saved finance-only and text-feature variants.
- **Data:** The interface reads the historical metric values for XGBoost and LSTM, then compares each pair.
- **Calculation:** Each difference is the text-feature score minus its finance-only counterpart. For accuracy, a difference of 0.50 means 0.50 percentage points.
- **Interpretation:** The page does not establish that adding news caused the difference. Different training behaviour and zero text coverage can still produce different model outputs.
- **Code / evidence:** `frontend/src/pages/Compare.jsx`

## Radar chart: model profiles

- **What it is:** The radar chart compares the four core models' historical metric profiles.
- **Data:** Its axes use the same saved accuracy, F1, precision, recall and AUC values as the metrics table.
- **Calculation:** Each coloured shape plots one model's scores on five axes. The legend identifies the finance-only and text-feature variants.
- **Interpretation:** A bigger shape is a visual summary, not a statistical test. Metrics have different meanings, and the chart does not establish better future portfolio returns.
- **Code / evidence:** `frontend/src/pages/Compare.jsx`

## Per-ticker model comparison

- **What it is:** The ticker selector compares the four core models' saved decisions for one instrument.
- **Data:** Selecting AAPL calls the ticker-comparison API for the saved model signals, probabilities and variant differences.
- **Calculation:** Each model card shows its BUY/SELL label and historical P(UP). The comparison lets the user inspect disagreement hidden by an overall vote.
- **Interpretation:** This uses saved historical outputs. It does not retrain the models or produce four current live predictions. A difference between text and finance variants does not prove news caused a gain.
- **Code / evidence:** `frontend/src/pages/Compare.jsx`

## Model leaderboard: all trained variants

- **What it is:** The leaderboard ranks six model families, their two input variants and the four-core-model probability ensemble: 13 entries.
- **Data:** The metrics API reads the stored model results. Column headers allow sorting by accuracy, F1, precision, recall or AUC.
- **Calculation:** Sorting changes which model leads for the chosen metric. GRU Finance has the highest saved classification accuracy, 54.56%, in this file.
- **Interpretation:** This is a model leaderboard, separate from player scores. Its static 50% reference is only a visual guide; always-UP and training-prior baselines are more informative for an imbalanced target.
- **Code / evidence:** `frontend/src/pages/Leaderboard.jsx`

## Backtesting: settings and net performance

- **What it is:** Backtesting simulates a clearly specified trading policy using the saved model predictions.
- **Data:** The report combines saved predictions and historical Open/Close prices for 44 instruments, from 22 February 2023 to 20 December 2024: 462 sessions.
- **Calculation:** Starting capital is $100,000. Default costs are 10 bps commission plus 5 bps slippage per side. Selecting a strategy loads its saved results and matching benchmarks.
- **Interpretation:** The default ensemble shows 61.98% net return, Sharpe 1.44 and maximum drawdown -17.80%. These are exploratory simulations after costs, not a verified live trading advantage.
- **Code / evidence:** `backend/data/backtests/report.json`

## Backtesting: growth of capital

- **What it is:** The equity chart shows what the simulated account was worth at each daily close.
- **Data:** Daily values are saved by the backtest engine. Selected strategy, SPY buy-and-hold and the equal-weight universe use the same observed dates.
- **Calculation:** Equity equals cash plus marked holding value. Shares remain invested between rebalances, so opening gaps and weekend changes are included. Costs reduce cash.
- **Interpretation:** A higher final value can come with larger losses along the way. Charts show historical simulation; they do not represent money invested by the project.
- **Code / evidence:** `backend/backtesting/engine.py`

## Backtesting: drawdown and downside risk

- **What it is:** Drawdown measures how far the account fell below its previous peak.
- **Data:** The engine derives drawdown, volatility, Sortino and Calmar from the continuous daily equity and net-return series.
- **Calculation:** Drawdown is equity / previous peak - 1. The ensemble's worst decline is 17.80%, with a longest underwater period of 127 sessions. Initial capital is included as a peak.
- **Interpretation:** Sharpe and downside ratios depend on the period and the explicit 0% risk-free assumption. The ensemble has more total return than SPY here but a lower Sharpe and worse drawdown.
- **Code / evidence:** `backend/backtesting/engine.py`

## Backtesting: year-by-year consistency

- **What it is:** Period performance checks whether the overall result holds across different parts of the study.
- **Data:** The period report compounds the same saved daily net returns. Holdings and costs continue across year and month boundaries.
- **Calculation:** In the observed 2023 window, ensemble return is 29.40% versus SPY 20.60%. In 2024 through 20 December, it is 25.18% versus SPY 25.78%.
- **Interpretation:** These are partial observed years, not annualised returns. The ensemble beat SPY in 9 of 23 months; a strong total return does not mean it won consistently.
- **Code / evidence:** `docs/BACKTEST_PERIOD_CONSISTENCY.md`

## Backtesting: monthly breakdown

- **What it is:** The expanded monthly table helps identify good and weak periods instead of relying only on one final score.
- **Data:** Monthly returns come from the period report built from continuous daily equity. The table contains 23 observed months; this screenshot shows its first section.
- **Calculation:** The net return is the product of (1 + daily return) minus 1. The best ensemble month is May 2023, +12.23%; the worst is April 2024, -8.93%.
- **Interpretation:** Month returns compound; they should not be added. Partial first and last months must be read using their observed dates. This is descriptive evidence, not a significance test.
- **Code / evidence:** `backend/backtesting/periods.py`

## Backtesting: compare all strategies

- **What it is:** The strategy table compares 12 individual model variants, the core-model ensemble and two benchmarks.
- **Data:** All 15 portfolios come from the same saved run, with matching evaluation dates and transaction-cost accounting.
- **Calculation:** Columns show net return, CAGR, Sharpe, Sortino, maximum drawdown and exposure. Selecting a strategy changes its charts. More metrics can be viewed by scrolling.
- **Interpretation:** GRU Finance has the strongest historical return in this run, but selecting it after seeing results would introduce selection bias. Strategy rankings do not prove future performance.
- **Code / evidence:** `backend/data/backtests/report.json`

## Backtesting: classification is separate

- **What it is:** This card measures the models' five-session direction predictions, rather than the simulated portfolio's returns.
- **Data:** The engine aligns 20,372 common ticker-days and compares model decisions with five-session outcome labels.
- **Calculation:** The ensemble has 54.52% accuracy, while always predicting UP has 54.71% on the same rows. Balanced accuracy, macro F1, AUC, MCC, Brier and BUY rate add context.
- **Interpretation:** High BUY recall can result from predicting BUY almost all the time. Neither accuracy nor game score can be substituted for Sharpe or a net return.
- **Code / evidence:** `backend/backtesting/engine.py`

## Backtesting: random selection and trading costs

- **What it is:** These checks ask whether selection is better than a matched random policy and how sensitive returns are to costs.
- **Data:** The run includes 1,000 random portfolios matched to the ensemble's schedule, position count and exposure, plus fixed cost scenarios.
- **Calculation:** The ensemble's Sharpe is at the 97.50th random percentile. Net return falls from 86.03% at zero cost to 17.25% at 50 bps per side.
- **Interpretation:** Both panels always refer to the ensemble. The random test does not remove reused-history bias. Cost scenarios are assumptions, not measured broker spreads.
- **Code / evidence:** `backend/data/backtests/report.json`

## Backtesting: download the audit evidence

- **What it is:** Downloads make the simulation easier to reproduce and defend during an FYP discussion.
- **Data:** The published run stores metrics CSV, per-strategy daily values and trades CSV, the full report JSON and the period report JSON.
- **Calculation:** Trade rows include signal and execution dates, ticker, quantity, price, traded amount, commission and slippage. Source hashes bind the report to its inputs.
- **Interpretation:** Downloads refer to a saved run. The browser does not retrain models or silently rerun the simulation. Changed or missing sources are flagged instead of being mixed with old results.
- **Code / evidence:** `docs/BACKTESTING.md`

## Live Track Record: waiting for real outcomes

- **What it is:** The forward tracker records forecasts before the next market open and checks them after five NYSE sessions.
- **Data:** The API reads live_predictions_v2 in Supabase. The captured window contains 44 logged predictions: 44 pending, zero resolved.
- **Calculation:** Once the exact target close is available, the outcome is UP if target close exceeds entry close; otherwise DOWN. Accuracy uses resolved records only, with a matching always-UP baseline and Brier.
- **Interpretation:** There is no live accuracy result yet. The 28 September forecasts target 5 October's US close. Overlapping five-session outcomes are dependent. A transient connection failure recovered during capture.
- **Code / evidence:** `backend/prediction_tracker.py`

## Live Track Record: dated prediction records

- **What it is:** The recent-prediction table makes each forecast and its timing visible.
- **Data:** Supabase stores the original data session, recording time, target session, direction and probabilities, along with model/input versions and source times.
- **Calculation:** The server inserts a forecast only once for its ticker, session and protocol. It resolves against the exact fifth-session close when available. Legacy records with unknown horizons are excluded.
- **Interpretation:** These rows are still pending. Frozen records support an audit, but database administrators remain trusted. Recording a forecast is not proof that a scheduled job or its later outcome resolution always succeeds.
- **Code / evidence:** `backend/prediction_tracker.py`

## Portfolio: add a holding

- **What it is:** The tracker lets a user enter a ticker, number of shares and purchase price.
- **Data:** Inputs are entered by the user. Guest holdings stay in browser localStorage; signed-in holdings use the owner's Supabase portfolio_holdings rows.
- **Calculation:** The form validates positive shares and price, then stores the holding. The example uses 10 AAPL shares at $200 in an isolated guest session.
- **Interpretation:** These inputs are a documentation example, not the owner's real portfolio. The project records holdings but does not place broker orders or verify a purchase.
- **Code / evidence:** `frontend/src/pages/Portfolio.jsx`

## Portfolio: value, profit and model recommendation

- **What it is:** Holdings show their current value, unrealised profit/loss and a rule-based recommendation.
- **Data:** Yahoo supplies a quote where available; otherwise the API marks the last saved historical price as a fallback. Model recommendations still use saved historical signals.
- **Calculation:** Value = shares x quote. Profit/loss = shares x (quote - buy price). Return divides this profit/loss by the purchase cost. Recommendations use consensus, confidence and holding return.
- **Interpretation:** Today's P&L currently estimates current value x reported daily percentage change; it is an approximation. Quote values can change, and combining current prices with old signals does not make the signals current.
- **Code / evidence:** `frontend/src/pages/Portfolio.jsx`

## Portfolio: suggested instruments

- **What it is:** The suggestion list highlights historical BUY signals outside the user's current holdings.
- **Data:** The explanation API supplies the same saved four-model consensus used elsewhere. Already-held tickers are excluded.
- **Calculation:** The list prioritises instruments where all four core models agree on BUY. Users can open Stock Detail to inspect the evidence before drawing conclusions.
- **Interpretation:** The list is based on old model outputs. It is not personalised investment advice, a portfolio optimiser or a verified list of stocks that will rise.
- **Code / evidence:** `frontend/src/pages/Portfolio.jsx`

## Prediction game: player leaderboard

- **What it is:** The game leaderboard displays public player scores. It is different from the model-performance leaderboard.
- **Data:** The app reads the public ranking fields from Supabase game_progress. Names shown here are public aliases, not private email addresses.
- **Calculation:** Players are ranked by their saved high score. The displayed accuracy measures how often each player's answers matched the selected model's saved labels.
- **Interpretation:** A high score measures quiz progress, not profit or future market prediction. Account progress is client-synced; this is an educational leaderboard, not a tamper-proof competition.
- **Code / evidence:** `frontend/src/pages/Game.jsx`

## Prediction game: choose the challenge

- **What it is:** The game teaches users to interpret stock indicators by guessing a model's saved BUY/SELL decision.
- **Data:** Questions come from the saved core-model predictions and dated historical indicator rows. A separate public player leaderboard uses game_progress.
- **Calculation:** Choose Easy, Medium or Hard and one of the four core models. Difficulty changes the confidence range used when sampling questions. B and S keyboard shortcuts also answer questions.
- **Interpretation:** The game evaluates agreement with a model label, not the actual future market move. High game accuracy is not evidence that either the player or model predicts returns well.
- **Code / evidence:** `frontend/src/pages/Game.jsx`

## Prediction game: a real question

- **What it is:** A question shows a random ticker and historical date, a close price, RSI and MACD hints.
- **Data:** The question API chooses an eligible saved model row and reads the matching stock indicators for that date.
- **Calculation:** The user selects BUY or SELL before the model label is revealed. The score, streak, level and answered count reflect guest or signed-in game progress.
- **Interpretation:** This is an educational quiz. The displayed date is historical, and answering does not execute a trade or insert a live forecast.
- **Code / evidence:** `backend/main.py`

## Prediction game: result and explanation

- **What it is:** The result reveals whether the guess matched the selected model and offers short indicator-based reasons.
- **Data:** The answer API compares the guess with the saved model prediction. Guest progress is stored in the temporary browser; signed-in progress syncs to Supabase.
- **Calculation:** A matching guess earns base points plus a difficulty bonus. Game accuracy is matching answers / answered questions. Recent results and high scores summarise play.
- **Interpretation:** The interface's Actual signal means the model's label in this game, not the realised stock return. The shown guest round was answered for documentation only.
- **Code / evidence:** `backend/main.py`

## Settings: account and game preferences

- **What it is:** Settings lets users manage the account view and choose defaults for the next game round.
- **Data:** Game preferences are saved to this browser. Account identity, password and sign-out actions use Supabase Auth and profile fields.
- **Calculation:** Controls set sound, default difficulty and default model. Signed-in account controls include display name, password change and sign-out options; the screenshot shows guest mode.
- **Interpretation:** Authenticated account changes were reviewed in code, not submitted for this guide. They require a working Supabase session and the expected database fields.
- **Code / evidence:** `frontend/src/pages/Settings.jsx`

## Settings: accessibility, exports and reset

- **What it is:** Reduce motion limits game animations. Exports help users keep a copy of their own holdings and game history.
- **Data:** The export buttons read the current guest data or signed-in account rows. The reset action targets guest Portfolio and Game keys in this browser.
- **Calculation:** Export buttons create CSV files. Clear local guest-mode data uses a confirmation step and removes local guest records; synced account records are separate.
- **Interpretation:** Exports describe recorded app data, not verified broker activity. Local guest storage is browser-specific and can be lost when browser data is cleared.
- **Code / evidence:** `frontend/src/pages/Settings.jsx`

## Optional sign-in and account creation

- **What it is:** Users can browse as guests or sign in to sync Portfolio and Game progress across devices.
- **Data:** Supabase Auth handles email/password and Google sign-in. The frontend uses the public project client configuration.
- **Calculation:** The form signs in an existing user. Create one switches to account registration. After authentication, app data uses owner-scoped database access.
- **Interpretation:** No account was created or signed into for these screenshots. Google redirects and email confirmation depend on the project's Supabase Auth configuration.
- **Code / evidence:** `frontend/src/context/AuthContext.jsx`

## Account registration form

- **What it is:** The registration view collects the details required to create an optional synced account.
- **Data:** The form sends registration to Supabase Auth when a user submits it. The guide captures only the form switch, without submission.
- **Calculation:** The user enters credentials and follows any required email confirmation. Returning to Sign In allows access after account creation.
- **Interpretation:** This guide does not verify a new account or its email delivery. Guest mode remains useful for a local demonstration without account changes.
- **Code / evidence:** `frontend/src/pages/Login.jsx`

## AI Assistant: explain the project's evidence

- **What it is:** The floating assistant answers questions about indicators, saved stock signals, model results and project evidence.
- **Data:** The backend calls Groq, using its configured chat model, and exposes bounded read-only project tools. The current default is openai/gpt-oss-120b.
- **Calculation:** The assistant can retrieve dated stock data before writing an answer. This captured real response describes AAPL's December 2024 signal and explains that confidence is not accuracy.
- **Interpretation:** Generated text can still be wrong. Check dates and evidence. It needs internet, a valid backend Groq key and provider availability. The widget supports Stop, retry and Clear.
- **Code / evidence:** `backend/chatbot_engine.py`
