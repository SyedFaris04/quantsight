"""Build the illustrated project guide from isolated real-browser captures.

Run with the documentation-tools Python, which contains reportlab and Pillow.
Screenshots are prepared by backend/research/capture_project_guide.py.
This script does not alter app data, models or account records.
"""
from pathlib import Path
import argparse
import json
import math
from html import escape

from PIL import Image as PILImage
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor, Color
from reportlab.lib.utils import ImageReader
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[2]
SHOTS = ROOT / 'backend/data/research/documentation_20261005'
SCRATCH = ROOT / 'tmp/pdfs/project_guide'
OUTPUT = ROOT / 'output/pdf/QuantSight_Project_Guide_2026-10-05.pdf'
VERSION = '8bd515acbc006ebafbdd69037e2c8dbaccab2744'
REPO = 'https://github.com/SyedFaris04/quantsight/blob/' + VERSION + '/'
W, H = 841.89, 595.28  # A4 landscape
NAVY, INK, BLUE, TEAL = map(HexColor, ['#101d36', '#26364c', '#325ddd', '#087f8c'])
MUTED, PALE, LINE = map(HexColor, ['#596a80', '#f0f4fa', '#d9e2ef'])

PAGES = []


def add(part, title, image, y, height, what, data, calculation, limit, source, x=0, width=804):
    PAGES.append(dict(part=part, title=title, image=image, crop=[x, y, width, height],
                      what=what, data=data, calculation=calculation, limit=limit, source=source))


add('01 / Dashboard and market', 'Dashboard: the saved model snapshot', 'dashboard', 25, 630,
    'The home page gives a quick view of the research system and links to its evidence.',
    'The API reads the latest saved prediction row for each of 44 stocks and ETFs. The displayed signal date is 20 December 2024.',
    'The four core models vote on direction. More BUY votes give BUY; more SELL votes give SELL; a tie gives HOLD. The cards count these outcomes.',
    'Mean confidence is an average model score. It is not the measured accuracy of the ensemble or a current market signal.',
    'frontend/src/pages/Dashboard.jsx')
add('01 / Dashboard and market', 'Dashboard: historical BUY candidates', 'dashboard', 813, 575,
    'This list helps the user find instruments to inspect more closely.',
    'The same dated, four-model prediction snapshot supplies the candidates and their individual scores.',
    'The page keeps overall BUY results, then ranks them by model agreement and confidence. Selecting a ticker opens Stock Detail.',
    'Agreement means the models chose the same direction. Models can agree and still be wrong. These are saved research outputs.',
    'backend/copilot_engine.py', x=0, width=407)
add('01 / Dashboard and market', 'Dashboard: saved news sentiment', 'dashboard', 826, 465,
    'Positive, neutral and negative shares describe the language in the saved headlines.',
    'The GDELT archive is scored using VADER. This view covers 1,032 unique headlines in its recent saved 14-day window, ending 23 April 2026.',
    'The server deduplicates archived headlines, groups their polarity labels and calculates each group as a share of the total. Daily values are expandable.',
    'This is an archive, not a live feed. Its 2026 news is separate from the models\' 2024 signal date. Language sentiment is not return accuracy.',
    'backend/dashboard_evidence.py', x=407, width=397)
add('01 / Dashboard and market', 'Dashboard: archived headline links', 'dashboard', 1405, 533,
    'The news list lets the user inspect the underlying headline, ticker, date and publisher.',
    'These are stored GDELT entries returned by the market-news API. The page links to the original publishers.',
    'The API sorts saved entries by date and returns a bounded list. Clicking a headline opens its publisher in a new tab.',
    'A ticker match can be noisy, and publishers may change their pages. Saved headlines do not become inputs to a past prediction simply by appearing here.',
    'backend/main.py')
add('01 / Dashboard and market', 'Market: scan and filter instruments', 'market', 0, 650,
    'The market table compares the four core model signals for each instrument.',
    'Historical mode reads the saved prediction CSVs and the explanation API. The default rows are from the December 2024 snapshot.',
    'Search by ticker; filter by BUY, SELL or HOLD and agreement; sort the results. Overall direction comes from model votes. Risk uses indicator rules.',
    'Historical confidence columns contain calibrated P(UP), including on SELL rows. The displayed risk categories are rules, not estimated loss probabilities.',
    'frontend/src/pages/Overview.jsx')
add('01 / Dashboard and market', 'Market: current daily inference', 'market_live', 0, 650,
    'Live mode requests a new finance-only XGBoost prediction using a completed daily market session.',
    'Yahoo Finance supplies adjusted daily prices for the required instrument and sector universe. The backend applies the saved model bundle.',
    'It checks the NYSE calendar, excludes unfinished bars, recreates the 30 finance inputs, scales them and applies the saved calibrator. The target is five sessions later.',
    'This is daily inference, not an intraday forecast. Missing or stale required inputs produce an unavailable result. Historical four-model votes are not used in live mode.',
    'backend/live_signals.py')

add('02 / Stock analysis and explainable AI', 'Stock Detail: signal and navigation', 'detail_AAPL', 0, 350,
    'Stock Detail brings together the signal, charts, explanations, news, history and learning tools for one ticker.',
    'The stock and explanation APIs read the saved price features and four core model outputs for AAPL in this example.',
    'The overview combines model votes and their average P(UP). Tabs switch between Overview, Technical, Sentiment & Emotion, History and Learn.',
    'The overall vote can differ from one individual model. The headline confidence is an average score, not the probability that buying the stock will make money.',
    'frontend/src/pages/Detail.jsx')
add('02 / Stock analysis and explainable AI', 'Price chart: candles and volume', 'detail_AAPL', 351, 535,
    'Each candle shows one session\'s open, high, low and close. The bars below show volume.',
    'The stock API reads the saved OHLCV price panel. Period controls request different lengths of this daily history.',
    'The candle body runs from open to close; the wick shows the high and low. Colour shows whether that day closed above or below its open.',
    'Green candles are observed price moves, not BUY predictions. This historical chart ends in December 2024 and does not show intraday candles.',
    'frontend/src/components/CandlestickChart.jsx', x=24, width=496)
add('02 / Stock analysis and explainable AI', 'RSI chart and saved signal markers', 'detail_chart_rsi', 351, 515,
    'The RSI view compares recent upward and downward price moves on a scale from 0 to 100.',
    'RSI values come from the saved feature panel. The chosen model\'s saved BUY/SELL dates supply the optional markers.',
    'RSI uses a 14-session lookback. Guide lines at 30 and 70 help identify low and high momentum readings; controls choose the period and model overlay.',
    'An RSI threshold is a rule of thumb. A high or low value does not guarantee a reversal, and markers are model decisions rather than executed trades.',
    'frontend/src/pages/Detail.jsx', x=24, width=496)
add('02 / Stock analysis and explainable AI', 'MACD chart and model overlay', 'detail_chart_macd', 351, 515,
    'MACD shows the difference between shorter and longer exponential moving averages.',
    'The saved price features provide MACD. Signal markers are read from the selected core model\'s historical predictions.',
    'MACD compares the 12- and 26-session averages. Its 9-session signal line is used by the indicator rules; the chart displays MACD over the selected period.',
    'A bullish indicator reading does not force a model to predict BUY. The model considers several inputs, while the chart shows one indicator.',
    'backend/technical_indicators.py', x=24, width=496)
add('02 / Stock analysis and explainable AI', 'Indicator scores and model votes', 'detail_AAPL', 351, 790,
    'The summary separates technical, sentiment and emotion assessments from the four saved model signals.',
    'The explanation API reads the latest historical feature row and each core model\'s output.',
    'Indicator rules produce the summary scores. The model section shows XGBoost Finance, XGBoost +Sentiment, LSTM Finance and LSTM +Sentiment.',
    'These indicator scores are not SHAP contributions. Historical text values are zero in the current test period. Differences between variants do not prove a news benefit.',
    'backend/copilot_engine.py', x=535, width=245)
add('02 / Stock analysis and explainable AI', 'Explainable AI: the exact saved prediction', 'detail_AAPL', 1160, 320,
    'The attribution panel explains one saved XGBoost output. Finance and Finance +Sentiment can be selected.',
    'The backend loads the exact saved model bundle and ticker/date input row, then verifies the prediction against the saved CSV.',
    'The panel separates raw P(UP), calibrated P(UP) and the raw score. For AAPL Finance, raw P(UP) is 46.49%, calibrated P(UP) is 55.19%, and direction remains SELL.',
    'Direction uses the raw 50% threshold. Calibration can cross 50% without changing that saved label. This panel explains a historical XGBoost output, not the live model or consensus.',
    'backend/feature_attribution.py')
add('02 / Stock analysis and explainable AI', 'Explainable AI: feature contribution bars', 'detail_AAPL', 1440, 605,
    'The bars show which inputs pushed the tree model\'s raw score toward UP or DOWN.',
    'Native XGBoost TreeSHAP calculates contributions from the saved trees after applying the bundle\'s feature order and scaler.',
    'The chart shows the eight largest absolute contributions and combines the remaining inputs. Baseline plus all contributions reconstructs the raw log-odds score.',
    'Contributions are log-odds, not percentage points or causal effects on stock prices. A zero-valued input can still have a contribution in a tree model.',
    'backend/feature_attribution.py')
add('02 / Stock analysis and explainable AI', 'Explainable AI: inputs and verification', 'detail_attribution_expanded', 2030, 625,
    'The expanded table lets the user inspect the actual input values, scaled values and contributions.',
    'The table is built from the exact row used for this saved prediction. It contains all 30 finance inputs; the screenshot shows its first section.',
    'The backend checks raw-score reconstruction, raw probability, calibrated probability and saved direction. Model and input hashes identify the evidence version.',
    'A verification failure produces an unavailable explanation. Matching the saved prediction proves consistency, not that the prediction is correct or that the model has a fresh trading edge.',
    'backend/feature_attribution.py')
add('02 / Stock analysis and explainable AI', 'Bull case and bear case', 'detail_AAPL', 2090, 575,
    'This panel presents evidence for both directions and scenarios that could change an indicator assessment or model vote count.',
    'The explanation engine combines saved RSI, MACD, price trend, historical accuracy and the core models\' votes.',
    'Rules describe supportive and opposing indicators. Reliability uses historical performance. The scenario section shows relevant indicator thresholds or vote changes.',
    'These are rule-based summaries. Scenarios do not rerun the trained predictor or guarantee that its forecast would change. Historical reliability may not carry into a new period.',
    'backend/copilot_engine.py')
add('02 / Stock analysis and explainable AI', 'Indicator and vote scenarios', 'detail_AAPL', 2650, 196,
    'Scenarios describe a threshold or vote change that would alter an indicator assessment or the overall model consensus.',
    'The explanation engine uses the saved indicator values and the four models\' direction votes.',
    'The displayed conditions identify a change to RSI, momentum, trend or the voting balance. They give the user a simple way to inspect sensitivity.',
    'These conditions do not rerun the trained predictor. They are not verified counterfactual predictions and do not guarantee a change in its raw or calibrated score.',
    'backend/copilot_engine.py')
add('02 / Stock analysis and explainable AI', 'Risk assessment', 'detail_AAPL', 2845, 300,
    'The risk card gives a simple warning level for the current saved stock analysis.',
    'It uses saved price indicators, model disagreement and confidence from the explanation engine.',
    'A fixed rule-based score maps the evidence to Low, Medium or High and lists contributing factors. It helps the user inspect uncertainty.',
    'This is not portfolio VaR, a probability of losing money or a personalised suitability check. Backtesting has separate measured volatility and drawdown metrics.',
    'backend/copilot_engine.py')
add('02 / Stock analysis and explainable AI', 'Technical tab: indicator explanations', 'detail_technical', 232, 390,
    'The technical cards describe RSI, MACD, Bollinger Bands and the moving-average trend in simple terms.',
    'The saved finance feature row supplies each numeric value. The explanation engine supplies the accompanying text.',
    'Rules compare RSI with thresholds, MACD with its signal line, price with its bands, and price/SMA10 with SMA50.',
    'The words Bullish and Bearish describe indicator rules. They are not independent trained models and do not guarantee the stock\'s next move.',
    'backend/copilot_engine.py')
add('02 / Stock analysis and explainable AI', 'Technical tab: LSTM attention', 'detail_technical', 615, 428,
    'Attention weights show how the LSTM layer distributes weight across its ten input sessions.',
    'These weights are saved with the historical LSTM prediction and returned by the stock-analysis data.',
    'The ten weights add to approximately 100%. The chart scales bars relative to one another so small differences are visible.',
    'For AAPL, all weights are near the uniform 10% level. Attention is internal weighting, not exact reasoning, causal importance or TreeSHAP.',
    'frontend/src/pages/Detail.jsx')
add('02 / Stock analysis and explainable AI', 'Sentiment tab: language scores and variants', 'detail_sentiment', 232, 553,
    'This view separates saved news and social sentiment from the differences between finance-only and text-feature models.',
    'GDELT and Reddit/WSB supply the original text sources. VADER scores language polarity, and daily aggregates are joined to the feature panel.',
    'The page displays available polarity scores and compares the saved model variants\' P(UP) values. A difference is a descriptive model-output difference.',
    'Current historical text inputs are zero on the latest prediction date. Labels such as improvement or impact in this older view should not be interpreted as proven sentiment benefit.',
    'frontend/src/pages/Detail.jsx')
add('02 / Stock analysis and explainable AI', 'Sentiment tab: emotion features', 'detail_sentiment', 779, 161,
    'Emotion features describe feelings expressed in social posts, beyond positive or negative sentiment.',
    'A pretrained GoEmotions-based RoBERTa model scores WSB posts. The project aggregates six groups: fear, optimism, anger, excitement, confusion and disappointment.',
    'Post scores are aggregated by ticker and day for the text-feature variants. The interface shows no emotion chart when the latest date has no WSB coverage.',
    'AAPL has no matching WSB posts on its latest saved prediction date. Missing coverage is not evidence that traders felt neutral, and emotion-model F1 is not stock prediction accuracy.',
    'backend/build_features.py')
add('02 / Stock analysis and explainable AI', 'Sentiment tab: ticker-specific news', 'detail_sentiment', 942, 603,
    'Stock Detail also lists saved headlines associated with the selected ticker.',
    'The news API reads the GDELT archive and returns a limited set of ticker matches, dates and publisher links.',
    'Entries are filtered by ticker and ordered by saved date. The user can open a publisher to check the full context.',
    'The AAPL headlines shown here are from April 2026. They cannot explain a December 2024 forecast, and multilingual or ambiguous matches require care.',
    'backend/main.py')
add('02 / Stock analysis and explainable AI', 'History tab: per-stock accuracy', 'detail_history', 231, 318,
    'This shows how often each core model matched the historical five-session outcome for this particular stock.',
    'The accuracy-history API compares the saved prediction labels with the stored historical target labels for that ticker.',
    'Correct divided by evaluated predictions gives accuracy. The page shows a recent 20-prediction sample as well as each model\'s longer saved history.',
    'A small recent sample can look unusually strong. This reused historical window is separate from the forward Live Track Record, and overlapping targets are dependent.',
    'backend/copilot_engine.py')
add('02 / Stock analysis and explainable AI', 'History tab: decision timeline', 'detail_history', 541, 552,
    'The timeline shows how the saved signal changed over the last seven available sessions.',
    'The history API reads dated core-model outputs and the corresponding historical indicator rows.',
    'It displays one dated signal card per session and a short indicator summary. The user can compare changes in direction and scores.',
    'This is a history of model decisions, not a record of real trades. Its confidence labels are historical P(UP), including on SELL decisions.',
    'backend/copilot_engine.py')
add('02 / Stock analysis and explainable AI', 'Learn tab: the indicator glossary', 'detail_learn_open', 232, 610,
    'The glossary helps a new user understand the terms used across Stock Detail.',
    'Definitions are written in the frontend. This feature does not request a price forecast or use a language-model call.',
    'Open the glossary and select a term to reveal its plain-language explanation. The example expands the decision timeline.',
    'Some older definitions say tomorrow or describe confidence loosely. The actual system target is five sessions, and historical confidence is P(UP). Use the current method explanation in this guide.',
    'frontend/src/pages/Detail.jsx')

add('03 / Model comparison and research', 'Finance development: compare with a baseline', 'compare', 145, 440,
    'This study asks whether three fixed finance models outperform a simple training-frequency forecast.',
    'A frozen Yahoo price snapshot supplies 44 instruments and 13 fixed finance inputs. Validation uses 2022, 2023 and 2024: 32,472 ticker-days.',
    'Logistic regression, histogram boosting and XGBoost use chronological fit, separate calibration and validation periods. There are nine predictor fits and nine calibration fits.',
    'These are development years. No tested predictor beat the training-prior baseline on equal-year mean Brier loss, so none was promoted to the live system.',
    'backend/data/research_reports/development_models.json')
add('03 / Model comparison and research', 'Finance development: probability quality', 'compare', 490, 647,
    'The table compares direction accuracy and the quality of the probability forecasts on matching observations.',
    'The study JSON contains raw and calibrated versions of the three predictors plus training-prior, recent-prior and always-UP baselines.',
    'Choose an individual validation year or the equal-year mean. Brier is the mean squared probability error; smaller is better. Other columns show balance, MCC, AUC and UP rate.',
    'The training-prior baseline has mean Brier 0.250877, versus 0.251999 for raw XGBoost. A similar accuracy score does not imply a better probability forecast.',
    'backend/data/research_reports/development_models.json')
add('03 / Model comparison and research', 'Probability reliability and uncertainty', 'compare_method', 1120, 625,
    'The expanded panel checks whether predicted probabilities match observed outcomes and shows uncertainty in the study comparison.',
    'It uses the same fixed finance-development predictions. Select a candidate and year to inspect occupied probability bins.',
    'For the displayed raw logistic model in 2024, the large 50-60% bin predicts 55.02% UP and observes 54.76%. Paired, 20-session block resamples estimate uncertainty in the Brier difference.',
    'A close match in one bin does not prove overall superiority. Some bins are small. The displayed improvement interval includes zero and does not correct for selecting among several candidates.',
    'frontend/src/components/DevelopmentComparison.jsx')
add('03 / Model comparison and research', 'Confidence audit: accuracy versus coverage', 'compare_confidence', 1480, 530,
    'This diagnostic asks what is lost or gained when lower-confidence predictions are removed.',
    'It reuses the fixed development predictions. It compares nine outputs, seven confidence thresholds and four period views, without fitting new models.',
    'Confidence is max(P(UP), P(DOWN)). For every retained set, the training-prior baseline is evaluated on exactly the same rows. Coverage is retained rows divided by eligible rows.',
    'At the raw XGBoost 60% threshold, only 12.91% of rows remain and accuracy is 51.51%. Higher confidence did not guarantee better results. No cutoff is recommended or applied live.',
    'backend/data/research_reports/development_confidence.json')
add('03 / Model comparison and research', 'Controlled news study', 'compare', 1334, 620,
    'This separate experiment compares finance, news coverage, VADER and FinBERT on the same historical observations.',
    'A timestamped financial-news archive supplies historical text. The 2023 comparison has 10,780 evaluated ticker-days, 3,658 with news and coverage for 26 of 44 instruments.',
    'The study adds coverage and sentiment inputs under a fixed time-availability rule. The model selector switches between logistic regression and histogram boosting.',
    'The screenshot uses logistic regression. Its constant training-prior baseline is stronger on Brier than the tested news variants. This study is exploratory and does not update serving models.',
    'docs/research/FINBERT_STUDY.md')
add('03 / Model comparison and research', 'FinBERT versus VADER: inspect uncertainty', 'compare', 1942, 221,
    'This panel checks whether the finance-language sentiment model gives a clear improvement over the simpler VADER scorer.',
    'Both variants use the same research rows and news timing. The report stores their Brier difference and a block-bootstrap interval.',
    'A positive Brier improvement favours FinBERT. Blocks preserve some time dependence when resampling the observed period.',
    'For the displayed logistic comparison, the interval includes zero, so there is no clear improvement. Results depend on the selected predictor and do not prove better live accuracy.',
    'frontend/src/components/ResearchComparison.jsx')
add('03 / Model comparison and research', 'Historical core models: table and bar chart', 'compare', 2522, 438,
    'The historical comparison displays accuracy, F1, precision, recall and AUC for the four core variants.',
    'Saved model_metrics.json supplies these figures. The adjacent bar chart plots the same measurements side by side.',
    'A star marks the largest saved score in each column. The LSTM +Sentiment variant has 54.25% saved accuracy; finance-only LSTM has 53.75%.',
    'These are older reused-period results, not the controlled development study above. Legacy LSTM+Transformer labels refer to LSTM with attention; the separate Transformer Encoder is another model.',
    'backend/data/predictions/model_metrics.json')
add('03 / Model comparison and research', 'Historical profiles: bars, differences and radar', 'compare', 2990, 414,
    'The charts make trade-offs between the four saved core variants easier to see.',
    'All bars, variant differences and radar axes are derived from the same historical metrics file.',
    'Bars compare each metric directly. Variant differences subtract finance-only scores from text-feature scores. The radar view plots all five metrics as a profile.',
    'Chart size is not evidence of robustness. A positive variant difference cannot isolate the effect of sentiment when the evaluated historical text inputs are zero.',
    'frontend/src/pages/Compare.jsx')
add('03 / Model comparison and research', 'Historical variant differences', 'compare', 3415, 341,
    'This section shows the numerical differences between the saved finance-only and text-feature variants.',
    'The interface reads the historical metric values for XGBoost and LSTM, then compares each pair.',
    'Each difference is the text-feature score minus its finance-only counterpart. For accuracy, a difference of 0.50 means 0.50 percentage points.',
    'The page does not establish that adding news caused the difference. Different training behaviour and zero text coverage can still produce different model outputs.',
    'frontend/src/pages/Compare.jsx')
add('03 / Model comparison and research', 'Radar chart: model profiles', 'compare_ticker', 3747, 427,
    'The radar chart compares the four core models\' historical metric profiles.',
    'Its axes use the same saved accuracy, F1, precision, recall and AUC values as the metrics table.',
    'Each coloured shape plots one model\'s scores on five axes. The legend identifies the finance-only and text-feature variants.',
    'A bigger shape is a visual summary, not a statistical test. Metrics have different meanings, and the chart does not establish better future portfolio returns.',
    'frontend/src/pages/Compare.jsx')
add('03 / Model comparison and research', 'Per-ticker model comparison', 'compare_ticker', 4180, 390,
    'The ticker selector compares the four core models\' saved decisions for one instrument.',
    'Selecting AAPL calls the ticker-comparison API for the saved model signals, probabilities and variant differences.',
    'Each model card shows its BUY/SELL label and historical P(UP). The comparison lets the user inspect disagreement hidden by an overall vote.',
    'This uses saved historical outputs. It does not retrain the models or produce four current live predictions. A difference between text and finance variants does not prove news caused a gain.',
    'frontend/src/pages/Compare.jsx')
add('03 / Model comparison and research', 'Model leaderboard: all trained variants', 'leaderboard', 0, 667,
    'The leaderboard ranks six model families, their two input variants and the four-core-model probability ensemble: 13 entries.',
    'The metrics API reads the stored model results. Column headers allow sorting by accuracy, F1, precision, recall or AUC.',
    'Sorting changes which model leads for the chosen metric. GRU Finance has the highest saved classification accuracy, 54.56%, in this file.',
    'This is a model leaderboard, separate from player scores. Its static 50% reference is only a visual guide; always-UP and training-prior baselines are more informative for an imbalanced target.',
    'frontend/src/pages/Leaderboard.jsx')

add('04 / Historical backtesting', 'Backtesting: settings and net performance', 'backtesting', 355, 420,
    'Backtesting simulates a clearly specified trading policy using the saved model predictions.',
    'The report combines saved predictions and historical Open/Close prices for 44 instruments, from 22 February 2023 to 20 December 2024: 462 sessions.',
    'Starting capital is $100,000. Default costs are 10 bps commission plus 5 bps slippage per side. Selecting a strategy loads its saved results and matching benchmarks.',
    'The default ensemble shows 61.98% net return, Sharpe 1.44 and maximum drawdown -17.80%. These are exploratory simulations after costs, not a verified live trading advantage.',
    'backend/data/backtests/report.json')
add('04 / Historical backtesting', 'Backtesting: growth of capital', 'backtesting', 851, 439,
    'The equity chart shows what the simulated account was worth at each daily close.',
    'Daily values are saved by the backtest engine. Selected strategy, SPY buy-and-hold and the equal-weight universe use the same observed dates.',
    'Equity equals cash plus marked holding value. Shares remain invested between rebalances, so opening gaps and weekend changes are included. Costs reduce cash.',
    'A higher final value can come with larger losses along the way. Charts show historical simulation; they do not represent money invested by the project.',
    'backend/backtesting/engine.py')
add('04 / Historical backtesting', 'Backtesting: drawdown and downside risk', 'backtesting', 1285, 588,
    'Drawdown measures how far the account fell below its previous peak.',
    'The engine derives drawdown, volatility, Sortino and Calmar from the continuous daily equity and net-return series.',
    'Drawdown is equity / previous peak - 1. The ensemble\'s worst decline is 17.80%, with a longest underwater period of 127 sessions. Initial capital is included as a peak.',
    'Sharpe and downside ratios depend on the period and the explicit 0% risk-free assumption. The ensemble has more total return than SPY here but a lower Sharpe and worse drawdown.',
    'backend/backtesting/engine.py')
add('04 / Historical backtesting', 'Backtesting: year-by-year consistency', 'backtesting', 1889, 513,
    'Period performance checks whether the overall result holds across different parts of the study.',
    'The period report compounds the same saved daily net returns. Holdings and costs continue across year and month boundaries.',
    'In the observed 2023 window, ensemble return is 29.40% versus SPY 20.60%. In 2024 through 20 December, it is 25.18% versus SPY 25.78%.',
    'These are partial observed years, not annualised returns. The ensemble beat SPY in 9 of 23 months; a strong total return does not mean it won consistently.',
    'docs/BACKTEST_PERIOD_CONSISTENCY.md')
add('04 / Historical backtesting', 'Backtesting: monthly breakdown', 'backtest_months', 2350, 630,
    'The expanded monthly table helps identify good and weak periods instead of relying only on one final score.',
    'Monthly returns come from the period report built from continuous daily equity. The table contains 23 observed months; this screenshot shows its first section.',
    'The net return is the product of (1 + daily return) minus 1. The best ensemble month is May 2023, +12.23%; the worst is April 2024, -8.93%.',
    'Month returns compound; they should not be added. Partial first and last months must be read using their observed dates. This is descriptive evidence, not a significance test.',
    'backend/backtesting/periods.py')
add('04 / Historical backtesting', 'Backtesting: compare all strategies', 'backtesting', 2474, 642,
    'The strategy table compares 12 individual model variants, the core-model ensemble and two benchmarks.',
    'All 15 portfolios come from the same saved run, with matching evaluation dates and transaction-cost accounting.',
    'Columns show net return, CAGR, Sharpe, Sortino, maximum drawdown and exposure. Selecting a strategy changes its charts. More metrics can be viewed by scrolling.',
    'GRU Finance has the strongest historical return in this run, but selecting it after seeing results would introduce selection bias. Strategy rankings do not prove future performance.',
    'backend/data/backtests/report.json')
add('04 / Historical backtesting', 'Backtesting: classification is separate', 'backtesting', 3540, 286,
    'This card measures the models\' five-session direction predictions, rather than the simulated portfolio\'s returns.',
    'The engine aligns 20,372 common ticker-days and compares model decisions with five-session outcome labels.',
    'The ensemble has 54.52% accuracy, while always predicting UP has 54.71% on the same rows. Balanced accuracy, macro F1, AUC, MCC, Brier and BUY rate add context.',
    'High BUY recall can result from predicting BUY almost all the time. Neither accuracy nor game score can be substituted for Sharpe or a net return.',
    'backend/backtesting/engine.py')
add('04 / Historical backtesting', 'Backtesting: random selection and trading costs', 'backtesting', 3820, 397,
    'These checks ask whether selection is better than a matched random policy and how sensitive returns are to costs.',
    'The run includes 1,000 random portfolios matched to the ensemble\'s schedule, position count and exposure, plus fixed cost scenarios.',
    'The ensemble\'s Sharpe is at the 97.50th random percentile. Net return falls from 86.03% at zero cost to 17.25% at 50 bps per side.',
    'Both panels always refer to the ensemble. The random test does not remove reused-history bias. Cost scenarios are assumptions, not measured broker spreads.',
    'backend/data/backtests/report.json')
add('04 / Historical backtesting', 'Backtesting: download the audit evidence', 'backtesting', 4295, 250,
    'Downloads make the simulation easier to reproduce and defend during an FYP discussion.',
    'The published run stores metrics CSV, per-strategy daily values and trades CSV, the full report JSON and the period report JSON.',
    'Trade rows include signal and execution dates, ticker, quantity, price, traded amount, commission and slippage. Source hashes bind the report to its inputs.',
    'Downloads refer to a saved run. The browser does not retrain models or silently rerun the simulation. Changed or missing sources are flagged instead of being mixed with old results.',
    'docs/BACKTESTING.md')

add('05 / Live evidence and user tools', 'Live Track Record: waiting for real outcomes', 'track-record', 0, 457,
    'The forward tracker records forecasts before the next market open and checks them after five NYSE sessions.',
    'The API reads live_predictions_v2 in Supabase. The captured window contains 44 logged predictions: 44 pending, zero resolved.',
    'Once the exact target close is available, the outcome is UP if target close exceeds entry close; otherwise DOWN. Accuracy uses resolved records only, with a matching always-UP baseline and Brier.',
    'There is no live accuracy result yet. The 28 September forecasts target 5 October\'s US close. Overlapping five-session outcomes are dependent. A transient connection failure recovered during capture.',
    'backend/prediction_tracker.py')
add('05 / Live evidence and user tools', 'Live Track Record: dated prediction records', 'track-record', 474, 507,
    'The recent-prediction table makes each forecast and its timing visible.',
    'Supabase stores the original data session, recording time, target session, direction and probabilities, along with model/input versions and source times.',
    'The server inserts a forecast only once for its ticker, session and protocol. It resolves against the exact fifth-session close when available. Legacy records with unknown horizons are excluded.',
    'These rows are still pending. Frozen records support an audit, but database administrators remain trusted. Recording a forecast is not proof that a scheduled job or its later outcome resolution always succeeds.',
    'backend/prediction_tracker.py')
add('05 / Live evidence and user tools', 'Portfolio: add a holding', 'portfolio_form', 0, 316,
    'The tracker lets a user enter a ticker, number of shares and purchase price.',
    'Inputs are entered by the user. Guest holdings stay in browser localStorage; signed-in holdings use the owner\'s Supabase portfolio_holdings rows.',
    'The form validates positive shares and price, then stores the holding. The example uses 10 AAPL shares at $200 in an isolated guest session.',
    'These inputs are a documentation example, not the owner\'s real portfolio. The project records holdings but does not place broker orders or verify a purchase.',
    'frontend/src/pages/Portfolio.jsx')
add('05 / Live evidence and user tools', 'Portfolio: value, profit and model recommendation', 'portfolio_example', 0, 624,
    'Holdings show their current value, unrealised profit/loss and a rule-based recommendation.',
    'Yahoo supplies a quote where available; otherwise the API marks the last saved historical price as a fallback. Model recommendations still use saved historical signals.',
    'Value = shares x quote. Profit/loss = shares x (quote - buy price). Return divides this profit/loss by the purchase cost. Recommendations use consensus, confidence and holding return.',
    'Today\'s P&L currently estimates current value x reported daily percentage change; it is an approximation. Quote values can change, and combining current prices with old signals does not make the signals current.',
    'frontend/src/pages/Portfolio.jsx')
add('05 / Live evidence and user tools', 'Portfolio: suggested instruments', 'portfolio_example', 611, 308,
    'The suggestion list highlights historical BUY signals outside the user\'s current holdings.',
    'The explanation API supplies the same saved four-model consensus used elsewhere. Already-held tickers are excluded.',
    'The list prioritises instruments where all four core models agree on BUY. Users can open Stock Detail to inspect the evidence before drawing conclusions.',
    'The list is based on old model outputs. It is not personalised investment advice, a portfolio optimiser or a verified list of stocks that will rise.',
    'frontend/src/pages/Portfolio.jsx')
add('05 / Live evidence and user tools', 'Prediction game: player leaderboard', 'game', 108, 510,
    'The game leaderboard displays public player scores. It is different from the model-performance leaderboard.',
    'The app reads the public ranking fields from Supabase game_progress. Names shown here are public aliases, not private email addresses.',
    'Players are ranked by their saved high score. The displayed accuracy measures how often each player\'s answers matched the selected model\'s saved labels.',
    'A high score measures quiz progress, not profit or future market prediction. Account progress is client-synced; this is an educational leaderboard, not a tamper-proof competition.',
    'frontend/src/pages/Game.jsx', x=58, width=672)
add('05 / Live evidence and user tools', 'Prediction game: choose the challenge', 'game', 642, 403,
    'The game teaches users to interpret stock indicators by guessing a model\'s saved BUY/SELL decision.',
    'Questions come from the saved core-model predictions and dated historical indicator rows. A separate public player leaderboard uses game_progress.',
    'Choose Easy, Medium or Hard and one of the four core models. Difficulty changes the confidence range used when sampling questions. B and S keyboard shortcuts also answer questions.',
    'The game evaluates agreement with a model label, not the actual future market move. High game accuracy is not evidence that either the player or model predicts returns well.',
    'frontend/src/pages/Game.jsx', x=58, width=672)
add('05 / Live evidence and user tools', 'Prediction game: a real question', 'game_question', 0, 723,
    'A question shows a random ticker and historical date, a close price, RSI and MACD hints.',
    'The question API chooses an eligible saved model row and reads the matching stock indicators for that date.',
    'The user selects BUY or SELL before the model label is revealed. The score, streak, level and answered count reflect guest or signed-in game progress.',
    'This is an educational quiz. The displayed date is historical, and answering does not execute a trade or insert a live forecast.',
    'backend/main.py')
add('05 / Live evidence and user tools', 'Prediction game: result and explanation', 'game_result', 0, 721,
    'The result reveals whether the guess matched the selected model and offers short indicator-based reasons.',
    'The answer API compares the guess with the saved model prediction. Guest progress is stored in the temporary browser; signed-in progress syncs to Supabase.',
    'A matching guess earns base points plus a difficulty bonus. Game accuracy is matching answers / answered questions. Recent results and high scores summarise play.',
    'The interface\'s Actual signal means the model\'s label in this game, not the realised stock return. The shown guest round was answered for documentation only.',
    'backend/main.py')
add('05 / Live evidence and user tools', 'Settings: account and game preferences', 'settings', 100, 518,
    'Settings lets users manage the account view and choose defaults for the next game round.',
    'Game preferences are saved to this browser. Account identity, password and sign-out actions use Supabase Auth and profile fields.',
    'Controls set sound, default difficulty and default model. Signed-in account controls include display name, password change and sign-out options; the screenshot shows guest mode.',
    'Authenticated account changes were reviewed in code, not submitted for this guide. They require a working Supabase session and the expected database fields.',
    'frontend/src/pages/Settings.jsx')
add('05 / Live evidence and user tools', 'Settings: accessibility, exports and reset', 'settings', 612, 393,
    'Reduce motion limits game animations. Exports help users keep a copy of their own holdings and game history.',
    'The export buttons read the current guest data or signed-in account rows. The reset action targets guest Portfolio and Game keys in this browser.',
    'Export buttons create CSV files. Clear local guest-mode data uses a confirmation step and removes local guest records; synced account records are separate.',
    'Exports describe recorded app data, not verified broker activity. Local guest storage is browser-specific and can be lost when browser data is cleared.',
    'frontend/src/pages/Settings.jsx')
add('05 / Live evidence and user tools', 'Optional sign-in and account creation', 'login', 85, 545,
    'Users can browse as guests or sign in to sync Portfolio and Game progress across devices.',
    'Supabase Auth handles email/password and Google sign-in. The frontend uses the public project client configuration.',
    'The form signs in an existing user. Create one switches to account registration. After authentication, app data uses owner-scoped database access.',
    'No account was created or signed into for these screenshots. Google redirects and email confirmation depend on the project\'s Supabase Auth configuration.',
    'frontend/src/context/AuthContext.jsx', x=130, width=544)
add('05 / Live evidence and user tools', 'Account registration form', 'signup', 85, 545,
    'The registration view collects the details required to create an optional synced account.',
    'The form sends registration to Supabase Auth when a user submits it. The guide captures only the form switch, without submission.',
    'The user enters credentials and follows any required email confirmation. Returning to Sign In allows access after account creation.',
    'This guide does not verify a new account or its email delivery. Guest mode remains useful for a local demonstration without account changes.',
    'frontend/src/pages/Login.jsx', x=130, width=544)
add('05 / Live evidence and user tools', 'AI Assistant: explain the project\'s evidence', 'chat_answer', 0, 544,
    'The floating assistant answers questions about indicators, saved stock signals, model results and project evidence.',
    'The backend calls Groq, using its configured chat model, and exposes bounded read-only project tools. The current default is openai/gpt-oss-120b.',
    'The assistant can retrieve dated stock data before writing an answer. This captured real response describes AAPL\'s December 2024 signal and explains that confidence is not accuracy.',
    'Generated text can still be wrong. Check dates and evidence. It needs internet, a valid backend Groq key and provider availability. The widget supports Stop, retry and Clear.',
    'backend/chatbot_engine.py', width=384)


def normal(text):
    return text.replace('\u2013', '-').replace('\u2014', '-').replace('\u2011', '-').replace('\u2019', "'")


class Guide:
    def __init__(self, output):
        self.c = canvas.Canvas(str(output), pagesize=(W, H), pageCompression=1)
        self.c.setTitle('QuantSight - Illustrated Project Guide')
        self.c.setAuthor('QuantSight project documentation')
        self.c.setSubject('Features, datasets, methods, evaluation and deployment; verified against the current project')
        self.n = 0
        self.toc = []

    def paragraph(self, text, x, top, width, size=11, color=INK, bold=False, leading=None):
        style = ParagraphStyle('p', fontName='GuideBold' if bold else 'Guide', fontSize=size,
                               leading=leading or size * 1.4, textColor=color)
        p = Paragraph(normal(text), style)
        _, height = p.wrap(width, 1000)
        p.drawOn(self.c, x, top - height)
        return top - height

    def bullet(self, text, x, top, width, size=11, color=INK):
        self.c.setFillColor(TEAL)
        self.c.circle(x + 2, top - 6, 1.8, fill=1, stroke=0)
        return self.paragraph(escape(text), x + 12, top, width - 12, size, color) - 10

    def box(self, x, y, width, height, fill=PALE, stroke=None, radius=9):
        self.c.setFillColor(fill)
        self.c.setStrokeColor(stroke or fill)
        self.c.roundRect(x, y, width, height, radius, fill=1, stroke=int(stroke is not None))

    def start(self, part, title):
        self.n += 1
        self.c.setFillColor(HexColor('#ffffff'))
        self.c.rect(0, 0, W, H, stroke=0, fill=1)
        self.c.setFillColor(NAVY)
        self.c.rect(0, H - 9, W, 9, stroke=0, fill=1)
        self.paragraph(escape(part.upper()), 32, H - 28, W - 64, 9, TEAL, True)
        self.paragraph(escape(title), 32, H - 50, W - 64, 23, NAVY, True, 28)
        key = 'page_' + str(self.n)
        self.c.bookmarkPage(key)
        self.c.addOutlineEntry(title, key, 0, False)
        self.toc.append((part, title, self.n))

    def end(self, source=None):
        self.c.setStrokeColor(LINE)
        self.c.line(32, 37, W - 32, 37)
        self.paragraph('QUANTSIGHT  /  ILLUSTRATED PROJECT GUIDE  /  05 OCT 2026', 32, 25, 600, 7.5, MUTED)
        self.c.setFillColor(MUTED)
        self.c.setFont('GuideBold', 8)
        self.c.drawRightString(W - 32, 17, f'{self.n:02d}')
        if source:
            link = f'<link href="{escape(REPO + source)}" color="#325ddd">Code / evidence: {escape(source)}</link>'
            self.paragraph(link, 32, 49, W - 64, 7.7, MUTED)
        self.c.showPage()

    def image(self, path, x, y, width, height, frame=True):
        im = PILImage.open(path)
        scale = min(width / im.width, height / im.height)
        iw, ih = im.width * scale, im.height * scale
        px, py = x + (width - iw) / 2, y + (height - ih) / 2
        if frame:
            self.box(px - 3, py - 3, iw + 6, ih + 6, LINE, radius=6)
        self.c.drawImage(ImageReader(im), px, py, iw, ih, mask='auto')

    def feature(self, entry):
        self.start(entry['part'], entry['title'])
        image_path = SHOTS / (entry['image'] + '.png')
        im = PILImage.open(image_path).convert('RGB')
        meta = json.loads((SHOTS / 'captures.json').read_text(encoding='utf-8'))[entry['image']]
        scale = im.width / meta['width']
        x, y, width, height = entry['crop']
        cropped = im.crop((round(x * scale), round(y * scale),
                           min(im.width, round((x + width) * scale)), min(im.height, round((y + height) * scale))))
        image_path = SCRATCH / ('crop_%02d.png' % self.n)
        if entry['title'] == 'Indicator scores and model votes':
            # Reflow two unaltered crops beside one another to keep this narrow column readable.
            split = round(423 * scale)
            top_half = cropped.crop((0, 0, cropped.width, split))
            bottom_half = cropped.crop((0, split, cropped.width, cropped.height))
            reflow = PILImage.new('RGB', (cropped.width * 2 + 24, split), '#101827')
            reflow.paste(top_half, (0, 0))
            reflow.paste(bottom_half, (cropped.width + 24, 0))
            cropped = reflow
        cropped.save(image_path)
        self.box(30, 78, 520, 418, HexColor('#f4f6fa'))
        self.image(image_path, 35, 85, 510, 404)
        self.paragraph('REAL LOCAL SCREENSHOT  /  GUEST SESSION', 34, 69, 516, 7.3, MUTED, True)
        self.box(570, 78, 240, 418, PALE)
        top = 481
        labels = [('WHAT IT IS', 'what'), ('WHERE DATA COMES FROM', 'data'),
                  ('HOW THE RESULT IS MADE', 'calculation'), ('HOW TO READ IT', 'limit')]
        for label, key in labels:
            top = self.paragraph(label, 584, top, 212, 8.1, TEAL, True, 11) - 6
            top = self.paragraph(escape(entry[key]), 584, top, 212, 10.4, INK, leading=14.2) - 17
        if top < 79:
            raise RuntimeError(f'Explanation overflows on page {self.n}: {entry["title"]}: {top}')
        self.end(entry['source'])


def cover(g):
    g.start('Project documentation', 'QuantSight')
    g.c.setFillColor(NAVY)
    g.c.rect(0, 0, W, H, stroke=0, fill=1)
    g.box(31, 491, 131, 25, HexColor('#243b60'), radius=5)
    g.paragraph('FINAL YEAR PROJECT', 42, 509, 120, 9, HexColor('#90d7e7'), True)
    g.paragraph('QuantSight', 32, 457, 305, 40, HexColor('#ffffff'), True, 45)
    g.paragraph('An illustrated guide to<br/>the complete system', 33, 392, 300, 24, HexColor('#ffffff'), True, 31)
    g.paragraph('Features, data, models,<br/>explainable AI and backtesting', 33, 294, 294, 15, HexColor('#c7d4e8'), leading=22)
    g.paragraph('Simple explanations beside real screenshots.<br/>Includes the workflow from datasets to deployment.', 33, 220, 285, 11, HexColor('#c7d4e8'), leading=17)
    g.image(SHOTS / 'dashboard_overview.png', 356, 121, 453, 385, frame=False)
    g.box(356, 66, 451, 43, HexColor('#243b60'))
    g.paragraph('44 instruments  /  5-session direction target  /  15 backtest portfolios', 370, 96, 423, 10.4, HexColor('#ffffff'), True)
    g.paragraph('Prepared 5 October 2026<br/>Code baseline: 8bd515a  |  Screenshots: local running system', 33, 100, 309, 9.2, HexColor('#c7d4e8'), leading=15)
    g.c.setFont('Guide', 8)
    g.c.setFillColor(HexColor('#c7d4e8'))
    g.c.drawRightString(W - 32, 24, '01')
    g.c.showPage()


def contents(g):
    g.start('Reading the guide', 'Find the feature, then follow the evidence')
    groups = []
    for entry in PAGES:
        if entry['part'] not in groups:
            groups.append(entry['part'])
    top = 474
    for part in groups:
        index = next(i for i, entry in enumerate(PAGES) if entry['part'] == part)
        count = sum(entry['part'] == part for entry in PAGES)
        g.box(32, top - 54, 490, 47, PALE)
        g.paragraph(escape(part), 46, top - 12, 396, 13, NAVY, True)
        g.paragraph(f'Pages {6 + index}-{5 + index + count}', 424, top - 15, 91, 9, BLUE, True)
        top -= 63
    g.box(32, top - 54, 490, 47, PALE)
    g.paragraph('06 / Deployment, metrics and demo', 46, top - 12, 374, 13, NAVY, True)
    g.paragraph(f'Pages {6 + len(PAGES)}-{12 + len(PAGES)}', 424, top - 15, 91, 9, BLUE, True)
    g.paragraph('Before the feature pages', 570, 476, 228, 15, NAVY, True)
    for text in ['Page 3: dataset-to-deployment graphic tree.', 'Page 4: datasets, cleaning and prediction target.', 'Page 5: model families and training methods.']:
        top_right = locals().get('top_right', 442)
        top_right = g.bullet(text, 570, top_right, 228, 11)
    top_right = g.paragraph('After the feature pages', 570, top_right - 6, 228, 15, NAVY, True) - 16
    for text in ['Deployment and service connections.', 'Backtest execution and metric cheat sheet.', 'Current limitations and a demo route.', 'Evidence references and reproducibility.']:
        top_right = g.bullet(text, 570, top_right, 228, 11)
    g.paragraph('Each feature page explains what it is, its data source, its calculation and how to interpret it.', 570, 120, 228, 10.2, MUTED)
    g.end()


def node(g, x, y, width, height, title, body, fill=PALE):
    g.box(x, y, width, height, fill, LINE)
    bottom = g.paragraph(escape(title), x + 11, y + height - 10, width - 22, 11, NAVY, True, 14) - 5
    g.paragraph(escape(body), x + 11, bottom, width - 22, 9.5, MUTED, leading=12.4)


def arrow(g, points, dashed=False):
    c = g.c
    c.setStrokeColor(TEAL)
    c.setLineWidth(1.4)
    if dashed:
        c.setDash(4, 3)
    path = c.beginPath()
    path.moveTo(*points[0])
    for p in points[1:]:
        path.lineTo(*p)
    c.drawPath(path)
    c.setDash()
    a, b = points[-2], points[-1]
    angle = math.atan2(b[1] - a[1], b[0] - a[0])
    c.setFillColor(TEAL)
    path = c.beginPath()
    path.moveTo(*b)
    for turn in [-0.45, 0.45]:
        path.lineTo(b[0] - 6 * math.cos(angle + turn), b[1] - 6 * math.sin(angle + turn))
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def workflow(g):
    g.start('System workflow', 'Graphic tree: from datasets to deployment')
    node(g, 32, 434, 243, 77, '1A. Prices and volume', 'Yahoo daily OHLCV; historical snapshot; sector ETFs and SPY.')
    node(g, 299, 434, 243, 77, '1B. News language', 'GDELT archive; timestamped historical news for separate research.')
    node(g, 566, 434, 243, 77, '1C. Social language', 'Reddit/WSB posts; VADER polarity and GoEmotions-based scores.')
    for x in [153, 420, 687]:
        arrow(g, [(x, 434), (x, 406), (420, 406), (420, 395)])
    node(g, 147, 324, 545, 71, '2. Prepare daily datasets', 'Sort and check ticker/date rows; compute trailing indicators; normalise price-level inputs; aggregate text by available date; create the five-session UP/DOWN target.')
    arrow(g, [(420, 324), (420, 307), (153, 307), (153, 292)])
    node(g, 32, 207, 243, 85, '3. Fit and calibrate offline', 'Finance / text-feature variants; time-ordered splits; purge overlapping labels; fit scalers on training only.')
    arrow(g, [(275, 250), (299, 250)])
    node(g, 299, 207, 243, 85, '4. Evaluate the evidence', 'Classification and baselines; cost-aware backtests; controlled news and finance studies; forward tracker kept separate.')
    arrow(g, [(542, 250), (566, 250)])
    node(g, 566, 207, 243, 85, '5. Prepare serving assets', 'Saved bundles, prediction CSVs, compact UI snapshots, verified TreeSHAP and versioned reports.')
    arrow(g, [(687, 207), (687, 187), (420, 187), (420, 177)])
    node(g, 270, 132, 300, 45, '6. GitHub + checks', 'Version the code/assets, test and deploy.')
    for x in [153, 420]:
        arrow(g, [(420, 132), (420, 115), (x, 115), (x, 105)])
    arrow(g, [(570, 155), (687, 155), (687, 105)], dashed=True)
    node(g, 32, 51, 243, 54, 'Vercel: user interface', 'React / Vite dashboard in the browser.')
    node(g, 299, 51, 243, 54, 'Render: backend API', 'FastAPI reads assets and calls providers.')
    node(g, 566, 51, 243, 54, 'Supabase: accounts and records', 'Auth, holdings, game and live forecasts.')
    g.paragraph('Dashed branch: Supabase SQL migrations are applied separately; a git push does not run SQL.', 32, 35, 777, 8.2, MUTED)
    g.c.setFont('GuideBold', 8)
    g.c.setFillColor(MUTED)
    g.c.drawRightString(W - 32, 13, f'{g.n:02d}')
    g.c.showPage()


def data_page(g):
    g.start('Data and preparation', 'What enters the system, and what changes')
    rows = [
        ('Saved finance panel', '104,076 rows / 44 instruments', '2015-05-01 to 2024-12-20', 'Historical models, charts and backtests'),
        ('GDELT daily text panel', '2,127 daily rows', '2026-01-22 to 2026-04-23', 'Saved news display; no training overlap'),
        ('WSB text and emotion', '11,213 posts per scored source', 'Ends 2021-08-31', 'Sparse early text coverage; none in 2023-2024'),
        ('Research news archive', '70,974 source rows; 26 tickers covered', 'Timestamped historical coverage', 'Separate controlled VADER / FinBERT study'),
        ('Live Yahoo daily panel', 'Required trained universe + sector data', 'Latest eligible completed NYSE session', 'Single XGBoost finance inference'),
    ]
    top = 483
    for title, amount, dates, use in rows:
        g.box(32, top - 61, 777, 56, PALE)
        g.paragraph(escape(title), 45, top - 14, 178, 10.7, NAVY, True)
        g.paragraph(escape(amount), 230, top - 14, 270, 10.2)
        g.paragraph(escape(dates), 514, top - 14, 280, 10.2, MUTED)
        g.paragraph(escape(use), 230, top - 34, 560, 9.7, MUTED)
        top -= 65
    top = 141
    g.paragraph('Preparation in plain words', 33, top, 375, 13, NAVY, True)
    g.paragraph('Check dates, duplicates, gaps and valid prices. Calculate RSI, MACD, bands, averages, ranges, volume ratios and relative-market inputs using trailing data. Keep raw prices for display; trained price-level inputs use relative values.', 33, top - 23, 369, 10.3, leading=14)
    g.paragraph('The label to predict', 437, top, 370, 13, NAVY, True)
    g.paragraph('UP: Close five sessions later is higher than the signal-session Close. DOWN: equal or lower. Unknown future tails stay missing in new labels. Keep text availability flags before filling missing numeric values; old text joins still have timing limitations.', 437, top - 23, 370, 10.3, leading=14)
    g.end('docs/data_audit.json')


def models_page(g):
    g.start('Models and training', 'What the prediction models do')
    families = [
        ('XGBoost', 'Boosted decision trees for daily tabular inputs. The finance bundle is also used for live inference.'),
        ('Random Forest', 'Many decision trees vote together; useful as a different tree-based comparison.'),
        ('Logistic Regression', 'A simple linear probability model; a baseline against more complex methods.'),
        ('LSTM with attention', 'A recurrent sequence model using ten sessions. Attention weights its input days.'),
        ('GRU', 'Another recurrent sequence model with a simpler gated state than LSTM.'),
        ('Transformer Encoder', 'A separate self-attention sequence model. It is not the legacy LSTM+Transformer label.'),
    ]
    top = 484
    for title, body in families:
        g.box(32, top - 57, 460, 51, PALE)
        g.paragraph(title, 44, top - 14, 164, 10.8, NAVY, True)
        g.paragraph(escape(body), 205, top - 13, 272, 9.9, leading=13.5)
        top -= 61
    right = 480
    for title, text in [
        ('Two input variants', 'Each family has finance-only and finance + sentiment/emotion versions: 12 saved variants. The ensemble averages the four core XGBoost/LSTM probabilities.'),
        ('Time-ordered training', 'Legacy training uses chronological train/test dates and purged walk-forward checks. Scalers use training data only. XGBoost uses training out-of-fold outputs for isotonic or Platt calibration.'),
        ('Saved output contract', 'Historical confidence is calibrated P(UP) in percent. Legacy direction uses the raw score. Live direction confidence converts P(UP) to the selected direction.'),
        ('Research stays separate', 'The newer controlled finance study fits three fixed models on 13 ratio features. Its results do not replace the legacy serving bundle. The candidate 2025 test has not been scored here.'),
    ]:
        right = g.paragraph(title, 525, right, 280, 12, TEAL, True) - 7
        right = g.paragraph(escape(text), 525, right, 280, 10.5, leading=14.6) - 20
    g.paragraph('A model outputs a five-session direction score. It does not predict a guaranteed price, execute orders or retrain itself when a page opens.', 33, 101, 459, 10.2, MUTED)
    g.end('backend/train_xgboost.py')


def runtime_page(g):
    g.start('06 / Deployment and operation', 'How the running services connect')
    node(g, 32, 367, 229, 114, 'User browser / React', 'Vercel hosts the built Vite site. React Router handles pages; Recharts and custom candles draw the results. Axios/fetch requests the configured API.')
    node(g, 310, 367, 229, 114, 'FastAPI / Render', 'The Python server loads saved assets once. It returns stock analysis, explanations, reports and tracker summaries. Heavy training stays offline.')
    node(g, 587, 367, 221, 114, 'External providers', 'Yahoo: current prices and daily bars. Groq: chatbot answers. Supabase: authentication and durable user / prediction records.')
    arrow(g, [(261, 425), (310, 425)])
    arrow(g, [(539, 425), (587, 425)])
    node(g, 32, 182, 229, 125, 'GitHub: code and releases', 'Pushes to main trigger configured Render and Vercel deployments. Quality checks cover API serving, evaluation, database rules and frontend build/tests.')
    node(g, 310, 182, 229, 125, 'Scheduled forward forecasts', 'GitHub Actions calls a secret-protected admin endpoint. Schedule: 22:00 UTC Mon-Fri, or 06:00 Malaysia Tue-Sat. Recording obeys market-session timing.')
    node(g, 587, 182, 221, 125, 'Supabase: separate SQL setup', 'Auth users; portfolio_holdings; game_progress; live_predictions_v2. User data has owner-based rules. Backend-only forecast writes preserve frozen records.')
    arrow(g, [(146, 367), (146, 330), (697, 330), (697, 307)])
    g.paragraph('Browser also connects directly for sign-in and its own account data', 245, 349, 488, 9, TEAL)
    arrow(g, [(261, 244), (310, 244)])
    arrow(g, [(539, 244), (587, 244)])
    g.paragraph('Public addresses', 33, 143, 240, 12, NAVY, True)
    g.paragraph('<link href="https://quantsight.vercel.app" color="#325ddd">quantsight.vercel.app</link><br/><link href="https://quantsight-rikc.onrender.com/health/ready" color="#325ddd">quantsight-rikc.onrender.com</link>', 33, 121, 350, 10.5, leading=17)
    g.paragraph('Keep secrets on the backend', 426, 143, 382, 12, NAVY, True)
    g.paragraph('The Groq key, Supabase service-role key and admin-job secret belong in backend / hosting settings. The frontend uses the public Supabase client key. A git push does not apply SQL migrations or prove scheduled jobs succeeded.', 426, 121, 382, 10.3, leading=14)
    g.end('.github/workflows/daily-predictions.yml')


def execution_page(g):
    g.start('Backtest method', 'How the simulation avoids looking ahead')
    items = [
        ('1. Decide after a close', 'Use only the immediately previous session\'s saved forecast. Never select a stock using the execution day\'s signal.'),
        ('2. Trade at the next open', 'Rebalance at the first observed session of each week. Rank eligible BUY scores; exclude SPY from stock selection.'),
        ('3. Limit position size', 'Select up to five instruments. Target 1/5 of post-cost equity per slot, capped at 25%. Missing slots stay cash.'),
        ('4. Carry shares and cash', 'Hold fractional shares continuously until a recorded change. Include gaps and weekends. Mark every session at its close.'),
        ('5. Charge changed notional', 'Deduct 10 bps commission and 5 bps slippage on each buy or sell. Solve post-cost targets so the policy does not borrow.'),
        ('6. Finish and compare fairly', 'Sell remaining holdings at the last close with exit costs. SPY and equal-weight benchmarks use matching dates and cost rules.'),
    ]
    top = 483
    for title, body in items:
        g.box(32, top - 57, 777, 51, PALE)
        g.paragraph(escape(title), 46, top - 14, 231, 11.5, NAVY, True)
        g.paragraph(escape(body), 283, top - 12, 511, 10.5, leading=14.5)
        top -= 61
    g.paragraph('Audit: 15 portfolios and 462 sessions were independently replayed from their execution exports; reconstructed values matched within USD 0.000001.', 33, 108, 773, 11, TEAL)
    g.paragraph('Limits: no leverage, shorting, taxes, market impact or explicit dividend cashflows. Original adjustment and point-in-time universe provenance still need auditing.', 33, 75, 773, 10, MUTED)
    g.end('docs/BACKTESTING.md')


def metrics_page(g, classification=False):
    if classification:
        g.start('Measurement cheat sheet', 'Prediction quality: more than accuracy')
        entries = [
            ('Accuracy', 'Correct direction labels / all evaluated labels.', 'Compare with always-UP on the same rows; class imbalance matters.'),
            ('Balanced accuracy / MCC', 'Balanced accuracy averages each class\'s recall. MCC measures agreement from -1 to 1.', 'Useful when a model predicts mostly UP. MCC near zero indicates little directional agreement.'),
            ('Precision / Recall / F1', 'Precision: correct BUY / predicted BUY. Recall: detected UP / actual UP. F1 balances both.', 'Positive-class F1 differs from macro F1, which averages the two classes.'),
            ('ROC-AUC / Average precision', 'AUC measures ranking across decision thresholds. Average precision summarises UP precision-recall.', 'Neither is a realised return. AUC near 0.5 indicates weak directional ranking.'),
            ('Brier / Log loss', 'Brier = mean((P(UP) - outcome)^2). Log loss penalises confidently wrong probabilities.', 'Lower is better. They measure overall probability quality, not calibration alone.'),
            ('Calibration / Coverage', 'Calibration compares predicted probabilities with observed frequencies. Coverage is the share retained.', 'Reliability bins can be noisy. High confidence with very low coverage may give a small, unrepresentative sample.'),
        ]
    else:
        g.start('Measurement cheat sheet', 'Portfolio success: returns, risk and costs')
        entries = [
            ('Net return / CAGR', 'Net return = final equity / starting equity - 1. CAGR annualises growth using 252 sessions per year.', 'Read net costs and the observed dates. CAGR uses a trading-session convention.'),
            ('Sharpe ratio', 'Mean daily excess return / sample standard deviation of excess returns x square root of 252.', 'Higher means more return per unit of variation in this sample. Default risk-free rate is 0%.'),
            ('Sortino / Calmar', 'Sortino uses downside variation. Calmar = CAGR / absolute maximum drawdown.', 'They focus on loss variation or peak-to-trough damage; undefined denominators remain missing.'),
            ('Drawdown / Duration', 'Drawdown = equity / earlier running peak - 1. Duration counts consecutive underwater sessions.', 'More negative means a deeper loss. Long recovery periods matter even with a good final return.'),
            ('VaR95 / CVaR95', 'VaR95 is the observed daily loss threshold at the 5th return percentile. CVaR averages that lower tail.', 'These describe sample tail losses. They do not guarantee a maximum future loss.'),
            ('Exposure / Turnover / Costs', 'Exposure is invested value / equity. Turnover tracks traded notional; costs track commissions and slippage.', 'These explain how active and costly a strategy is. Positive-return days are not winning trades.'),
        ]
    top = 481
    for title, formula, reading in entries:
        g.box(32, top - 67, 777, 62, PALE)
        g.paragraph(escape(title), 44, top - 13, 163, 11, NAVY, True)
        g.paragraph(escape(formula), 210, top - 12, 309, 10.1, leading=13.4)
        g.paragraph(escape(reading), 535, top - 12, 261, 9.9, MUTED, leading=13.2)
        top -= 71
    g.paragraph('Also in the backtest report: beta, descriptive annual alpha and information ratio versus SPY. No significance claim is made from these regression quantities.', 33, 65, 777, 9.4, MUTED)
    g.end('docs/BACKTESTING.md')


def limitations_page(g):
    g.start('Current status and honest limits', 'What can be shown, and what still needs evidence')
    top = 481
    for title, text in [
        ('Working system', 'Local screenshots cover all main routes, stock tabs, TreeSHAP, research reports, backtesting, guest tools and a real chatbot reply. The connected architecture uses GitHub, Render, Vercel and Supabase.'),
        ('Live evaluation is still pending', 'The captured tracker has 44 logged and zero resolved predictions. A blank live accuracy is the correct result while outcomes are pending. One temporary database read failure recovered on retry.'),
        ('Historical results are exploratory', 'The 2023-2024 period has influenced development. Fixed surviving tickers and uncertain original adjustment metadata limit the backtest. A stronger-looking historical model is not automatically a deployable winner.'),
        ('Sentiment benefit is not established', 'Old GDELT news is dated 2026 and WSB ends in 2021; saved 2023-2024 text inputs are zero. Controlled news research has not established a robust probability advantage over simple baselines.'),
        ('A fresh final test remains separate', 'The candidate 2025 snapshot is not scored in this guide. Its eligibility as an untouched final test depends on confirming whether those results were used before. Do not tune using the final test.'),
        ('Personal and AI tools have limits', 'Portfolio daily P&L is an approximation; saved model signals can be older than a quote. Account views were reviewed without changing accounts. Chat answers require factual checks and provider access.'),
    ]:
        g.box(32, top - 63, 777, 57, PALE)
        g.paragraph(escape(title), 45, top - 13, 218, 11.2, NAVY, True)
        g.paragraph(escape(text), 279, top - 12, 516, 10.3, leading=14.1)
        top -= 68
    g.paragraph('Describe the FYP as an auditable decision-support and evaluation system. Better prediction performance remains a research question to verify, not a claim to assume.', 33, 70, 777, 10.2, TEAL)
    g.end('docs/IMPROVEMENT_ROADMAP.md')


def local_page(g):
    g.start('Demonstration and local backup', 'Run the same interface on your computer')
    g.paragraph('1. Backend - PowerShell window A', 32, 482, 380, 14, NAVY, True)
    g.box(32, 370, 777, 77, NAVY)
    g.paragraph('cd C:/Users/syeda/OneDrive/Desktop/quantsightv2/backend<br/>.venv/Scripts/python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001', 47, 430, 744, 11.5, HexColor('#ffffff'), leading=21)
    g.paragraph('2. Frontend - PowerShell window B', 32, 345, 380, 14, NAVY, True)
    g.box(32, 237, 777, 74, NAVY)
    g.paragraph('cd C:/Users/syeda/OneDrive/Desktop/quantsightv2/frontend<br/>npm run dev -- --host 127.0.0.1 --port 3000', 47, 294, 744, 11.5, HexColor('#ffffff'), leading=21)
    g.paragraph('3. Open the dashboard', 32, 212, 380, 14, NAVY, True)
    g.paragraph('<link href="http://127.0.0.1:3000" color="#325ddd">http://127.0.0.1:3000</link>  /  Frontend .env: VITE_API_URL=http://127.0.0.1:8001', 32, 186, 777, 11.5)
    g.paragraph('The local services were already running during capture. Use these commands after stopping them or restarting the computer; avoid launching duplicate servers on the same ports.', 32, 154, 777, 10.5, MUTED)
    g.paragraph('A useful demo order', 32, 111, 246, 12, NAVY, True)
    g.paragraph('Dashboard > Market > AAPL / TreeSHAP > AI Compare > Backtesting > Live Track Record > Portfolio / Game > Assistant.', 32, 90, 777, 10.6)
    g.paragraph('Local mode avoids Render startup delay. Yahoo, Groq, sign-in and Supabase records still need internet. Saved charts and backtests read local assets.', 32, 67, 777, 9.5, MUTED)
    g.end('HOW_TO_RUN.md')


def references_page(g):
    g.start('Evidence and reproducibility', 'How this guide was checked')
    g.paragraph('Project baseline', 32, 483, 250, 14, NAVY, True)
    top = 455
    for text in [
        'Documentation date: 5 October 2026, Malaysia. Code baseline: 8bd515a. Later releases may change screenshots, quotes and pending counts.',
        'Screenshots come from http://127.0.0.1:3000 with the local API on port 8001. A new guest browser was used; no private account session was captured.',
        'Screenshots are cropped for readability. The narrow model-summary column is split into two side-by-side crops. The floating launcher is hidden during section captures; the actual chat panel is shown separately.',
        'A demonstration holding and one game answer were entered only in isolated guest storage. No broker orders, account changes, new live forecasts or final-test scoring were performed.',
        'The PDF was rendered page by page for visual review. The complete feature content is also saved in a readable Markdown companion.',
    ]:
        top = g.bullet(text, 32, top, 375, 10.5)
    g.paragraph('Main implementation and evidence', 447, 483, 360, 14, NAVY, True)
    top = 455
    refs = [
        ('App routes and feature pages', 'frontend/src/App.jsx'),
        ('API and serving paths', 'backend/main.py'),
        ('Feature construction and label', 'backend/build_features.py'),
        ('Data audit and coverage', 'docs/data_audit.json'),
        ('Backtest method and formulas', 'docs/BACKTESTING.md'),
        ('Explainable AI and chatbot scope', 'docs/CHATBOT_AND_XAI.md'),
        ('Controlled finance research', 'docs/research/FINANCE_DEVELOPMENT_STUDY.md'),
        ('Confidence versus coverage', 'docs/research/CONFIDENCE_COVERAGE_AUDIT.md'),
        ('Forward forecast protocol', 'backend/prediction_tracker.py'),
        ('Deployment schedule', '.github/workflows/daily-predictions.yml'),
    ]
    for title, path in refs:
        link = f'<link href="{REPO + path}" color="#325ddd">{escape(title)}</link>'
        top = g.paragraph(link, 447, top, 360, 10.2, BLUE) - 12
    g.box(32, 61, 777, 70, PALE)
    g.paragraph('Interpretation rule', 46, 117, 180, 12, TEAL, True)
    g.paragraph('Keep four ideas separate: a language score, a model probability, observed direction accuracy and net portfolio performance. Each has its own data, date and measurement.', 46, 94, 747, 11, leading=15)
    g.end()


def markdown():
    lines = ['# QuantSight illustrated project guide', '', 'Documentation: 5 October 2026. Code baseline: `8bd515a`.', '',
             'This companion contains the feature explanations used in the PDF. Screenshots are real local UI captures from an isolated guest session.', '',
             '## System flow', '', 'Prices / news / social posts -> checked daily feature panels -> chronological training and calibration -> classification, controlled studies and backtests -> versioned serving assets -> GitHub checks -> Vercel frontend / Render API; Supabase schema is configured separately.', '']
    for entry in PAGES:
        lines += ['## ' + entry['title'], '', '- **What it is:** ' + entry['what'],
                  '- **Data:** ' + entry['data'], '- **Calculation:** ' + entry['calculation'],
                  '- **Interpretation:** ' + entry['limit'], '- **Code / evidence:** `' + entry['source'] + '`', '']
    (ROOT / 'docs/ILLUSTRATED_PROJECT_GUIDE.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    SCRATCH.mkdir(parents=True, exist_ok=True)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fonts = Path('C:/Windows/Fonts')
    pdfmetrics.registerFont(TTFont('Guide', str(fonts / 'segoeui.ttf')))
    pdfmetrics.registerFont(TTFont('GuideBold', str(fonts / 'segoeuib.ttf')))
    pdfmetrics.registerFontFamily('Guide', normal='Guide', bold='GuideBold')
    # Fail early on missing evidence or mistaken paths rather than publishing broken references.
    for entry in PAGES:
        if not (ROOT / entry['source']).is_file():
            raise FileNotFoundError(entry['source'])
        if not (SHOTS / (entry['image'] + '.png')).is_file():
            raise FileNotFoundError(entry['image'])
    guide = Guide(OUTPUT)
    cover(guide)
    contents(guide)
    workflow(guide)
    data_page(guide)
    models_page(guide)
    for entry in PAGES:
        guide.feature(entry)
    runtime_page(guide)
    execution_page(guide)
    metrics_page(guide)
    metrics_page(guide, classification=True)
    limitations_page(guide)
    local_page(guide)
    references_page(guide)
    guide.c.save()
    markdown()
    (SCRATCH / 'page_index.json').write_text(json.dumps(guide.toc, indent=2), encoding='utf-8')
    print(json.dumps({'pdf': str(OUTPUT), 'pages': guide.n, 'feature_pages': len(PAGES), 'bytes': OUTPUT.stat().st_size}))


if __name__ == '__main__':
    main()
