# QuantSight: simple project cheat sheet

- **Version of these notes:** 27 September 2026; describes the current code and saved reports.
- **Presentation and startup instructions:** [Supervisor progress and demo guide](SUPERVISOR_PROGRESS.md).

## 1. What the project does

- QuantSight is an educational stock/ETF prediction and decision-support application.
- It covers **44 US instruments**, including stocks and ETFs.
- Main question: **will the closing price be higher five trading sessions later?**
- UP label = future close is greater than the current close; DOWN label includes unchanged prices.
- Five trading sessions exclude market holidays and weekends; they are not five calendar days.
- The model predicts a direction/probability, not an exact future price.
- BUY/SELL are directional signals. A historical vote tie can produce HOLD; HOLD is not a separately trained third class.
- Historical simulation, stored predictions and newly generated live predictions are separate kinds of evidence.

## 2. How the parts connect

- **Research flow:** price/news data → cleaning and time alignment → features → model fitting/calibration → predictions → evaluation/backtesting → saved reports.
- **Website flow:** browser → React frontend → FastAPI backend → prepared files, prediction code or external services → charts and explanations.
- **GitHub:** stores source code and serving artifacts; runs automated checks and the scheduled prediction job.
- **Vercel:** hosts the React frontend at [quantsight.vercel.app](https://quantsight.vercel.app).
- **Render:** runs the FastAPI backend at [quantsight-rikc.onrender.com](https://quantsight-rikc.onrender.com).
- **Supabase:** PostgreSQL database and authentication; stores account-related data and the v2 forward-prediction log.
- Many historical features, model files, predictions and reports are **files shipped with the backend**, not database rows.
- Pushing GitHub changes triggers the configured web deployments; database schema changes require separate migrations.

## 3. Languages, libraries and services

- **Python:** data processing, machine learning, API and evaluation.
- **FastAPI + Uvicorn:** API endpoints and backend server.
- **pandas + NumPy:** data tables, time series and numerical calculations.
- **scikit-learn + XGBoost:** conventional classifiers, preprocessing and evaluation.
- **PyTorch:** LSTM, GRU and Transformer neural networks.
- **Hugging Face Transformers:** pretrained financial sentiment and emotion models.
- **JavaScript + React 18 + Vite 5:** interface and frontend development/build.
- **Tailwind CSS 3:** styling; **Recharts 2:** charts; **Axios:** API requests; **React Router 6:** page navigation.
- **Supabase:** email/Google login, database and access policies.
- **Groq:** hosts the language model used by the conversational assistant; requires internet and a backend API key.
- Backend service keys remain private; the browser uses the public Supabase client configuration.

## 4. Data and engineered inputs

- **OHLCV:** open, high, low, close and trading volume.
- **Price sources:** prepared historical price files; Yahoo Finance access for price collection/live data.
- **Original finance panel:** 104,076 rows from May 2015 to December 2024 across 44 instruments.
- **Technical features:** RSI, moving averages, MACD, Bollinger Bands, ATR, recent returns and volume changes/ratios.
- **Relative features:** normalized values and comparisons across instruments/sectors.
- Some features are order-flow proxies derived from price/volume; they are not direct exchange order-book data.
- **WSB Reddit text:** historical posts ending August 2021; used by the original sentiment/emotion pipeline.
- **Dashboard news:** a saved GDELT news archive ending 23 April 2026; not a continuously live news feed.
- **New research news:** historical Drinkall archive; 70,974 source rows; ticker coverage for 26 of the 44 instruments.
- **News availability:** conservatively use publication/archive/modification timing, then align to an eligible NYSE close; exclude fields containing future prices.
- **Fresh price preparation:** 135,545 rows, 50 instruments including six additional sector ETFs, through 2025; separate hashed development and candidate final-evaluation files.
- A file hash records which exact dataset was used; it does not prove the dataset is free of bias.

## 5. Prediction models

- **XGBoost:** many decision trees added sequentially to correct earlier errors; handles tabular indicator features.
- **Random Forest:** averages predictions from many decision trees.
- **Logistic Regression:** a simple linear probability classifier and useful comparison model.
- **LSTM with attention:** a recurrent neural network for sequences; attention pools information across input days.
- **GRU with attention:** another recurrent network with a simpler gating structure.
- **Transformer Encoder:** a sequence model using self-attention.
- Each of the six historical families has finance-only and finance-plus-sentiment variants: **12 historical variants**.
- Sequence models use **10-session input windows** in the existing training setup.
- The four core historical models are XGBoost finance, XGBoost sentiment, LSTM finance and LSTM sentiment.
- The historical four-model ensemble averages their probabilities; Dashboard consensus instead uses their directional votes.
- **Current live price prediction uses one finance-only XGBoost model**, its 30 saved input features and saved calibration.
- The controlled news study uses **Logistic Regression and Histogram Gradient Boosting** to compare feature sets fairly.
- An LSTM with attention is not the same architecture as a Transformer Encoder.

## 6. Sentiment, emotion and chat models

- **VADER:** a rule/lexicon-based method that assigns sentiment scores to text.
- **FinBERT (`ProsusAI/finbert`):** classifies financial headlines as positive, negative or neutral; its outputs become numerical news features.
- FinBERT is a text classifier, not the model that directly predicts a five-session stock return.
- **GoEmotions (`SamLowe/roberta-base-go_emotions`):** pretrained emotion classifier used for historical WSB features, including fear, optimism and excitement.
- **Chat assistant (`openai/gpt-oss-120b` through Groq, updated 28 September):** answers questions using tools that retrieve project data, then sends the answer in chunks. `GROQ_MODEL` can override the default model.
- The chat language model explains information; it is separate from the numerical stock-prediction models.
- `copilot_engine.py` builds indicator/model explanations; `chatbot_engine.py` handles the conversational assistant.

## 7. Training and evaluation techniques

- **Feature engineering:** turn raw prices/text into useful numerical inputs.
- **Scaling/normalization:** put inputs on suitable scales; learn transformations from training data for a valid experiment.
- **Sequence windows:** supply several past sessions to recurrent/Transformer models.
- **Dropout and early stopping:** reduce neural-network overfitting; class weighting/focal loss are used in existing neural training code.
- **Chronological split:** earlier dates train the model; later dates assess it.
- **Purging:** remove samples whose five-session outcome crosses into the next split, preventing boundary leakage.
- **Calibration:** map model scores to probabilities using held-out calibration data; existing code uses sigmoid/Platt or isotonic methods.
- **Walk-forward validation:** repeat training/evaluation with time moving forward. Three development folds are prepared; the new experiments are not yet completed.
- **Baseline:** a simple reference the model should improve on, such as always predicting UP or using the training UP frequency.
- **Controlled feature comparison:** keep evaluation rows consistent while comparing finance, news coverage, VADER and FinBERT.
- **Block bootstrap:** resample blocks of dates to estimate uncertainty while retaining time dependence; the news study uses 20-session blocks and the published backtest uses 10-session blocks, each with 1,000 samples.
- **Final holdout:** a genuinely unused period, opened only after choices are fixed. 2025 is a candidate whose prior use still needs confirmation.
- Some original feature/model selection used already examined evaluation results; those historical results remain exploratory.

## 8. Prediction-quality metrics

- **Accuracy:** fraction of UP/DOWN predictions that are correct; compare against the market's UP frequency.
- **Balanced accuracy:** average recall for UP and DOWN; useful when one direction is more common.
- **Precision:** among predicted UP cases, how many actually went UP.
- **Recall:** among actual UP cases, how many the model detected.
- **F1:** combines precision and recall; it does not measure investment return.
- **MCC:** combines all confusion-matrix outcomes; approximately 0 indicates no useful association, 1 perfect prediction and -1 reversed prediction.
- **ROC-AUC:** how well scores rank UP cases above DOWN cases across thresholds; 0.5 is chance-level ranking.
- **Brier loss:** average squared difference between predicted probability and actual outcome; lower is better.
- **Log loss:** probability error that strongly penalizes confident wrong predictions; lower is better.
- **Calibration:** among cases assigned around 70% UP probability, approximately 70% should go UP over enough observations.
- **Coverage/sample size:** number of eligible forecasts/news observations; small or selectively available samples weaken conclusions.
- A displayed confidence/probability is not the model's measured accuracy and is not a guarantee.

## 9. Backtesting: how it works

- Backtesting simulates a trading rule on historical data to study returns and risk.
- Use information available at the prior close; execute at the next opening price.
- Start with USD 100,000, rebalance weekly, use five slots and cap model-portfolio target weights at 25%; weights can drift between rebalances. The SPY benchmark holds SPY alone.
- Buy holdings or stay in cash; no short selling and no leverage in this protocol.
- Deduct commission of 10 basis points and slippage of 5 basis points per side: **0.15% per buy or sell**.
- Track actual shares, cash, overnight price changes and daily portfolio value; include closing liquidation costs.
- Compare 12 model variants and the four-model ensemble with SPY and an equal-weight universe over matching dates.
- Export all strategy metrics, selected daily values, selected trades or the full report.
- The published simulation covers 462 sessions in February 2023–December 2024.
- Independent ledger replay verifies accounting; it does not remove look-ahead, selection or survivorship concerns in the underlying research.
- The final five saved labels per instrument cannot be independently verified from the truncated historical price file; this limitation is disclosed in the protocol.
- These are simulated trades, not broker orders or real earned money.

## 10. Return and risk metrics

- **Net total return:** total portfolio growth after trading costs.
- **CAGR:** the equivalent compounded yearly growth rate.
- **Volatility:** variability of daily returns, annualized using 252 trading sessions.
- **Sharpe ratio:** average excess daily return divided by daily volatility, multiplied by the square root of 252. The current report assumes a 0% risk-free rate.
- **Sortino ratio:** return relative to downside variability; focuses on negative returns.
- **Maximum drawdown:** the largest percentage fall from a previous portfolio peak; a deeper negative value means a worse loss period.
- **Drawdown duration:** how long the portfolio remains below a previous peak.
- **Calmar ratio:** CAGR divided by the absolute maximum drawdown.
- **Return minus SPY:** difference in total returns, expressed in percentage points; 61.98% minus 51.69% is about 10.29 percentage points.
- **Exposure:** how much capital is invested; **turnover:** how much the portfolio trades.
- **Daily win rate:** share of days with positive return; it is not the percentage of profitable individual trades.
- **Cost sensitivity:** repeat the simulation at different trading costs to see whether performance survives.
- **Confidence interval:** a range expressing estimation uncertainty; it is not a promise of future performance.
- Higher accuracy alone does not establish a successful strategy: mistakes, trade size, costs and losses also matter.

## 11. Explanations and their limits

- The interface explains technical indicators, sentiment readings, model outputs and historical attention weights.
- Indicator scores in the Copilot are rule-based summaries; they are not automatically SHAP values.
- Offline SHAP analysis exists for feature importance; it is not the source of every displayed explanation.
- Attention weights indicate which input days receive more weight; they do not establish cause and effect.
- Existing threshold-based "what would change" explanations describe indicator/vote conditions; they are not verified model counterfactuals that guarantee a prediction flip.
- Explanation fidelity and user understanding remain evaluation tasks.

## 12. Pages: what to demonstrate

- **Dashboard:** historical snapshot dated 20 December 2024, historical candidates, saved-news date and links to evaluation evidence.
- **Market:** browse 44 instruments and four historical model signals; optional Live Mode calls the current live model.
- **Instrument details:** charts, technical indicators, historical predictions, news and explanation/attention views.
- **AI Compare:** controlled VADER/FinBERT study and older model comparisons with limitations identified.
- **Backtesting:** net performance, benchmark curves, risk metrics, method and exports; strongest page for explaining evaluation.
- **Model Leaderboard:** historical model rankings; these are different from the Game player leaderboard.
- **Track Record:** forward predictions and their eventual outcomes; currently no resolved v2 forecasts.
- **Portfolio:** tracked holdings; guest data in browser storage and signed-in data in Supabase; live prices need internet.
- **Game:** learning/quiz features and scoring; not evidence of profitable trading.
- **Login/Settings:** authentication, preferences and account-related controls.
- **AI Assistant:** tool-assisted chat; depends on the Groq service and backend configuration.

## 13. Live prediction tracking

- A new forecast records its original timestamp, model version, input vector, probability and five-session target.
- Recording is restricted to the period after a completed NYSE close plus 20 minutes and before the next opening; historical forecasts are not invented afterwards.
- The evaluator waits for the exact fifth subsequent NYSE session and refetches entry/target prices on a consistent adjustment basis.
- Missing target data leaves a prediction pending instead of marking a guessed result.
- The v2 table preserves forecast metadata separately from old predictions whose horizon was not reliable.
- Database protections reduce accidental rewriting; administrators remain trusted.
- The scheduled job is set for **22:00 UTC Monday–Friday**, equivalent to **06:00 Malaysia time the following morning**.
- Cloud reads and schema are verified; actual scheduled v2 writes and matured outcomes still need observation.
- At the 27 September check there were 0 logged/resolved v2 predictions. Do not report a live accuracy percentage yet.

## 14. Current findings and what success means

- The ensemble returned 61.98% versus SPY's 51.69% historically, but its Sharpe was lower and drawdown deeper.
- Original 2023–2024 text inputs were all zero; legacy sentiment-variant differences cannot establish a sentiment benefit.
- The controlled 2023 news study did not beat the simple baseline; no new model was promoted to live service.
- The historical test period was reused, news coverage is incomplete, and the fixed instrument universe and retrospectively adjusted prices have limitations.
- **Prediction success:** consistent baseline-relative improvement on an unused evaluation period, reasonable calibration and reported uncertainty.
- **Trading-simulation success:** acceptable risk and after-cost performance against matching benchmarks across periods, not just a high best-run return.
- **System success:** reliable pages, understandable explanations, acceptable response time and successful user tasks.
- Numeric acceptance targets should be agreed with the supervisor before final evaluation; there is no universal accuracy or Sharpe threshold that proves this project successful.

## 15. Files to know and short answers

- `frontend/src/pages/`: application screens; `frontend/src/components/`: reusable interface pieces.
- `backend/main.py`: API routes and serving-data loading.
- `backend/live_signals.py`: current live inference; `backend/prediction_tracker.py`: forward log and resolution.
- `backend/build_features.py`, `backend/train_xgboost.py`, `backend/train_lstm.py`: original feature/training pipeline.
- `backend/research/`: reproducible research/data preparation and browser verification utilities.
- `notebooks/15_backtesting.py`: entry point to regenerate the audited historical backtest; not required to run the website.
- `backend/data/`: prepared features, models, predictions and evaluation artifacts.
- `backend/supabase/002_live_predictions_v2.sql`: applied migration for the five-session forward log.
- `.github/workflows/`: automated checks and prediction scheduling.
- **"Why not just use accuracy?"** A model can be accurate by mostly saying UP, yet trade badly; assess class balance, probabilities, costs and risk too.
- **"Why keep an unsuccessful sentiment result?"** It answers the research question honestly and prevents replacing a model without evidence.
- **"Is this live?"** The website is deployed; the default Dashboard/backtest are dated historical artifacts. Live inference and prospective tracking are separate.
- **"Can it run locally?"** Yes. Two local servers provide the core saved-data demo; live data, cloud accounts and chat still need internet.
- More detail: [backtesting protocol](docs/BACKTESTING.md), [FinBERT study](docs/research/FINBERT_STUDY.md), [fresh validation plan](docs/research/FRESH_VALIDATION.md), [research sources](docs/research/RESEARCH_RECOMMENDATIONS.md).
