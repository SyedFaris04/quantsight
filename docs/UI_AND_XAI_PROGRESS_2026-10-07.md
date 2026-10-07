# QuantSight update — 7 October 2026

## What changed

- The home page now starts with an instrument search, saved signal dates and covered instruments.
- Removed the promotional gradient hero and average-confidence headline.
- Historical performance shows the model ensemble beside SPY: net return, Sharpe and maximum drawdown on the same dates.
- Forward results have a separate entry point. Historical backtests are not presented as live outcomes.
- Saved news sentiment loads only when the archive is expanded. Publication dates and archive age remain visible.
- Navigation is grouped into Workspace, Evaluation, and Learning & account.
- Replaced navigation emoji with consistent line icons. Updated neutral surfaces, restrained teal controls, spacing, table contrast and keyboard focus.
- Mobile navigation supports Escape, focus return and focus containment. Closed mobile links are hidden from keyboard navigation.

## Meaning and calculation fixes

- The old API field called `confidence` is calibrated **P(UP)** for saved models, including SELL labels. API keys remain compatible.
- P(UP) is not prediction accuracy or the probability of profit.
- Fixed the stock-detail comparison that incorrectly used `100 - P(UP)` for SELL.
- Input variant differences are now labelled as probability changes, not improvements.
- Changed visible LSTM+Transformer labels to **LSTM with attention**. The separately trained Transformer Encoder remains a different model.
- Corrected the glossary target from “tomorrow” to **five NYSE trading sessions**.
- Saved direction labels use the raw model threshold. Calibration can cross 50% without changing that saved label. The LSTM uses its saved validation-selected threshold.
- Agreement describes votes. It does not establish reliability because models can share errors.
- Historical risk levels are heuristic flags, not validated probabilities of loss.
- Removed the unvalidated portfolio BUY MORE rule and duplicate action recommendation. Holdings show saved votes and their dates.
- Corrected daily quote P&L: current value × daily return / (1 + daily return). It stays unavailable for old quotes or missing daily changes.
- The prediction quiz matches saved model labels, not actual future outcomes.
- Live direction probability and the separate research confidence audit retain their own definitions; they were not relabelled as historical P(UP).

## New XAI comparison

- Open **Model comparison → Open XAI study**.
- XGBoost Finance: exact native TreeSHAP versus LIME with a historical neighborhood.
- Attention LSTM Finance: Integrated Gradients versus day-window occlusion.
- Same saved model, date and input within each pair. Both explain raw log-odds before calibration.
- Fixed sample: 44 instruments per model, saved prediction date **20 December 2024**.
- All **88 saved predictions reproduced** before explanations were accepted.
- LSTM reconstruction uses the original export window: ten sessions before the prediction date, excluding that date. Latest window: 6–19 December 2024.
- Reference data come only from the existing finance feature file before 2023.
- Three LIME seeds, 20 random removal orders, and mean/median reference checks.
- The study protocol was committed before evaluation (`233edfc`).

| Model | Method | Mean removal response | Response above random | Mean local CPU time |
|---|---|---:|---:|---:|
| XGBoost Finance | TreeSHAP | 0.4171 | +0.1564 | 2.3 ms |
| XGBoost Finance | LIME, historical neighborhood | 0.3208 | +0.0601 | 33.4 ms |
| Attention LSTM Finance | Integrated Gradients | 0.0434 | +0.0088 | 35.8 ms |
| Attention LSTM Finance | Day-window occlusion | 0.0417 | +0.0071 | 2.0 ms |

- Removal response: mean absolute change in raw log-odds after replacing ranked inputs, cumulatively, with reference values.
- Higher response means stronger model sensitivity in this test. **It is not forecasting or trading success.**
- Compare results within each model; the two architectures have different scales and input units.
- TreeSHAP maximum reconstruction error: **5.96 × 10⁻⁷**.
- Integrated Gradients maximum integration error: **5.88 × 10⁻⁴**.
- LIME mean weighted neighborhood R²: **0.182**; mean original-input error: **0.3054 log-odds**. Its weak local fit limits trust in these explanations.
- LIME mean top-three feature overlap across seeds: **47.8%**.
- Mean top-three day overlap across mean/median references: IG **75.0%**, occlusion **70.7%**. This measures reference sensitivity, not seed stability.
- Interpretation: keep the exact TreeSHAP explanation as the primary XGBoost view. Keep LIME as a research comparison. IG and occlusion offer complementary sequence diagnostics; this study does not establish a universal winner.

## Research limitations

- Existing historical data were reused. This is an exploratory explanation study, not an independent forecasting evaluation.
- LIME uses actual historical rows with the official LimeBase weighted regression, rather than the default independent synthetic column sampling. This adaptation and its weak fit are disclosed.
- TreeSHAP and LIME use different reference conventions. Their signed magnitudes are not interchangeable.
- Occlusion and removal can break temporal continuity and relationships between indicators. IG interpolates synthetic inputs. These are sensitivity checks, not feasible financial counterfactuals or causal explanations.
- Completeness checks numerical reconstruction; it does not prove the explanation is understandable or useful to a person.
- No universal winner was registered. User understanding has not yet been measured.
- No predictor retraining, new calibration fitting, serving-model replacement or accuracy improvement is claimed.
- The 2025 candidate holdout files were not opened in this update.

## Serving and deployment

- Prepared report: `backend/data/research_reports/xai_comparison.json`, about 307 KB.
- API: `GET /research/xai-comparison`; JSON export: `GET /research/xai-comparison/export`.
- Report sources include model bundles, prediction CSVs, finance features, the protocol and calculation code.
- The API verifies source hashes, predictions, method structure and summary calculations. Missing evidence is explicit; stale or invalid evidence fails with a clear unavailable response.
- Text hashes allow Git's LF/CRLF checkout conversion without accepting changed values.
- Web requests do not compute the study or import PyTorch, Captum or LIME.
- The home page uses a small, source-checked `GET /backtest/summary` response. A stale backtest is not shown as current evidence.
- No database migration or new Render/Vercel environment variable is required.
- GitHub-connected Render and Vercel can deploy this commit. Supabase schema and data are unchanged.

## Verification

- **152 backend tests passed**.
- **11 frontend tests passed**; production Vite build passed.
- Two additional offline mathematical checks passed for the actual LIME and Captum paths.
- Existing compact serving snapshots, both TreeSHAP snapshots, all 15 period analyses and the confidence audit passed their repository checks.
- Serving smoke test confirmed the new API responses and no model-library imports on the prepared serving paths.
- Browser checks: all 11 routes, instrument search, date/count/return display, desktop and 390/320 px layouts, mobile keyboard menu, XAI model/instrument selection, retry and independent API failure handling.
- Portfolio calculations were tested in an isolated browser with explicit quote fixtures. No account or database holdings were modified.
- Local screenshots and timestamped QA records are in the ignored folder `backend/data/research/ui_xai_20261007/screens`.
- These are technical checks, not a user study or a guarantee that every external provider is available.

## Reproduce the research

Use a separate Python 3.12 environment for offline work; keep Render on its existing serving requirements.

```powershell
# From backend, in a dedicated research environment:
python -m pip install torch==2.12.1 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-xai-research.txt
python -m research.verify_xai_methods
python -m research.xai_comparison
```

- Do not rerun the study to select favorable seeds or dates. Any new protocol should be registered separately.
- Research environment used: NumPy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0, XGBoost 3.3.0, PyTorch 2.12.1 CPU, Captum 0.9.0 and LIME 0.2.0.1.
- Runtime measurements were made locally on Windows with one PyTorch CPU thread. They are not Render latency measurements.

## Primary method references

- [LIME implementation and LimeBase documentation](https://lime-ml.readthedocs.io/en/latest/lime.html): locality-weighted surrogate fitting and diagnostics.
- [Captum Integrated Gradients documentation](https://captum.ai/docs/extension/integrated_gradients): path attribution and completeness/convergence delta.
- [XGBoost prediction API](https://xgboost.readthedocs.io/en/stable/python/python_api.html#xgboost.Booster.predict): exact native contribution predictions and raw margins.
- [WCAG 2.2 guidance](https://www.w3.org/WAI/WCAG22/quickref/): keyboard access, focus and readable controls. This update is not a formal WCAG certification.

## Next recommended work

- Run a small user-comprehension pilot: can users distinguish old signals, P(UP), accuracy, drawdown and XAI limitations?
- Audit a suitable dated news dataset before a new prediction experiment.
- Register a fixed, chronological experiment with baselines, calibration and transaction costs before changing serving models.
- Continue forward evaluation with resolved forecasts and clearly stated sample sizes.
