# Chatbot repair and explainable AI scope

Updated 28 September 2026.

## Chatbot failure and repair

- Both the deployed and local chatbot returned the same generic failure message.
- Direct provider diagnosis returned HTTP 404 with `model_not_found` for `llama-3.3-70b-versatile`.
- Groq retired that model for free/developer accounts on 16 August 2026. See [Groq's deprecation notice](https://console.groq.com/docs/deprecations).
- The existing key could list current models. `openai/gpt-oss-120b` appeared in that list and successfully answered questions and requested QuantSight tools.
- The backend now defaults to `openai/gpt-oss-120b`. An optional `GROQ_MODEL` environment variable can select another compatible Groq chat/tool model.
- There is no silent fallback to a different provider or invented answer.
- The provider has a 20-second timeout per request and no automatic SDK retries. The browser has a 90-second overall wait limit and a Stop control.
- Provider authentication, missing-model, usage-limit, timeout and connection failures return safe error messages with non-success HTTP statuses.
- The backend finishes generating the answer before sending text chunks, so initial provider failures no longer masquerade as successful chat responses.
- Raw provider exception bodies are not returned to the browser. Failure logs record the exception type, status and configured model.
- The widget supports retry without duplicating the user's message. Failed responses stay outside the model conversation history.
- Empty replies, malformed stored history and unavailable browser storage no longer break normal use.
- Stop cancels the browser request. An upstream call already in progress may still finish on the server.

## Configuration

- Keep the existing `GROQ_API_KEY` in the backend environment and Render. It stays private.
- No new API key or Supabase migration is required for this repair.
- Leaving `GROQ_MODEL` unset uses the new default. If an existing override points to a retired model, replace it with a supported model ID.
- Restart a local backend after changing its environment. Render uses the deployed code and its own environment settings.
- Model availability, quotas and billing depend on Groq and the account. Check [current models](https://console.groq.com/docs/models) before changing the override.

## What explainable AI currently means in QuantSight

- **Indicator explanations:** rule-based descriptions of RSI, MACD, moving averages and other saved inputs.
- **Bull/bear summaries:** evidence supporting or opposing the historical signal.
- **LSTM attention:** saved weights across ten input sessions. They describe internal weighting, not exact reasoning or causal importance.
- **Indicator/vote scenarios:** thresholds that change an indicator assessment or vote count. They do not rerun the predictor or guarantee that its output will flip.
- **Chat explanations:** generated text based on retrieved project data and instructions. Tool use reduces unsupported claims but does not guarantee correctness.
- **Per-prediction TreeSHAP:** Stock Detail now has a separate, verified attribution panel for both saved XGBoost models. Existing indicator summaries remain rule-based; offline global SHAP analysis is separate.
- This release corrects the interface's previous claim that attention was the model's "exact reasoning" and clarifies the limits of its scenarios.

## Grounding improvements

- The assistant describes LSTM with attention correctly and distinguishes it from the separate Transformer model.
- Saved signal/explanation tools provide historical source information and dates.
- Per-model historical confidence means P(UP) in percent, including when the directional label is SELL. It is not measured accuracy or automatically confidence in SELL.
- Latest live inference uses completed market sessions and a five-session target, rather than claiming an intraday quote.
- The prompt distinguishes the price panel, WSB history and saved news archive, and discloses the reused evaluation period and unsupported sentiment-benefit claim.
- Responses prefer short bullets, matching the widget's supported formatting.

## Verification

- Eleven new regression tests cover the model override, tool-call evidence, bounded tool rounds, history limits, provider errors, empty responses and HTTP streaming/error behavior.
- The complete backend suite passes: 83 tests. Frontend production build passes with the existing large-bundle warning.
- Local browser checks cover a real Groq answer, retry, empty response, cancellation, disabled Clear during requests, invalid stored history and 390/320-pixel layouts.
- A real saved-AAPL query returned dated model signals and P(UP) labels through the updated local backend.
- GitHub's evaluation, frontend, database and Supabase Preview checks passed for release `92e3598`; Vercel reported successful deployment.
- Hosted verification on 28 September: Render eventually recovered from initial request timeouts, then answered real RSI and saved-AAPL questions with HTTP 200. The Vercel chat widget also displayed a real provider response without uncaught page errors.
- Production verification artifacts remain local in `backend/data/research/chat_repair/`. No Render configuration or Supabase schema change was needed.
- Browser fault tests intercept only the test browser's requests. Test fixtures are not served by the application.
- These are functional checks, not an evaluation proving all chatbot answers correct. An initial open-ended answer still made an incorrect RSI-scale claim, reinforcing the need for factual checks. The prompt now explicitly supplies the correct scale, and the subsequent real response passed that check.

## Per-prediction XGBoost attribution — implemented 28 September 2026

The [4 October serving update](XAI_SERVING_PERFORMANCE.md) now prepares the same
native results offline, binds them to source/code hashes and recomputes all 88
in CI. The native calculation described below remains the preparation and fallback
path. The public response and explanation method are preserved.

- Open **Stock Detail → Overview → What moved this model's prediction?** and choose Finance or Finance + Sentiment.
- Endpoint: `GET /feature-attribution/AAPL?model=xgb_finance`; only the two saved XGBoost variants are supported.
- Scope: latest saved historical prediction per ticker (currently 20 December 2024). This endpoint does not explain a live prediction, LSTM output or the four-model consensus.
- The server selects the exact ticker/date feature row, preserves the bundle's feature order and applies its saved StandardScaler. Only the latest rows are cached to avoid duplicating the full feature panels in memory.
- XGBoost's native `pred_contribs=True`, with `approx_contribs=False`, computes exact TreeSHAP from the saved trees. No new SHAP runtime dependency or external model API is needed.
- Contributions use the tree model's reference distribution; they describe model associations, not causal market effects. Correlated inputs can make the attribution interpretation sensitive to its assumptions.
- Positive contributions increase the raw UP log-odds; negative contributions decrease them. They are not percentage-point changes, measured accuracy or separate explanations of the calibrated output.
- The baseline plus **all** feature contributions must match the raw margin (absolute/relative tolerance `1e-5`). The sigmoid of that margin must match classifier P(UP) (tolerance `1e-6`). Any early-stopped model uses the same best iteration as classifier inference.
- The regenerated raw direction and calibrated P(UP) must also match the saved CSV. Probability tolerance is `0.00005001`, reflecting rounding of the CSV percentage to two decimals. Invalid/missing/duplicate input evidence or mismatches produce an unavailable explanation, not invented bars.
- Direction uses raw P(UP) at 50%; calibration can cross that threshold without changing the saved direction. For AAPL Finance: raw P(UP) is approximately 46.49%, calibrated P(UP) 55.19%, and the saved direction is SELL.
- The panel displays the eight largest absolute contributions plus the sum of remaining inputs. An expandable table includes every raw/scaled input and contribution, full model/input SHA-256 hashes, tree count and reconstruction error.
- Model hash covers the saved bundle, including scaler and calibrator; input hash covers the ordered raw feature names/values. Ticker/date are returned separately. These identify evidence; they do not establish how the historical model was trained or remove evaluation-period reuse.
- Finance + Sentiment uses saved inputs, not current news. The latest historical text values are zero; SHAP can assign a nonzero contribution to a zero-valued feature based on tree splits and the reference score. This does not demonstrate a benefit from fresh sentiment data.
- The earlier “Signal Contribution” section is now labelled “Indicator & Model Summary” to distinguish heuristic scores from actual model attribution.
- No new environment variable, API key or Supabase migration is required. Deploy the updated backend and frontend together.

### Verification of this addition

- All **88 latest predictions** (44 tickers × 2 models) reproduce saved directions and probabilities. Maximum raw-score reconstruction error: **6.28 × 10⁻⁷**.
- Seven new regression tests cover real-tree reconstruction, early stopping, calibration/direction separation, corrupted contributions, invalid inputs, saved-output mismatches, exact-date matching, cache refresh, duplicate evidence and HTTP failure handling.
- Full backend suite: **90 passing tests**. Frontend production build passes with its existing bundle-size warning.
- Local Chromium checks pass for both model options, all-input tables, 390/320-pixel layouts, a simulated API failure and retry. No uncaught page errors or page-level horizontal overflow were observed.
- Test screenshots and per-prediction reconstruction results are local in `backend/data/research/xai_verification/` (excluded from Git).
- Functional consistency is verified. Human understanding, usefulness and trust still need a user study; this addition does not change model accuracy.

Implementation references: [XGBoost prediction and early stopping](https://xgboost.readthedocs.io/en/latest/prediction.html) and [native contribution API](https://xgboost.readthedocs.io/en/latest/python/python_api.html).
