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
- **Offline SHAP work:** feature-importance analysis exists in the research workflow. The current stock-detail summaries are not live per-prediction SHAP attributions.
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
- Browser fault tests intercept only the test browser's requests. Test fixtures are not served by the application.
- These are functional checks, not an evaluation proving all chatbot answers correct. An initial open-ended answer still made an incorrect RSI-scale claim, reinforcing the need for factual checks. The prompt now explicitly supplies the correct scale, and the subsequent real response passed that check.

## Next XAI improvement

1. Add actual per-prediction XGBoost feature contributions, using the exact saved input vector and model version.
2. Label the contribution scale correctly: tree-model raw score contributions do not directly explain a separately calibrated probability.
3. Verify that the baseline plus contributions reconstructs the model output within numerical tolerance.
4. Show prediction date, model version, leading positive/negative contributions and an explanation-method label.
5. Test whether users understand these explanations and their limits before claiming improved trust or decision quality.
