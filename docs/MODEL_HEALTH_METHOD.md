# Model-health monitoring method

- Registered before calculating the new monitoring results.
- Read existing five-session forecasts only; use no outcome labels in input monitoring.
- Keep forecast dates, model versions, feature-code versions and calibrators separate.
- Compare each required input with the same ticker's historical 1st–99th percentile band.
- Historical reference: saved finance inputs dated 2015–2022. The saved model has no training-date manifest, so call this a historical development reference, not its proven training distribution.
- Review a feature when at least 20 finite recorded values are available and at least 20% fall outside that band's limits. These fixed thresholds are practical flags, not statistical tests.
- Flag a raw or calibrated probability span of at most 0.005 when at least 20 forecasts are available. Distinguish a flat raw model from compression by the calibrator.
- Check missing/non-finite inputs, probabilities, raw decision agreement, timestamp provenance and latest-session saved coverage.
- Display unsupported model versions and missing reference data explicitly. Never silently compare another model with this reference.
- Show sample size, dates and simple explanations. A stock snapshot is not a sample of independent market dates.
- Preserve trained models, prediction decisions and the reserved 2025 candidate holdout.

References: [NumPy quantiles](https://numpy.org/doc/stable/reference/generated/numpy.quantile.html) and [scikit-learn calibration](https://scikit-learn.org/stable/modules/calibration.html). Isotonic calibration can create tied probabilities; a narrow spread alone does not establish a calibration error.
