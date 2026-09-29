"""Lightweight validation for published development evidence; no training imports."""
import math

CANDIDATES = ['fit_prior', 'calibration_prior', 'always_up', 'logistic_raw', 'logistic_sigmoid',
              'hist_boost_raw', 'hist_boost_sigmoid', 'xgboost_raw', 'xgboost_sigmoid']
METRICS = ['accuracy', 'balanced_accuracy', 'mcc', 'roc_auc', 'brier_score', 'log_loss', 'buy_fraction']


def validate_report(report):
    def require(condition):
        if not condition:
            raise ValueError('Invalid development report')

    def number(value, low, high):
        require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high)

    require(report['schema_version'] == 1 and report['status'] == 'development_only')
    require(all(report[key] is False for key in ['holdout_files_opened', 'holdout_scores_computed', 'serving_models_changed']))
    require(report['fits'] == {'predictors': 9, 'calibrators': 9})
    require(report['registration']['protocol']['protocol_id'] == 'finance_development_v1')
    require(len(report['run_id']) == 16 and all(c in '0123456789abcdef' for c in report['run_id']))
    require(type(report['validation_rows']) is int and report['validation_rows'] > 0)
    require(type(report['tickers']) is int and report['tickers'] > 0)
    require(len(report['registration']['protocol']['features']) == 13)
    require(len(report['registration']['protocol_sha256']) == 64)
    folds = {fold['name'] for fold in report['folds']}
    require(folds == {'validate_2022', 'validate_2023', 'validate_2024'})
    require(len(report['results']) == 27 and len(report['summaries']) == 9)
    pairs = set()
    for row in report['results']:
        require(row['fold'] in folds and row['candidate'] in CANDIDATES)
        require(row['year'] == int(row['fold'][-4:]))
        pair = (row['fold'], row['candidate'])
        require(pair not in pairs)
        pairs.add(pair)
        require(type(row['metrics']['n']) is int and row['metrics']['n'] > 0)
        for key in METRICS:
            number(row['metrics'][key], -1 if key == 'mcc' else 0, math.inf if key == 'log_loss' else 1)
        bins = row['reliability']
        require(len(bins) == 10 and sum(item['n'] for item in bins) == row['metrics']['n'])
        for index, item in enumerate(bins):
            require(item['lower'] == index / 10 and item['upper'] == (index + 1) / 10)
            require(type(item['n']) is int and item['n'] >= 0)
            if item['n']:
                number(item['mean_probability'], item['lower'], item['upper'])
                number(item['observed_up_rate'], 0, 1)
            else:
                require(item['mean_probability'] is None and item['observed_up_rate'] is None)
    for fold in folds:
        require(len({row['metrics']['n'] for row in report['results'] if row['fold'] == fold}) == 1)
    require(sum(row['metrics']['n'] for row in report['results'] if row['candidate'] == 'fit_prior') == report['validation_rows'])
    require([row['candidate'] for row in report['summaries']] == CANDIDATES)
    baseline = report['summaries'][0]['mean_metrics']['brier_score']
    for summary in report['summaries']:
        rows = [row for row in report['results'] if row['candidate'] == summary['candidate']]
        for key in METRICS:
            mean = sum(row['metrics'][key] for row in rows) / 3
            require(math.isclose(summary['mean_metrics'][key], mean, abs_tol=1e-12))
        require(math.isclose(summary['worst_year_brier'], max(row['metrics']['brier_score'] for row in rows), abs_tol=1e-12))
        require(math.isclose(summary['brier_gain_vs_fit_prior'], baseline - summary['mean_metrics']['brier_score'], abs_tol=1e-12))
        low, high = summary['interval_95']
        number(low, -1, 1)
        number(high, -1, 1)
        require(low <= high)
    chosen = min(report['summaries'], key=lambda row: row['mean_metrics']['brier_score'])['candidate']
    require(chosen == report['selected_candidate'])
    return report
