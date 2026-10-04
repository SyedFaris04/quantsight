"""Lightweight integrity and arithmetic checks for confidence audit evidence."""
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'data/research_reports/development_confidence.json'
PROTOCOL = Path(__file__).with_name('confidence_audit_protocol.json')


def checksum(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=32)
def _text_hashes(path, modified_ns, size):
    text = Path(path).read_text(encoding='utf-8')
    return {hashlib.sha256(text.encode()).hexdigest(),
            hashlib.sha256(text.replace('\n', '\r\n').encode()).hexdigest()}


def text_hashes(path):
    path = Path(path)
    stat = path.stat()
    return _text_hashes(str(path), stat.st_mtime_ns, stat.st_size)


def normalized_hash(path):
    return hashlib.sha256(Path(path).read_text(encoding='utf-8').encode()).hexdigest()


def validate(report, parent, protocol):
    def require(condition):
        if not condition:
            raise ValueError('Invalid confidence audit')

    def number(value, low=0, high=1):
        require(type(value) in (int, float) and math.isfinite(value) and low <= value <= high)

    require(report['schema_version'] == 1 and report['status'] == 'exploratory_diagnostic')
    require(report['parent_run_id'] == parent['run_id'] == protocol['parent_run_id'])
    require(report['protocol'] == protocol and report['new_fits'] == 0)
    require(report['holdout_files_opened'] is False and report['serving_models_changed'] is False)
    content = {k: v for k, v in report.items() if k != 'content_sha256'}
    require(report['content_sha256'] == checksum(content))
    require(report['parent_report_sha256'] == checksum(parent))
    require(set(report['periods']) == {'pooled', '2022', '2023', '2024'})
    candidates = protocol['candidates']
    fields = ['accuracy', 'error_rate', 'mean_confidence', 'brier_score', 'baseline_accuracy',
              'baseline_brier', 'always_up_accuracy', 'predicts_up']
    for name, period in report['periods'].items():
        require(set(period['candidates']) == set(candidates))
        expected_n = parent['validation_rows'] if name == 'pooled' else next(
            row['metrics']['n'] for row in parent['results'] if row['candidate'] == 'fit_prior' and str(row['year']) == name)
        require(period['total_rows'] == expected_n)
        for candidate, rows in period['candidates'].items():
            require([r['threshold'] for r in rows] == protocol['thresholds'])
            previous_n, previous_dates = expected_n, math.inf
            for row in rows:
                n, dates = row['retained_rows'], row['retained_dates']
                require(type(n) is int and type(dates) is int and 0 <= n <= previous_n and 0 <= dates <= previous_dates)
                require(dates <= n and (dates > 0) == (n > 0))
                number(row['coverage'])
                require(math.isclose(row['coverage'], n / expected_n, abs_tol=1e-12))
                require(row['small_sample'] == (0 < n and (n < 200 or dates < 20)))
                if n:
                    for field in fields:
                        number(row[field])
                    require(row['mean_confidence'] + 1e-12 >= row['threshold'])
                    require(math.isclose(row['accuracy'] + row['error_rate'], 1, abs_tol=1e-12))
                else:
                    require(all(row[field] is None for field in fields))
                previous_n, previous_dates = n, dates
            require(rows[0]['retained_rows'] == expected_n and rows[0]['coverage'] == 1)
            if name != 'pooled':
                original = next(r['metrics'] for r in parent['results'] if r['candidate'] == candidate and str(r['year']) == name)
                for key in ['accuracy', 'brier_score']:
                    require(math.isclose(rows[0][key], original[key], abs_tol=3e-10))
    # The pooled view must be count-weighted for each conditional sample.
    for candidate in candidates:
        for i, pooled in enumerate(report['periods']['pooled']['candidates'][candidate]):
            years = [report['periods'][str(y)]['candidates'][candidate][i] for y in protocol['years']]
            require(pooled['retained_rows'] == sum(r['retained_rows'] for r in years))
            require(pooled['retained_dates'] == sum(r['retained_dates'] for r in years))
            if pooled['retained_rows']:
                for field in fields:
                    mean = sum(r[field] * r['retained_rows'] for r in years if r['retained_rows']) / pooled['retained_rows']
                    require(math.isclose(pooled[field], mean, abs_tol=1e-12))
    return report


def load(parent, path=REPORT):
    path = Path(path)
    if not path.exists():
        return None
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    report = validate(json.loads(path.read_text(encoding='utf-8')), parent, protocol)
    expected = {f"data/research_predictions/development_{parent['run_id']}/validate_{y}.csv"
                for y in protocol['years']}
    if set(report['sources']) != expected:
        raise ValueError('Invalid prediction sources')
    for relative, digest in report['sources'].items():
        source = (ROOT / relative).resolve()
        if not source.is_relative_to(ROOT.resolve()) or digest not in text_hashes(source):
            raise ValueError('Confidence audit sources have changed')
        year = Path(relative).stem
        if parent['artifact_hashes'][f'{year}/predictions.csv'] not in text_hashes(source):
            raise ValueError('Prediction exports differ from the completed study')
    for name, digest in report['code_hashes'].items():
        if name not in {'confidence_audit.py', 'confidence_report.py'} or digest != normalized_hash(Path(__file__).with_name(name)):
            raise ValueError('Confidence audit implementation has changed')
    if set(report['code_hashes']) != {'confidence_audit.py', 'confidence_report.py'}:
        raise ValueError('Incomplete code hashes')
    return report
