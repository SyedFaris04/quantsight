"""Publish a small aggregate report for the Compare page, without article text."""
import json
from pathlib import Path
from research.news_dataset import ROOT


def publish():
    docs = ROOT.parent / 'docs/research'
    parent = json.loads((docs / 'news_pilot_results.json').read_text(encoding='utf-8'))
    extension = json.loads((docs / 'finbert_pilot_results.json').read_text(encoding='utf-8'))
    audit = json.loads((docs / 'news_candidate_audit.json').read_text(encoding='utf-8'))
    if extension['protocol']['parent_run_id'] != parent['run_id'] or extension['segments'] != parent['segments']:
        raise ValueError('Experiments do not share a parent and evaluation segments')
    for name, value in parent['input_sha256'].items():
        if extension['input_sha256'][name] != value:
            raise ValueError('Experiments used different inputs')
    rows = []
    for family, label in [('logistic', 'Logistic regression'), ('hist_gradient_boosting', 'Gradient boosting')]:
        for arm, inputs in [('finance', 'Finance'), ('finance_coverage', 'Finance + coverage'),
                            ('finance_coverage_sentiment', 'Finance + VADER'), ('finance_coverage_finbert', 'Finance + FinBERT')]:
            key = family + '_' + arm
            source = extension if arm.endswith('finbert') else parent
            rows.append({'key': key, 'family': family, 'label': label, 'inputs': inputs,
                         'metrics': source['models'][key]['all_rows']})
    result = {'schema_version': 1, 'status': 'exploratory', 'parent_run_id': parent['run_id'],
              'run_id': extension['run_id'], 'completed_at': extension['completed_at'],
              'horizon_sessions': 5, 'segments': extension['segments'],
              'covered_tickers': len(audit['covered_tickers']),
              'universe_tickers': len(audit['covered_tickers']) + len(audit['missing_universe_tickers']),
              'unique_headlines': extension['scoring']['unique_headlines'],
              'models': rows, 'baseline': extension['baseline_training_prior'],
              'always_up_accuracy': extension['baseline_always_up']['accuracy'],
              'comparisons': extension['paired_comparisons'],
              'limitations': ['The 2023 period was previously examined; this is not a fresh final test.',
                             'News covers only part of the fixed instrument universe.',
                             'Pretrained model corpus overlap and article-version timing remain limitations.',
                             'Classification results do not establish trading returns. The live model is unchanged.'],
              'sources': [{'label': 'FinBERT model', 'url': 'https://huggingface.co/ProsusAI/finbert'},
                          {'label': 'News archive', 'url': 'https://github.com/FelixDrinkall/financial-news-dataset'}]}
    path = ROOT / 'data/research_reports/news_comparison.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('Published aggregate research report', extension['run_id'])


if __name__ == '__main__':
    publish()
