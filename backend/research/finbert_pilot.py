"""Pinned, resumable offline FinBERT comparison; never modifies serving models."""
import hashlib
import importlib.metadata
import json
import pickle
from pathlib import Path
import sqlite3
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from research.news_dataset import ROOT, REVISION
from research.news_pilot import chronological_segments, metrics, paired_brier

PROTOCOL = Path(__file__).with_name('finbert_protocol.json')
LABELS = ('positive', 'negative', 'neutral')


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def ordered_probabilities(probabilities, id2label):
    mapping = {str(label).lower(): int(index) for index, label in id2label.items()}
    if set(mapping) != set(LABELS) or set(mapping.values()) != {0, 1, 2}:
        raise ValueError('Expected exactly positive, negative and neutral labels')
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError('Invalid probability matrix')
    if ((values < 0) | (values > 1)).any() or not np.allclose(values.sum(axis=1), 1, atol=1e-6):
        raise ValueError('Invalid probability distribution')
    return values[:, [mapping[label] for label in LABELS]]


def align_parent(evaluation, parent):
    keys = ['ticker', 'date']
    if evaluation.duplicated(keys).any() or parent.duplicated(keys).any():
        raise ValueError('Duplicate evaluation keys')
    if len(evaluation) != len(parent):
        raise ValueError('Parent and extension row counts differ')
    joined = evaluation[keys + ['signal', 'news_observed']].merge(
        parent, on=keys, how='left', validate='one_to_one', suffixes=('', '_parent'), indicator=True)
    if not joined['_merge'].eq('both').all():
        raise ValueError('Parent evaluation rows differ')
    for name in ['signal', 'news_observed']:
        if not joined[name].eq(joined[name + '_parent']).all():
            raise ValueError('Parent labels or news coverage differ')
    return joined


def aggregate_scores(news, scores):
    if scores.headline.duplicated().any():
        raise ValueError('Duplicate headline scores')
    data = news.merge(scores, on='headline', how='left', validate='many_to_one')
    if data[list(LABELS)].isna().any().any():
        raise ValueError('Missing headline scores; refusing silent zero imputation')
    ordered_probabilities(data[list(LABELS)], dict(enumerate(LABELS)))
    data['polarity'] = data.positive - data.negative
    data['positive_share'] = data.polarity.gt(.05).astype(float)
    data['negative_share'] = data.polarity.lt(-.05).astype(float)
    daily = data.groupby(['ticker', 'date']).agg(
        news_count=('article_id', 'size'), news_polarity_mean=('polarity', 'mean'),
        news_polarity_std=('polarity', 'std'), news_positive_share=('positive_share', 'mean'),
        news_negative_share=('negative_share', 'mean')).reset_index()
    daily['news_observed'] = 1.
    daily['news_log_count'] = np.log1p(daily.pop('news_count'))
    daily['news_polarity_std'] = daily.news_polarity_std.fillna(0)
    return daily


def score_news(news, settings):
    # Heavy libraries are only loaded for an explicit offline research run.
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    files = ['config.json', 'pytorch_model.bin', 'vocab.txt',
             'tokenizer_config.json', 'special_tokens_map.json']
    try:
        snapshot = Path(snapshot_download(settings['model'], revision=settings['revision'],
                                         local_files_only=True))
    except FileNotFoundError:
        snapshot = Path(snapshot_download(settings['model'], revision=settings['revision'],
                                         allow_patterns=files))
    versions = {key: importlib.metadata.version(key) for key in ['torch', 'transformers', 'tokenizers']}
    context = {'settings': settings, 'software': versions,
               'model_files_sha256': {name: sha(snapshot / name) for name in files}}
    cache_id = hashlib.sha256(json.dumps(context, sort_keys=True).encode()).hexdigest()[:20]
    directory = ROOT / 'data/research/finbert_scores' / cache_id
    directory.mkdir(parents=True, exist_ok=True)
    manifest = directory / 'manifest.json'
    manifest.write_text(json.dumps(context, indent=2) + '\n', encoding='utf-8')
    texts = sorted(news.headline.unique())
    hashes = {text: hashlib.sha256(text.encode()).hexdigest() for text in texts}
    started = time.monotonic()
    with sqlite3.connect(directory / 'scores.sqlite') as db:
        db.execute('create table if not exists scores (text_sha text primary key, positive real, negative real, neutral real, token_count integer)')
        existing = {row[0]: row[1:] for row in db.execute('select * from scores')}
        missing = [text for text in texts if hashes[text] not in existing]
        print(f'FinBERT cache: {len(texts)-len(missing)}/{len(texts)} headlines available', flush=True)
        if missing:
            torch.set_num_threads(settings['threads'])
            torch.manual_seed(42)
            tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)
            model = AutoModelForSequenceClassification.from_pretrained(
                snapshot, local_files_only=True, trust_remote_code=False, weights_only=True)
            model.to(device='cpu', dtype=torch.float32).eval()
            lengths = [len(ids) for ids in tokenizer(missing, truncation=False)['input_ids']]
            # Similar-length batches reduce padding, without selecting articles by outcomes.
            ordered = sorted(zip(missing, lengths), key=lambda pair: (pair[1], pair[0]))
            batch_size = settings['batch_size']
            for offset in range(0, len(ordered), batch_size):
                batch = ordered[offset:offset + batch_size]
                inputs = tokenizer([text for text, _ in batch], padding=True, truncation=True,
                                   max_length=settings['max_length'], return_tensors='pt')
                with torch.inference_mode():
                    raw = torch.softmax(model(**inputs).logits, dim=-1).cpu().numpy()
                values = ordered_probabilities(raw, model.config.id2label)
                records = [(hashes[text], *map(float, probs), length)
                           for (text, length), probs in zip(batch, values)]
                db.executemany('insert into scores values (?, ?, ?, ?, ?)', records)
                db.commit()  # Resume after interruption without rescoring completed batches.
                existing.update({row[0]: row[1:] for row in records})
                done = min(offset + batch_size, len(ordered))
                if done % 1024 == 0 or done == len(ordered):
                    print(f'FinBERT scored {done}/{len(ordered)} new headlines ({time.monotonic()-started:.0f}s)', flush=True)
    output = pd.DataFrame([(text, *existing[hashes[text]][:3]) for text in texts],
                          columns=['headline', *LABELS])
    ordered_probabilities(output[list(LABELS)], dict(enumerate(LABELS)))
    return output, {**context, 'cache_id': cache_id, 'unique_headlines': len(texts),
                    'truncated_headlines': sum(existing[hashes[text]][3] > settings['max_length'] for text in texts),
                    'scores_sha256': sha(directory / 'scores.sqlite')}


def run():
    protocol = json.loads(PROTOCOL.read_text(encoding='utf-8'))
    parent_path = ROOT.parent / 'docs/research/news_pilot_results.json'
    parent = json.loads(parent_path.read_text(encoding='utf-8'))
    if parent['run_id'] != protocol['parent_run_id']:
        raise ValueError('Unexpected parent experiment')
    news_path = ROOT / 'data/research/drinkall' / REVISION / 'normalized_news.jsonl'
    input_path = ROOT / 'data/processed/features_finance.csv'
    model_path = ROOT / 'data/models/xgb_finance.pkl'
    hashes = {name: sha(path) for name, path in [('news', news_path), ('finance', input_path), ('feature_list_model', model_path)]}
    if hashes != parent['input_sha256']:
        raise ValueError('Inputs changed since parent pilot; do not reuse its controls')
    parent_predictions_path = ROOT / 'data/research/pilots' / parent['run_id'] / 'predictions.csv'
    hashes['parent_predictions'] = sha(parent_predictions_path)
    hashes['parent_results'] = sha(parent_path)
    protocol_hash = sha(PROTOCOL)
    run_id = hashlib.sha256((protocol_hash + json.dumps(hashes, sort_keys=True)).encode()).hexdigest()[:16]
    directory = ROOT / 'data/research/pilots' / run_id
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / 'results.json').exists():
        raise ValueError(f'Completed run {run_id} already exists; preserve it')
    saved_protocol = directory / 'protocol.json'
    if saved_protocol.exists() and sha(saved_protocol) != protocol_hash:
        raise ValueError('Cannot resume a changed protocol')
    saved_protocol.write_bytes(PROTOCOL.read_bytes())
    news = pd.read_json(news_path, lines=True, convert_dates=False)
    scores, score_metadata = score_news(news, protocol['scorer'])
    daily = aggregate_scores(news, scores)
    finance = pd.read_csv(input_path)
    frame = finance.merge(daily, on=['ticker', 'date'], how='left', validate='one_to_one')
    added = protocol['coverage_features'] + protocol['sentiment_features']
    frame[added] = frame[added].fillna(0)
    segments = chronological_segments(frame, protocol)
    train, calibration, evaluation = (segments[name] for name in ['fit', 'calibration', 'evaluation'])
    reference = pd.read_csv(parent_predictions_path)
    aligned = align_parent(evaluation, reference)
    # Verify stored controls agree with published numbers before calculating comparisons.
    for family in protocol['model_families']:
        for arm in ['finance', 'finance_coverage', 'finance_coverage_sentiment']:
            key = family + '_' + arm
            actual = metrics(evaluation.signal, aligned[key])
            expected = parent['models'][key]['all_rows']
            if not all(np.isclose(actual[k], expected[k], atol=1e-12, rtol=0) for k in actual):
                raise ValueError('Parent predictions disagree with the published result')
    with model_path.open('rb') as handle:
        columns = pickle.load(handle)['features'] + added  # Existing trusted local artifact only.
    output = {'run_id': run_id, 'protocol': protocol, 'input_sha256': hashes,
              'protocol_sha256': protocol_hash, 'scoring': score_metadata,
              'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [Path(__file__), Path(__file__).with_name('news_pilot.py')]},
              'software': {k: importlib.metadata.version(k) for k in ['numpy', 'pandas', 'scikit-learn']},
              'segments': {name: {'rows': len(part), 'date_start': part.date.min(), 'date_end': part.date.max(),
                                   'last_label_end': part.label_end.max(), 'news_observed_rows': int(part.news_observed.sum())}
                           for name, part in segments.items()},
              'models': {}, 'paired_comparisons': {}, 'baseline_training_prior': parent['baseline_training_prior'],
              'baseline_always_up': parent['baseline_always_up']}
    predictions = evaluation[['ticker', 'date', 'signal', 'news_observed']].copy()
    for family in protocol['model_families']:
        model = (make_pipeline(StandardScaler(), LogisticRegression(**protocol[family], random_state=protocol['seed']))
                 if family == 'logistic' else HistGradientBoostingClassifier(**protocol[family], random_state=protocol['seed']))
        with threadpool_limits(limits=4):
            model.fit(train[columns], train.signal)
            sigmoid = LogisticRegression(C=1e6, max_iter=1000, random_state=protocol['seed'])
            sigmoid.fit(model.predict_proba(calibration[columns])[:, 1].reshape(-1, 1), calibration.signal)
            probabilities = sigmoid.predict_proba(model.predict_proba(evaluation[columns])[:, 1].reshape(-1, 1))[:, 1]
        key = family + '_finance_coverage_finbert'
        predictions[key] = probabilities
        mask = evaluation.news_observed.eq(1).to_numpy()
        output['models'][key] = {'all_rows': metrics(evaluation.signal, probabilities),
                                'news_observed_rows': metrics(evaluation.signal.to_numpy()[mask], probabilities[mask]),
                                'features': columns}
        output['paired_comparisons'][family] = {
            name: paired_brier(evaluation, aligned[family + '_' + arm].to_numpy(), probabilities,
                               protocol['bootstrap_block_sessions'], protocol['bootstrap_repetitions'], protocol['seed'])
            for name, arm in [('versus_vader', 'finance_coverage_sentiment'), ('versus_coverage', 'finance_coverage')]}
        print(key, json.dumps(output['models'][key]['all_rows']), flush=True)
    output['completed_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    predictions.to_csv(directory / 'predictions.csv', index=False)
    text = json.dumps(output, indent=2) + '\n'
    (directory / 'results.json').write_text(text, encoding='utf-8')
    (ROOT.parent / 'docs/research/finbert_pilot_results.json').write_text(text, encoding='utf-8')
    print('Completed FinBERT extension', run_id, flush=True)


if __name__ == '__main__':
    run()
