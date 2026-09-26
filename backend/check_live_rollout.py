"""Read-only rollout checks. No forecasts, test rows or admin jobs are created."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'nyse-close-5-v2'
SCHEMA_COLUMNS = ('id,ticker,predicted_date,target_date,target_close_at,horizon_sessions,'
                  'protocol_version,model_version,feature_version,feature_values,data_cutoff_at,'
                  'data_fetched_at,generated_at,record_before,created_at,predicted_signal,'
                  'confidence,probability_up,raw_probability_up,calibration_method,price_basis,'
                  'price_at_prediction,resolved,resolved_date,resolved_at,actual_close,'
                  'resolution_entry_close,actual_return,actual_signal,correct')


def public_url(value):
    parsed = urlsplit(value)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise ValueError('Use a service URL without credentials, query or fragment.')
    return value.rstrip('/')


def get_json(client, url, **kwargs):
    try:
        response = client.get(url, **kwargs)
        try:
            body = response.json()
        except ValueError:
            body = None
        info = {'http_status': response.status_code}
        if response.is_error and isinstance(body, dict):
            # Do not include server messages, request headers, or credentials.
            info['error_code'] = body.get('code')
        return info, body
    except httpx.HTTPError as exc:
        return {'error_type': type(exc).__name__}, None


def check(backend_url=None):
    backend = {**dotenv_values(ROOT / '.env'), **os.environ}
    frontend = {**dotenv_values(ROOT.parent / 'frontend/.env'), **os.environ}
    results = []
    with httpx.Client(timeout=45, follow_redirects=False) as client:
        url = backend.get('SUPABASE_URL', '').rstrip('/')
        frontend_url = frontend.get('VITE_SUPABASE_URL', '').rstrip('/')
        results.append({'check': 'local_frontend_backend_database_match',
                        'passed': bool(url and frontend_url and url == frontend_url)})
        for role, key in [('service_role', backend.get('SUPABASE_SERVICE_ROLE_KEY')),
                          ('public_anon', frontend.get('VITE_SUPABASE_ANON_KEY'))]:
            result = {'check': f'database_v2_{role}_read', 'passed': False}
            if not url or not key:
                result['reason'] = 'Missing local configuration'
            else:
                headers = {'apikey': key}
                if key.count('.') == 2:  # Legacy JWT key; new opaque keys use apikey only.
                    headers['Authorization'] = 'Bearer ' + key
                info, body = get_json(client, public_url(url) + '/rest/v1/live_predictions_v2',
                                      params={'select': SCHEMA_COLUMNS, 'limit': 0}, headers=headers)
                result.update(info)
                result['passed'] = info.get('http_status') == 200 and isinstance(body, list)
                if info.get('error_code') == 'PGRST205':
                    result['reason'] = 'Table missing from API schema; apply migration and verify schema cache.'
            results.append(result)
        if backend_url:
            backend_url = public_url(backend_url)
            for path, name in [('/', 'backend_health'), ('/backtest', 'published_backtest'),
                               ('/live-track-record', 'five_session_tracker')]:
                info, body = get_json(client, backend_url + path)
                ok = info.get('http_status') == 200 and isinstance(body, dict)
                result = {'check': name, **info}
                if name == 'published_backtest':
                    ok = ok and body.get('available') is True and body.get('report', {}).get('freshness', {}).get('matches_current_files') is True
                elif name == 'five_session_tracker':
                    ok = ok and body.get('available') is True and body.get('protocol_version') == PROTOCOL and body.get('horizon_sessions') == 5 and body.get('legacy_excluded') is True
                    if isinstance(body, dict):
                        result['protocol_version'] = body.get('protocol_version')
                        result['horizon_sessions'] = body.get('horizon_sessions')
                result['passed'] = bool(ok)
                results.append(result)
    return {'checked_at': datetime.now(timezone.utc).isoformat(), 'mode': 'read_only',
            'backend_url': backend_url, 'checks': results,
            'all_requested_checks_passed': all(r['passed'] for r in results),
            'limitations': 'Reads do not verify hosted write permissions, triggers, scheduler secrets or future outcome resolution.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend-url', type=public_url)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = check(args.backend_url)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding='utf-8')
    print(text)
    raise SystemExit(0 if report['all_requested_checks_passed'] else 1)
