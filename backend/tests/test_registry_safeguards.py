"""Run ONLY against your own local or staging PulseIQ backend.

No keys or tokens are printed. The concurrency test creates two new model
versions, so use a disposable test model if you don't want extra history.

Windows PowerShell example:
  $env:PULSEIQ_API_URL='http://127.0.0.1:8000'
  $env:PULSEIQ_USER_A_TOKEN='<User A Supabase access token>'
  $env:PULSEIQ_USER_B_TOKEN='<User B Supabase access token>'
  $env:PULSEIQ_ANALYSIS_ID='<analysis owned by A>'
  $env:PULSEIQ_MODEL_ID='<existing churn model owned by A>'
  python test_registry_safeguards.py
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import os
import sys
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

API = os.getenv('PULSEIQ_API_URL', 'http://127.0.0.1:8000').rstrip('/')
A = os.getenv('PULSEIQ_USER_A_TOKEN')
B = os.getenv('PULSEIQ_USER_B_TOKEN')
ANALYSIS = os.getenv('PULSEIQ_ANALYSIS_ID')
MODEL = os.getenv('PULSEIQ_MODEL_ID')


def request(method, path, token, body=None):
    encoded = json.dumps(body).encode() if body is not None else None
    req = Request(API + path, data=encoded, method=method, headers={
        'Authorization': 'Bearer ' + token,
        'Content-Type': 'application/json',
    })
    try:
        with urlopen(req, timeout=120) as res:
            payload = json.loads(res.read().decode())
            return res.status, payload
    except HTTPError as err:
        raw = err.read().decode()
        try:
            payload = json.loads(raw)
        except ValueError:
            payload = {'detail': raw[:200]}
        return err.code, payload


def assert_check(condition, message):
    if not condition:
        raise AssertionError(message)
    print('PASS:', message)


def main():
    if not all([A, B, ANALYSIS, MODEL]):
        sys.exit('Set all four PULSEIQ_USER_A_TOKEN, PULSEIQ_USER_B_TOKEN, PULSEIQ_ANALYSIS_ID, PULSEIQ_MODEL_ID.')
    assert_check(A != B, 'Two distinct account tokens supplied')

    owner_status, owner = request('GET', f'/predict/models/{MODEL}', A)
    assert_check(owner_status == 200 and owner.get('id') == MODEL, 'User A can access own model')

    status, records = request('GET', '/predict/models', B)
    assert_check(status == 200 and all(m.get('id') != MODEL for m in records.get('models', [])),
                 'User B cannot see User A model in listing')
    for suffix in ('', '/versions'):
        status, _ = request('GET', f'/predict/models/{MODEL}{suffix}', B)
        assert_check(status in (403, 404), f'User B denied on model{suffix or " details"} ({status})')

    common = {
        'analysis_id': ANALYSIS,
        'target_column': 'churn',
        'problem_type': 'binary_classification',
        'feature_selection': {
            'included_features': [
                'monthly_spend', 'account_age_months',
                'feature_usage_score', 'support_tickets',
            ],
            'excluded_features': ['website_visits'],
        },
        'test_size': 0.2,
        'random_state': 42,
        'save_model': True,
        'model_id': MODEL,
    }
    status, _ = request('POST', '/predict/train', B, common)
    assert_check(status in (403, 404), f'User B denied retraining User A model/dataset ({status})')

    before_status, before = request('GET', f'/predict/models/{MODEL}/versions', A)
    assert_check(before_status == 200, 'Owner can inspect versions before concurrent test')
    before_versions = before['versions']
    old_max = max((v['version_number'] for v in before_versions), default=0)
    print('Running two simultaneous training requests; this will create two additional versions.')

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(request, 'POST', '/predict/train', A, common) for _ in range(2)]
        results = [future.result() for future in futures]

    for status, result in results:
        assert_check(status == 200 and result.get('saved') is True,
                     f'Concurrent training saved successfully (HTTP {status})')
        assert_check(result.get('model_id') == MODEL, 'Concurrent version belongs to original model')

    versions = sorted(result['version_number'] for _, result in results)
    assert_check(versions == [old_max + 1, old_max + 2],
                 f'Concurrent saves generated sequential versions {versions}')
    assert_check(len({result['version_id'] for _, result in results}) == 2,
                 'Concurrent saves have distinct version IDs')

    after_status, after = request('GET', f'/predict/models/{MODEL}/versions', A)
    assert_check(after_status == 200 and len(after['versions']) == len(before_versions) + 2,
                 'Both versions retrievable; prior history preserved')
    by_num = {v['version_number']: v for v in after['versions']}
    assert_check(by_num[old_max + 2]['parent_version_id'] == by_num[old_max + 1]['id'],
                 'Newest version links to immediately preceding version')
    print('\nALL SAFEGUARD TESTS PASSED')


if __name__ == '__main__':
    try:
        main()
    except (AssertionError, URLError, KeyError) as exc:
        print('\nTEST FAILED:', str(exc))
        sys.exit(1)
