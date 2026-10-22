#!/usr/bin/env python3
"""Bounded HTTP smoke test against an already running V4.1 server."""
import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--url', default='http://127.0.0.1:8080')
p.add_argument('--api-key', default='dummykey')
p.add_argument('--wait', type=float, default=120)
a = p.parse_args()
headers = {'Authorization': 'Bearer ' + a.api_key, 'Content-Type': 'application/json'}
deadline = time.monotonic() + a.wait
while True:
    try:
        with urlopen(Request(a.url + '/health', headers=headers), timeout=2) as r:
            health = json.load(r)
        if health.get('status') == 'ok':
            break
    except (URLError, TimeoutError):
        pass
    if time.monotonic() >= deadline:
        raise SystemExit('Server did not become ready within --wait')
    time.sleep(2)
print('health:', json.dumps(health), flush=True)
body = {'messages': [{'role': 'user', 'content': 'Explain story of the romance of three kingdoms.'}],
        'max_tokens': 32, 'temperature': 0, 'reasoning_effort': 'none'}
with urlopen(Request(a.url + '/v1/chat/completions', data=json.dumps(body).encode(), headers=headers), timeout=60) as r:
    answer = json.load(r)
print(json.dumps(answer, ensure_ascii=False, indent=2), flush=True)
if not answer.get('choices') or not answer['choices'][0].get('message', {}).get('content'):
    raise SystemExit('No generated content in response')
print('API generation PASS')
