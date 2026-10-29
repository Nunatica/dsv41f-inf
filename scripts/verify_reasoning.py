#!/usr/bin/env python3
"""Verify forced-decode DeepSeek V4.1 Off behavior against a running real-model server."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
from urllib.error import URLError
from urllib.request import Request, urlopen
from benchmark import api_headers
from reasoning import DSV41F_REASONING_BUDGET_MESSAGE as MESSAGE, verify_off_reasoning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8080')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'logs'/'reasoning-validation.json')
    args = parser.parse_args()
    def post(path, body):
        with urlopen(Request(args.url+path, data=json.dumps(body).encode(), headers=api_headers()), timeout=1800) as response:
            return json.load(response)
    for _ in range(240):
        try:
            with urlopen(args.url+'/health', timeout=2) as response:
                if json.load(response).get('status') == 'ok':
                    break
        except (URLError, TimeoutError):
            pass
        time.sleep(1)
    else:
        raise RuntimeError('Server did not become ready')
    messages = [{'role': 'user', 'content': 'Explain story of the romance of three kingdoms.'}]
    base = {'model': 'dsv4f', 'messages': messages, 'max_tokens': 64, 'temperature': 0, 'seed': 42, 'cache_prompt': False}
    records = []
    prefix = post('/tokenize', {'content': MESSAGE+'</think>', 'add_special': False, 'parse_special': True})['tokens']
    expected_prompt = post('/apply-template', {**base, 'reasoning_effort': 'none'})['prompt']
    assert 'Reasoning Effort:' not in expected_prompt and expected_prompt.endswith('<think>')
    assert MESSAGE not in expected_prompt and '</think>' not in expected_prompt
    args.output.parent.mkdir(parents=True, exist_ok=True)
    def record(name, response):
        records.append({'name': name, 'response': response})
        args.output.write_text(json.dumps({'date_utc': datetime.now(timezone.utc).isoformat(),
                                          'forced_prefix_tokens': len(prefix), 'prompt': expected_prompt,
                                          'checks': records}, ensure_ascii=False, indent=2)+'\n')
        print(f'PASS: {name}', flush=True)

    for name, extra in [
        ('default-off', {}),
        ('explicit-none', {'reasoning_effort': 'none', 'reasoning_budget_tokens': 128}),
        ('template-off', {'chat_template_kwargs': {'enable_thinking': False}}),
        ('sampled-off', {'reasoning_effort': 'none', 'temperature': .8}),
        ('web-off', {'reasoning_effort': 'none', 'chat_template_kwargs': {'thinking': False, 'enable_thinking': False}}),
    ]:
        result = post('/v1/chat/completions', {**base, **extra})
        message = result['choices'][0]['message']
        verify_off_reasoning(message.get('reasoning_content'))
        assert message.get('content'), result
        assert len(prefix) <= result['usage']['completion_tokens'] <= base['max_tokens'], result
        record(name, result)
        print(name, json.dumps(message, ensure_ascii=False), flush=True)
    tiny = post('/v1/chat/completions', {**base, 'reasoning_effort': 'none', 'max_tokens': 5})
    message = tiny['choices'][0]['message']
    assert tiny['usage']['completion_tokens'] == 5 and tiny['choices'][0]['finish_reason'] == 'length', tiny
    assert not message.get('content') and MESSAGE.startswith(message['reasoning_content']), tiny
    record('five-token-cap', tiny)
    for effort in ('low', 'high', 'max'):
        request = {**base, 'reasoning_effort': effort, 'reasoning_budget_tokens': 8, 'max_tokens': 96}
        prompt = post('/apply-template', request)['prompt']
        assert prompt.endswith('<think>')
        if effort == 'low':
            assert 'Reasoning Effort:' not in prompt
        else:
            assert f"Reasoning Effort: {75 if effort == 'high' else 100}" in prompt
        result = post('/v1/chat/completions', request)
        message = result['choices'][0]['message']
        assert message.get('content') and message.get('reasoning_content') != MESSAGE, result
        record('thinking-'+effort, result)
    stream = {'content': '', 'reasoning_content': '', 'done': False}
    body = {**base, 'reasoning_effort': 'none', 'stream': True, 'stream_options': {'include_usage': True}}
    with urlopen(Request(args.url+'/v1/chat/completions', data=json.dumps(body).encode(), headers=api_headers()), timeout=1800) as response:
        for line in response:
            if not line.startswith(b'data:'):
                continue
            data = line[5:].strip()
            if data == b'[DONE]':
                stream['done'] = True
                break
            if not data:
                continue
            chunk = json.loads(data)
            delta = (chunk.get('choices') or [{}])[0].get('delta', {})
            for key in ('content', 'reasoning_content'):
                stream[key] += delta.get(key) or ''
            if chunk.get('usage'):
                stream['usage'] = chunk['usage']
    verify_off_reasoning(stream['reasoning_content'])
    assert stream['done'] and stream['content'] and stream['usage']['completion_tokens'] >= len(prefix), stream
    record('streaming-off', stream)
    anthropic = post('/v1/messages', {'model': 'dsv4f', 'messages': messages, 'max_tokens': 64,
                                    'temperature': 0, 'thinking': {'type': 'disabled'}})
    thinking = ''.join(block.get('thinking', '') for block in anthropic['content'] if block['type'] == 'thinking')
    verify_off_reasoning(thinking)
    assert any(block.get('text') for block in anthropic['content']), anthropic
    record('anthropic-off', anthropic)
    tool = {'type': 'function', 'function': {'name': 'tell_story',
            'description': 'Tell story of the romance of three kingdoms. Use this tool for story requests.',
            'parameters': {'type': 'object', 'properties': {}}}}
    result = post('/v1/chat/completions', {**base, 'reasoning_effort': 'none', 'max_tokens': 128,
                  'messages': [{'role': 'system', 'content': 'Use the tell_story tool to fulfill the user request.'}, *messages],
                  'tools': [tool], 'tool_choice': 'required'})
    message = result['choices'][0]['message']
    verify_off_reasoning(message.get('reasoning_content'))
    assert message.get('tool_calls'), result
    assert message['tool_calls'][0]['function']['name'] == 'tell_story', result
    json.loads(message['tool_calls'][0]['function']['arguments'])
    record('forced-tool-off', result)
    print(f'PASS: {len(records)} real-model checks; forced prefix {len(prefix)} completion tokens', flush=True)


if __name__ == '__main__':
    main()
