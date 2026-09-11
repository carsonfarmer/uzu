"""Join upstream server timing logs with matched HTTP request IDs."""
import argparse
import json
import re
import statistics

parser = argparse.ArgumentParser()
parser.add_argument('responses')
parser.add_argument('server_log')
args = parser.parse_args()
with open(args.server_log) as source:
    lines = source.read().splitlines()
metrics = {}
for line in lines:
    match = re.search(r'\[req ([a-f0-9]+)\].*decode (\d+) tok @ ([\d.]+) tok/s', line)
    if match:
        metrics[match[1]] = float(match[3])
with open(args.responses) as source:
    rows = [json.loads(line) for line in source if line.strip()]
for label in dict.fromkeys(row['prompt'] for row in rows):
    samples = [r for r in rows if r['prompt'] == label and not r['warmup']]
    if not samples:
        continue
    rates = []
    for row in samples:
        tag = row['response']['id'].removeprefix('chatcmpl-')[:8]
        if tag in metrics:
            rates.append(metrics[tag])
    texts = [r['response']['choices'][0]['message'].get('content','') for r in samples]
    result = dict(prompt=label,runs=len(samples),prompt_tokens=sorted({r['response']['usage']['prompt_tokens'] for r in samples}),completion_tokens=[r['response']['usage']['completion_tokens'] for r in samples],median_http_wall_seconds=statistics.median(r['wall_seconds'] for r in samples),deterministic_text=len(set(texts)) == 1,server_timing_samples=len(rates))
    if rates:
        result['median_server_decode_tokens_per_second'] = statistics.median(rates)
    result['finish_reasons'] = sorted({r['response']['choices'][0]['finish_reason'] for r in samples})
    print(json.dumps(result))
