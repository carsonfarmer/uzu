"""Separate raw-seconds/order audit for the twelve-pair confirmation."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics

p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);a=p.parse_args()
rows=[json.loads(line) for line in a.artifact.read_text().splitlines()]
assert rows[0]['args']['runs']==12
results={}
for prompt in rows[0]['args']['prompts']:
    records=[r for r in rows if r.get('kind')=='generation' and r['prompt']==prompt and not r['warmup']]
    assert len(records)==24
    ratios=[];orders={0:[],1:[]}
    for rep in range(12):
        pair=[r for r in records if r['repetition']==rep]
        assert len(pair)==2
        control=next(r for r in pair if r['variant']=='exact_sync')
        candidate=next(r for r in pair if r['variant']=='prepared_async')
        assert control['token_ids']==candidate['token_ids'] and len(candidate['token_ids'])==128
        ratio=control['decode_seconds']/candidate['decode_seconds']
        assert math.isclose(ratio,candidate['decode_tokens_per_second']/control['decode_tokens_per_second'])
        ratios.append(ratio);orders[rep%2].append(ratio)
    logs=list(map(math.log,ratios));mean=statistics.mean(logs)
    error=statistics.stdev(logs)/math.sqrt(12)
    # Two-sided 95% t critical value for eleven degrees of freedom.
    t=2.200985
    results[prompt]=dict(paired_geo_from_seconds=math.exp(mean),
        log_t_95_interval=[math.exp(mean-t*error),math.exp(mean+t*error)],
        AB_geo=math.exp(statistics.mean(map(math.log,orders[0]))),
        BA_geo=math.exp(statistics.mean(map(math.log,orders[1]))),minimum_pair=min(ratios))
print(json.dumps(dict(artifact_sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
    method='Independent recomputation from raw seconds, using all 12 pairs; t11 log-ratio interval and order-stratified means. No GPU use.',results=results),indent=2))
