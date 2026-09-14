"""Audit corrected historical-loop contrasts and balanced ordering."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

p=argparse.ArgumentParser();p.add_argument('artifact',type=Path);a=p.parse_args()
rows=[json.loads(line) for line in a.artifact.read_text().splitlines()]
prov=rows[0];args=prov['args'];root=Path(__file__).resolve().parents[3]
assert prov['kind']=='provenance'
for name,digest in prov['sources'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
checks=defaultdict(list);runs=defaultdict(dict)
for r in rows[1:]:
    if r['kind']=='generation':
        assert r['warmup']==(r['repetition']<0)
        assert -1<=r['repetition']<args['runs']
    if r['kind']=='correctness':
        assert r['unequal_bytes']==0,r
        checks[(r['prompt'],r['variant'])].append(r['step'])
    elif r['kind']=='generation' and not r['warmup']:
        assert r['variant'] not in runs[(r['prompt'],r['repetition'])]
        assert len(r['token_ids'])==r['decode_steps']+1
        assert math.isclose(r['decode_tokens_per_second'],r['decode_steps']/r['decode_seconds'])
        runs[(r['prompt'],r['repetition'])][r['variant']]=r
def contrast(numerator,denominator):
    ratios=[c/b for c,b in zip(numerator,denominator)]
    logs=[math.log(r) for r in ratios];rng=random.Random(491)
    boots=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(20000))
    return dict(paired_ratios=ratios,paired_geometric_ratio=math.exp(statistics.mean(logs)),
        paired_bootstrap_95_percent=[boots[500],boots[19499]],
        ratio_of_medians=statistics.median(numerator)/statistics.median(denominator))
result={}
names=('exact_host','exact_joint','prepared_async')
for prompt in args['prompts']:
    values={name:[] for name in names};steps=[];expected=None
    byte_gate=all(checks[(prompt,name)]==list(range(args['tokens']-1)) for name in names)
    for rep in range(args['runs']):
        pair=runs[(prompt,rep)];assert set(pair)==set(names),(prompt,rep)
        tokens=pair['exact_host']['token_ids']
        if expected is None:expected=tokens
        assert tokens==expected
        assert all(pair[n]['token_ids']==tokens for n in names)
        for name in names:values[name].append(pair[name]['decode_tokens_per_second'])
        steps.append(pair['exact_host']['decode_steps'])
    contrasts={name+'_over_host':contrast(values[name],values['exact_host']) for name in names[1:]}
    contrasts['prepared_over_joint']=contrast(values['prepared_async'],values['exact_joint'])
    orders=[tuple(r['variant'] for r in rows if r.get('kind')=='generation' and not r['warmup'] and r['prompt']==prompt and r['repetition']==rep) for rep in range(args['runs'])]
    balance={}
    for i,x in enumerate(names):
        assert all(sum(order.index(x)==pos for order in orders)==args['runs']//len(names) for pos in range(len(names)))
        for y in names[i+1:]:
            before=sum(order.index(x)<order.index(y) for order in orders)
            assert before==args['runs']//2,(x,y,before)
            balance[x+'_before_'+y]=before
    result[prompt]=dict(pair_order_counts=balance,prompt_tokens=pair['exact_host']['prompt_tokens'],byte_gate=byte_gate,
        exact_steps_per_variant=args['tokens']-1,
        median_tps={name:statistics.median(v) for name,v in values.items()},contrasts=contrasts,
        conservative_prepared_pass=byte_gate and len(steps)>=10 and min(steps)>=127 and contrasts['prepared_async_over_host']['paired_bootstrap_95_percent'][0]>=1.10 and contrasts['prepared_over_joint']['paired_bootstrap_95_percent'][0]>=1.10)
print(json.dumps(dict(artifact=str(a.artifact),sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
    source_hashes_match=True,results=result,
    all_prompts_conservative_prepared_pass=all(v['conservative_prepared_pass'] for v in result.values()),
    scope='All rows retained. Whole-run paired bootstrap. Historical direct-argmax host loop is the primary denominator; joint evaluation is an additional stronger-control check. Preparation is separately compared with unchanged fused async and stock async. No GPU-side duration or occupancy measurement.'),indent=2))
