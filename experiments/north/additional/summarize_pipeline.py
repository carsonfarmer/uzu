"""Audit pipeline exactness and paired generation evidence using only the CPU."""
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
    if r['kind']=='correctness':
        assert r['unequal_bytes']==0,r
        checks[(r['prompt'],r['variant'])].append(r['step'])
    elif r['kind']=='generation' and not r['warmup']:
        assert r['variant'] not in runs[(r['prompt'],r['repetition'])]
        assert len(r['token_ids'])==r['decode_steps']+1
        assert math.isclose(r['decode_tokens_per_second'],r['decode_steps']/r['decode_seconds'])
        runs[(r['prompt'],r['repetition'])][r['variant']]=r
result={}
for prompt in args['prompts']:
    ratios=[];control=[];candidate=[];stock=[];steps=[]
    names=('exact_sync','exact_async','original_async')
    byte_gate=all(checks[(prompt,name)]==list(range(args['tokens']-1)) for name in names)
    expected=None
    for rep in range(args['runs']):
        pair=runs[(prompt,rep)];assert set(pair)==set(names),(prompt,rep)
        tokens=pair['exact_sync']['token_ids']
        if expected is None:expected=tokens
        assert tokens==expected
        assert all(pair[n]['token_ids']==tokens for n in names)
        b,c,s=[pair[n]['decode_tokens_per_second'] for n in names]
        control.append(b);candidate.append(c);stock.append(s);ratios.append(c/b)
        steps.append(pair['exact_async']['decode_steps'])
    logs=[math.log(r) for r in ratios];rng=random.Random(941)
    boots=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(20000))
    ratio=math.exp(statistics.mean(logs));ci=[boots[500],boots[19499]]
    result[prompt]=dict(prompt_tokens=pair['exact_sync']['prompt_tokens'],byte_gate=byte_gate,
        exact_steps_per_variant=len(checks[(prompt,'exact_async')]),paired_ratios=ratios,
        paired_geometric_ratio=ratio,paired_bootstrap_95_percent=ci,
        median_exact_sync_tps=statistics.median(control),median_exact_async_tps=statistics.median(candidate),
        median_original_async_tps=statistics.median(stock),
        ratio_of_medians=statistics.median(candidate)/statistics.median(control),
        every_pair_above_ten_percent=min(ratios)>=1.10,
        conservative_pass=byte_gate and len(ratios)>=6 and min(steps)>=127 and ci[0]>=1.10)
print(json.dumps(dict(artifact=str(a.artifact),sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
    source_hashes_match=True,results=result,
    all_prompts_conservative_pass=all(v['conservative_pass'] for v in result.values()),
    scope='Host submission overlap with unchanged exact Metal branch kernels. Separate full-logit gate, followed by independent greedy generations; fixed length and no early EOS. Whole-run pair bootstrap, no token-level pseudo-replication. No incremental megakernel claim.'),indent=2))
