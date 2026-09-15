"""Summarize native pairs, retain every stall, reject incomplete byte gates."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys

p=Path(sys.argv[1]);rows=[json.loads(x) for x in p.read_text().splitlines()]
gates=[x for x in rows if x['kind']=='native_correctness']
eligible=len(gates)==3 and all(not any(x['unequal_bytes']) and x['errors']==0 and x['prep_completed']==176 for x in gates)
data=[x for x in rows if x['kind']=='native_timing' and not x['warmup']]
names=('direct','late','early','fill_only');reps=sorted({x['repetition'] for x in data})
assert len(data)==len(names)*len(reps)
assert all(x['gpuTimestampsAvailable'] and x['gpuSeconds']>0 for x in data)
by={(x['repetition'],x['variant']):x for x in data};assert len(by)==len(data)
orders={rep:[x['variant'] for x in data if x['repetition']==rep] for rep in reps}
for a,b in itertools.combinations(names,2):
    assert sum(o.index(a)<o.index(b) for o in orders.values())*2==len(reps)
medians={v:statistics.median(by[(r,v)]['gpuSeconds'] for r in reps)*1e6 for v in names}
contrasts={}
for a,b in [('early','direct'),('late','direct'),('early','late')]:
    logs=[math.log(by[(r,b)]['gpuSeconds']/by[(r,a)]['gpuSeconds']) for r in reps]
    rng=random.Random(684);boots=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
    contrasts[a+'_over_'+b]=dict(speed_ratio=math.exp(statistics.mean(logs)),bootstrap95=[boots[249],boots[9749]],min_pair=math.exp(min(logs)),max_pair=math.exp(max(logs)))
print(json.dumps(dict(raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),eligible=eligible,correctness=gates,paired_balanced=True,median_gpu_us=medians,contrasts=contrasts,scope='Native real-weight constituent command duration only; no end-to-end or physical overlap inference'),indent=2))
