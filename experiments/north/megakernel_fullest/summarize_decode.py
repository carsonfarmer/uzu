"""Balanced four-way decode summary; all pairs and full-array gates retained."""
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import sys
p=Path(sys.argv[1]);rows=[json.loads(x) for x in p.read_text().splitlines()]
names=('prepared','ready_direct','ready_late','ready_early');out={}
for prompt in sorted({r['prompt'] for r in rows if r['kind']=='generation'}):
    g=[r for r in rows if r['kind']=='generation' and r['prompt']==prompt and not r['warmup']]
    checks=[r for r in rows if r['kind']=='correctness' and r['prompt']==prompt]
    reps=sorted({r['repetition'] for r in g});steps=g[0]['decode_steps'];expected=g[0]['token_ids']
    assert len(g)==len(reps)*4 and all(r['token_ids']==expected and r['decode_steps']==steps for r in g)
    assert len(checks)==steps*4 and len({(r['variant'],r['step']) for r in checks})==steps*4 and all(r['unequal_bytes']==0 for r in checks)
    by={(r['repetition'],r['variant']):r for r in g};assert len(by)==len(g)
    orders={rep:[r['variant'] for r in g if r['repetition']==rep] for rep in reps}
    for a,b in itertools.combinations(names,2):assert sum(o.index(a)<o.index(b) for o in orders.values())*2==len(reps)
    contrasts={}
    for name in names[1:]:
        logs=[math.log(by[(r,'prepared')]['decode_seconds']/by[(r,name)]['decode_seconds']) for r in reps]
        rng=random.Random(531);boot=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
        contrasts[name]=dict(speed_ratio=math.exp(statistics.mean(logs)),bootstrap95=[boot[249],boot[9749]],min_pair=math.exp(min(logs)),max_pair=math.exp(max(logs)))
    out[prompt]=dict(full_array_checks=len(checks),measured_generations=len(g),tokens_per_generation=len(expected),balanced_pairs=True,all_tokens_identical=True,median_tps={n:statistics.median(by[(r,n)]['decode_tokens_per_second'] for r in reps) for n in names},contrasts_over_prepared=contrasts)
print(json.dumps(dict(raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),results=out,scope='Fresh paired prepared+async denominator; pilot length retained, not extrapolated to longer contexts'),indent=2))
