import hashlib,itertools,json,math,random,statistics,sys
from pathlib import Path
p=Path(sys.argv[1]);rows=[json.loads(s) for s in p.read_text().splitlines()]
names=('prepared','phase','fine');g=[r for r in rows if r['kind']=='generation' and not r['warmup']];c=[r for r in rows if r['kind']=='correctness']
reps=sorted({r['repetition'] for r in g});steps=g[0]['decode_steps'];expected=g[0]['token_ids']
assert len(c)==3*steps and len({(r['variant'],r['step']) for r in c})==3*steps
assert all(r['unequal_bytes']==0 and r.get('errors',0)==0 and r.get('bad_task_states',0)==0 for r in c)
assert len(g)==3*len(reps) and all(r['token_ids']==expected for r in g)
by={(r['repetition'],r['variant']):r for r in g};assert len(by)==len(g)
orders={rep:[r['variant'] for r in g if r['repetition']==rep] for rep in reps}
for a,b in itertools.combinations(names,2):assert sum(o.index(a)<o.index(b) for o in orders.values())*2==len(reps)
contrasts={}
for a,b in [('fine','prepared'),('phase','prepared'),('fine','phase')]:
 logs=[math.log(by[(r,b)]['decode_seconds']/by[(r,a)]['decode_seconds']) for r in reps]
 rng=random.Random(718);boot=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
 contrasts[a+'_over_'+b]=dict(ratio=math.exp(statistics.mean(logs)),bootstrap95=[boot[249],boot[9749]],min_pair=math.exp(min(logs)),max_pair=math.exp(max(logs)))
print(json.dumps(dict(raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),full_array_checks=len(c),measured_generations=len(g),tokens_per_generation=len(expected),all_gates_pass=True,balanced=True,medians_tps={n:statistics.median(by[(r,n)]['decode_tokens_per_second'] for r in reps) for n in names},contrasts=contrasts,scope='Matched async shared-weight whole-pass pilot; no extrapolation beyond tested length'),indent=2))
