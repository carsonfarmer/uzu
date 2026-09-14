"""Pool every corrected long-context pair, stratified by experiment process."""
import argparse
import ast
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

p=argparse.ArgumentParser();p.add_argument('artifacts',nargs='+',type=Path);a=p.parse_args()
root=Path(__file__).resolve().parents[3]
strata={name:[] for name in ('exact_host','exact_joint')}
sources=[];previous_function=None;previous_hashes=None;expected_tokens=None
for artifact in a.artifacts:
    rows=[json.loads(line) for line in artifact.read_text().splitlines()]
    prov=rows[0];assert prov['kind']=='provenance'
    assert all(prov['batching_environment'].get(k) is None for k in ('MLX_MAX_MB_PER_BUFFER','MLX_MAX_OPS_PER_BUFFER')),'Nondefault batching cannot be pooled with the default control'
    for name,digest in prov['sources'].items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
    driver=next(name for name in prov['sources'] if name.endswith('/corrected_control.py') or name.endswith('/precision_long.py'))
    tree=ast.parse((root/driver).read_text())
    function=ast.dump(next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='generate'),include_attributes=False)
    shared={k:v for k,v in prov['sources'].items() if k!=driver}
    if previous_function is not None:
        assert function==previous_function,'Timing function changed'
        assert shared==previous_hashes,'Shared math/reference sources changed'
    previous_function=function;previous_hashes=shared
    data=[r for r in rows[1:] if r.get('prompt')=='long']
    for name in ('exact_host','exact_joint','prepared_async'):
        checks=[r for r in data if r['kind']=='correctness' and r['variant']==name]
        assert [r['step'] for r in checks]==list(range(127))
        assert all(r['unequal_bytes']==0 for r in checks)
    pairs=[]
    for rep in range(prov['args']['runs']):
        measured=[r for r in data if r['kind']=='generation' and r['repetition']==rep]
        assert all(not r['warmup'] for r in measured)
        by={r['variant']:r for r in measured};assert len(by)==len(measured)
        candidate=by['prepared_async'];assert candidate['prompt_tokens']==1672
        if expected_tokens is None:expected_tokens=candidate['token_ids']
        assert len(expected_tokens)==128
        assert all(r['token_ids']==expected_tokens and r['decode_steps']==127 for r in measured)
        pairs.append({name:math.log(by[name]['decode_seconds']/candidate['decode_seconds']) for name in strata})
    for name in strata:strata[name].append([pair[name] for pair in pairs])
    sources.append(dict(artifact=str(artifact),sha256=hashlib.sha256(artifact.read_bytes()).hexdigest(),pairs=len(pairs)))
results={}
for name,groups in strata.items():
    values=[v for group in groups for v in group]
    rng=random.Random(3091)
    boot=sorted(math.exp(statistics.mean([v for group in groups for v in rng.choices(group,k=len(group))])) for _ in range(20000))
    results[name]=dict(pairs=len(values),paired_geometric_ratio=math.exp(statistics.mean(values)),
        stratified_pair_bootstrap_95_percent=[boot[500],boot[19499]],
        process_geometric_ratios=[math.exp(statistics.mean(group)) for group in groups],
        minimum_pair=math.exp(min(values)))
print(json.dumps(dict(sources=sources,shared_sources_and_generate_function_match=True,
    results=results,pooled_conservative_pass=all(v['stratified_pair_bootstrap_95_percent'][0]>=1.10 for v in results.values()),
    scope='All selected prepared-async corrected-control long pairs included. Bootstrap resamples whole pairs within each process, retaining process sample counts. Excludes withdrawn extra-eval denominators and the different compact candidate. No cross-process population guarantee.'),indent=2))
