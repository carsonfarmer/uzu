"""Audit one decode artifact without importing MLX or using the GPU."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

p=argparse.ArgumentParser()
p.add_argument('artifact',type=Path)
a=p.parse_args()
rows=[json.loads(line) for line in a.artifact.read_text().splitlines()]
provenance=rows[0]
assert provenance['kind']=='provenance'
root=Path(__file__).resolve().parents[3]
source_matches={name:(root/name).exists() and hashlib.sha256((root/name).read_bytes()).hexdigest()==digest for name,digest in provenance['sources'].items()}
assert all(source_matches.values()),source_matches
checks=defaultdict(list)
timing=defaultdict(dict)
for row in rows[1:]:
    if row['kind']=='correctness':
        assert row['unequal_bytes']==0,row
        checks[(row['prompt'],row['variant'])].append(row['step'])
    if row['kind']=='generation' and not row['warmup']:
        key=(row['prompt'],row['repetition'])
        assert row['variant'] not in timing[key]
        timing[key][row['variant']]=row
args=provenance['args'];results={}
for prompt in args['prompts']:
    scope_ok=all(checks[(prompt,name)]==list(range(args['check_steps'])) for name in ('exact','candidate')) and args['check_steps']>=127
    ratios=[];control=[];candidate=[];lengths=[]
    for rep in range(args['runs']):
        pair=timing[(prompt,rep)]
        assert set(pair)=={'exact','candidate'},(prompt,rep)
        c,v=pair['exact'],pair['candidate']
        assert c['token_ids']==v['token_ids'],(prompt,rep)
        assert len(c['step_seconds'])==c['decode_steps'] and len(v['step_seconds'])==v['decode_steps']
        tc=c['decode_steps']/sum(c['step_seconds']);tv=v['decode_steps']/sum(v['step_seconds'])
        assert math.isclose(tc,c['decode_tokens_per_second']) and math.isclose(tv,v['decode_tokens_per_second'])
        ratios.append(tv/tc);control.append(tc);candidate.append(tv);lengths.append(c['decode_steps'])
    logs=[math.log(v) for v in ratios]
    rng=random.Random(9347)
    bootstrap=sorted(math.exp(statistics.mean(rng.choices(logs,k=len(logs)))) for _ in range(10000))
    ratio=math.exp(statistics.mean(logs))
    results[prompt]=dict(full_logit_scope_pass=scope_ok,paired_ratios=ratios,
        paired_geometric_ratio=ratio,paired_bootstrap_95_percent=[bootstrap[250],bootstrap[9749]],
        median_control_tps=statistics.median(control),median_candidate_tps=statistics.median(candidate),
        decode_steps=lengths,observed_ten_percent=ratio>=1.10,
        conservative_ten_percent=scope_ok and len(ratios)>=6 and min(lengths)>=127 and bootstrap[250]>=1.10)
print(json.dumps(dict(artifact=str(a.artifact),artifact_sha256=hashlib.sha256(a.artifact.read_bytes()).hexdigest(),
    source_hashes_match=True,results=results,
    note='Paired bootstrap resamples whole run pairs; it does not treat individual tokens as independent. A confidence interval from six pairs is still limited evidence. No cross-hardware or out-of-scope correctness claim.'),indent=2))
