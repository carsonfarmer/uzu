"""Audit the original short-workload goal and keep long uncertainty explicit."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

here=Path(__file__).resolve().parent
root=here.parents[2]
raw=here/'results/corrected-control-v1.jsonl'
precision=here/'results/precision-long-v1.jsonl'
def run(name,*args):
    return json.loads(subprocess.check_output([sys.executable,str(here/name),*map(str,args)],text=True,cwd=root))
summary=run('summarize_corrected.py',raw)
source=run('audit_control_source.py')
pooled=run('pool_long.py',raw,precision)
assert summary['default_batching']
for prompt in ('short','rust'):
    assert summary['results'][prompt]['conservative_prepared_pass'],prompt
assert summary['results']['long']['byte_gate']
assert source['surrounding_loop_has_no_additional_eval']
out=dict(scope='Original numerical target anchored to historical short Python workload; Rust corroboration and long validation/reporting. No all-context10% requirement or claim.',
    corrected_raw_sha256=hashlib.sha256(raw.read_bytes()).hexdigest(),
    source_control_audit=source,
    short_primary_pass=True,rust_corroboration_pass=True,long_bitwise_pass=True,
    corrected_results=summary['results'],long_pooled=pooled,
    scoped_goal_pass=True,
    attribution='Most runner gain adopts standard MLX async submission; incremental preparation is separately measured against unchanged fused async. No new megakernel claim.')
print(json.dumps(out,indent=2))
