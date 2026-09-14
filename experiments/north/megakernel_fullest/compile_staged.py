"""CPU-only Metal compilation; never creates a GPU device or imports MLX."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from staged_prep import build_source

header,body=build_source()
args=','.join('device '+('const ' if n not in ('prep','norm') else '')+'bfloat* '+n
              for n in ('x','qw','kw','vw','rw','nw','prep','norm'))
args+=',uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]'
rows=[]
with tempfile.TemporaryDirectory(prefix='north-staged-compile-') as d:
    for r,p in ((32,128),(32,256),(64,128)):
        for early in (False,True):
            src=f'{header}\n#define ROWS {r}\n#define ROUTER_ROWS 8\n#define PREFIX {p}\n#define EARLY {str(early).lower()}\nkernel void check({args}){{\n{body}\n}}'
            path=Path(d)/'check.metal';path.write_text(src)
            result=subprocess.run(['xcrun','metal','-std=metal3.2','-c',str(path),'-o',str(Path(d)/'check.air')],capture_output=True,text=True)
            rows.append(dict(rows=r,prefix=p,early=early,returncode=result.returncode,diagnostics=result.stderr,sha256=hashlib.sha256(src.encode()).hexdigest()))
print(json.dumps(dict(scope='CPU compile only; no numerical, runtime, or overlap validation',variants=rows),indent=2))
raise SystemExit(any(r['returncode'] for r in rows))
