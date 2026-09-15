"""CPU-only compile of the real branch-tail staging source."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
from ready_tail import build_source,U
h,s=build_source()
h='#include "utils.h"\n'+h
names=U['NAMES']+['scores','residual']+U['NEXT_NAMES']
args=','.join('device const '+('uint' if n in ('ids','up','gate','down') else 'float' if n=='scores' else 'bfloat')+'* '+n for n in names)
args+=',device uint* workspace,uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]'
results=[]
with tempfile.TemporaryDirectory(prefix='north-tail-compile-') as d:
    for rows,prefix in ((32,128),(32,256),(64,128)):
        for prefetch in (False,True):
            defines=dict(ROWS=rows,PREFIX=prefix,ROUTER_ROWS=8,PREP_TASKS=5120//rows+16,INTERLEAVE='true',PREFETCH=str(prefetch).lower(),AUDIT='true')
            src=h+'\n'+'\n'.join(f'#define {k} {v}' for k,v in defines.items())+f'\nkernel void check({args}){{\n{s}\n}}'
            path=Path(d)/'check.metal';path.write_text(src)
            r=subprocess.run(['xcrun','metal','-std=metal3.2','-I','/Users/carsonfarmer/Developer/Personal/uzu-metal-lab/work/mlx-src/mlx/backend/metal/kernels','-I','/Users/carsonfarmer/Developer/Personal/uzu-metal-lab/work/mlx-src','-c',str(path),'-o',str(Path(d)/'check.air')],capture_output=True,text=True)
            results.append(dict(rows=rows,prefix=prefix,prefetch=prefetch,returncode=r.returncode,diagnostics=r.stderr,sha256=hashlib.sha256(src.encode()).hexdigest()))
print(json.dumps(dict(scope='CPU compilation only; no numerical or runtime progress claim',variants=results),indent=2))
raise SystemExit(any(x['returncode'] for x in results))
