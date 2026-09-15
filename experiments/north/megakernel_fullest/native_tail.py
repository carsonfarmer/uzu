"""Export real primitive inputs, compile variants, measure native GPU durations.

Must be invoked via gpu_run.py; child native process inherits the held lock.
No model throughput or actual simultaneous hardware overlap claim is made here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'quantized'))
import mlx.core as mx
import numpy as np
from reference import ROOT,load
from kernels import inputs
from exact import run_exact
from ready_tail import build_source,U

p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--work',required=True);p.add_argument('--layer',type=int,default=7)
p.add_argument('--workers',type=int,default=32);p.add_argument('--runs',type=int,default=32)
p.add_argument('--storage',choices=['threadgroup','register','tuned_threadgroup','tuned_register','interleaved_threadgroup','interleaved_register'],default='threadgroup')
a=p.parse_args()
if a.storage=='register':
    from register_tail import build_source
if a.storage.startswith('tuned_'):
    from tuned_tail import build_source as tuned_builder
    build_source=lambda:tuned_builder(register=a.storage.endswith('register'))
if a.storage.startswith('interleaved_'):
    from interleaved_tail import build_source as interleaved_builder
    build_source=lambda:interleaved_builder(register=a.storage.endswith('register'))
assert a.runs%8==0
work=Path(a.work).resolve();work.mkdir(parents=True,exist_ok=True)
output=Path(a.output);output.parent.mkdir(parents=True,exist_ok=True)
model,_=load();layer=model.layers[a.layer];nxt=model.layers[a.layer+1]
mx.random.seed(611+a.layer)
x=mx.random.normal((1,1,2048)).astype(mx.bfloat16);ax=mx.random.normal((1,1,4096)).astype(mx.bfloat16)
residual=mx.random.normal((1,1,2048)).astype(mx.bfloat16)
ids=mx.array([3,17,28,45,66,79,91,113],mx.uint32).reshape(1,1,8)
scores=mx.array([.1+j*.09 for j in range(8)],mx.float32).reshape(1,1,8)
data=inputs(layer.mlp,x,ax,ids,layer.self_attn.o_proj.weight)
ref=run_exact(data,scores,residual,workers=160,rows=16);norm=nxt.input_layernorm(ref)
refs=[ref,norm,*[getattr(nxt.self_attn,k+'_proj')(norm) for k in ('q','k','v')],nxt.mlp.gate(norm)]
arrays=data+[scores,residual,nxt.input_layernorm.weight,nxt.self_attn.q_proj.weight,nxt.self_attn.k_proj.weight,nxt.self_attn.v_proj.weight,nxt.mlp.gate.weight]
mx.eval(refs,arrays);mx.synchronize()
names=U['NAMES']+['scores','residual']+U['NEXT_NAMES'];entries=[]
for name,value in zip(names,arrays):
    path=work/(name+'.bin');blob=np.array(value.view(mx.uint8)).tobytes();path.write_bytes(blob)
    entries.append(dict(path=str(path),name=name,bytes=len(blob),sha256=hashlib.sha256(blob).hexdigest()))
h,s=build_source();h='#include "utils.h"\n'+h
args=','.join('device const '+('uint' if n in ('ids','up','gate','down') else 'float' if n=='scores' else 'bfloat')+'* '+n for n in names)
args+=',device uint* workspace,uint3 thread_position_in_threadgroup [[thread_position_in_threadgroup]],uint3 threadgroup_position_in_grid [[threadgroup_position_in_grid]],uint simdgroup_index_in_threadgroup [[simdgroup_index_in_threadgroup]],uint thread_index_in_simdgroup [[thread_index_in_simdgroup]]'
mlxsrc=Path('/Users/carsonfarmer/Developer/Personal/uzu-metal-lab/work/mlx-src')
variants=[]
for name,prefix,early in [('direct',0,False),('late',128,False),('early',128,True)]:
    defines=dict(ROWS=32,PREFIX=prefix,ROUTER_ROWS=8,PREP_TASKS=176,INTERLEAVE='true',PREFETCH=str(early).lower(),AUDIT='false')
    src=h+'\n'+'\n'.join(f'#define {k} {v}' for k,v in defines.items())+f'\nkernel void check({args}){{\n{s}\n}}'
    metal=work/(name+'.metal');metal.write_text(src);air=work/(name+'.air');lib=work/(name+'.metallib')
    subprocess.run(['xcrun','metal','-std=metal4.1','-fmetal-math-mode=safe','-I',str(mlxsrc/'mlx/backend/metal/kernels'),'-I',str(mlxsrc),'-c',str(metal),'-o',str(air)],check=True)
    subprocess.run(['xcrun','metallib',str(air),'-o',str(lib)],check=True)
    variants.append(dict(name=name,library=str(lib),workers=a.workers,source_sha256=hashlib.sha256(src.encode()).hexdigest()))
exe=work/'native_runner'
subprocess.run(['swiftc',str(HERE/'native_runner.swift'),'-o',str(exe)],check=True)
manifest=dict(variants=variants,inputs=entries,outputBytes=U['SIZE']*4,repetitions=a.runs,output=str(work/'result'))
manifest_path=work/'manifest.json';manifest_path.write_text(json.dumps(manifest,indent=2))
with output.open('w') as f:
    f.write(json.dumps(dict(kind='export_provenance',args=vars(a),manifest=manifest,sources={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in HERE.iterdir() if q.suffix in ('.py','.swift')},scope='Native exported real-weight primitive, synthetic activations. Includes zero-fill. No model throughput claim.'))+'\n');f.flush()
    subprocess.run([str(exe),str(manifest_path)],stdout=f,check=True)
    for variant in variants:
        blob=(work/('result-'+variant['name']+'.bin')).read_bytes()
        offsets=[U['OUT']*4,U['NORM']*4,U['PREP']*4,U['PREP']*4+4096*2,U['PREP']*4+4608*2,U['PREP']*4+5120*2]
        unequal=[]
        for r,start in zip(refs,offsets):
            expected=np.array(r.view(mx.uint8)).tobytes();got=blob[start:start+len(expected)]
            assert len(got)==len(expected)
            unequal.append(sum(a!=b for a,b in zip(expected,got)))
        state=np.frombuffer(blob,dtype=np.uint32)
        row=dict(kind='native_correctness',variant=variant['name'],unequal_bytes=unequal,errors=int(state[U['ERROR']]),prep_completed=int(state[U['PREFETCH_DONE']]))
        f.write(json.dumps(row)+'\n');f.flush();print(row,flush=True)
        assert not any(unequal) and not row['errors'] and row['prep_completed']==176,row
