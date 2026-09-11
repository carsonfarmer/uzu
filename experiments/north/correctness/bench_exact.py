"""Geometry sweep of the numerically matched branch against native MLX."""
import argparse,json,sys,time,statistics,hashlib
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quantized'))
from reference import ROOT,load
from kernels import inputs
from exact import run_exact
p=argparse.ArgumentParser()
p.add_argument('--layer',type=int,default=1)
p.add_argument('--repetitions',type=int,default=31)
p.add_argument('--output',default='experiments/north/correctness/results/geometry-layer1.json')
a=p.parse_args();model,_=load();layer=model.layers[a.layer];mlp=layer.mlp
cap=ROOT/('work/north-layer' if a.layer==1 else 'work/north-layer7')
meta=json.loads((cap/'metadata.json').read_text())
def bf(name,shape):return mx.array(np.fromfile(cap/(name+'.bin'),dtype=np.uint16)).view(mx.bfloat16).reshape(shape)
x=bf('x',(1,1,2048));ax=bf('ax',(1,1,4096));res=bf('residual',(1,1,2048))
ids=mx.array(meta['experts'],dtype=mx.uint32).reshape(1,1,8)
scores=mx.array(np.fromfile(cap/'scores.bin',dtype=np.float32)).reshape(1,1,8)
data=inputs(mlp,x,ax,ids,layer.self_attn.o_proj.weight)
values=(x,ax,res,ids,scores)
def native(x,ax,res,ids,scores):
    d=mlp.switch_mlp(x,ids)
    return layer.self_attn.o_proj(ax)+(d*scores[...,None]).sum(-2).astype(d.dtype)+res
choices=[('native',native),('native_compiled',mx.compile(native))]
for workers in [64,128,160]:
    for rows in [8,16,32]:
        def fn(x,ax,res,ids,scores,workers=workers,rows=rows):
            return run_exact([x,ax,ids,*data[3:]],scores,res,workers=workers,rows=rows)
        choices.append((f'w{workers}_r{rows}',mx.compile(fn)))
mx.eval(*values,*data)
reference=native(*values);mx.eval(reference)
for name,fn in choices:
    value=fn(*values);mx.eval(value)
    assert bool(mx.all(value==reference).item()),name
    changed=(x*.75,ax,res,ids,scores)
    other=fn(*changed);mx.eval(other)
    assert not bool(mx.all(value==other).item()),name+' stale replay'
    for _ in range(8):mx.eval(fn(*values))
samples=[]
for rep in range(a.repetitions):
    order=list(range(len(choices)));shift=rep%len(order);order=order[shift:]+order[:shift]
    if rep%2:order.reverse()
    for index in order:
        name,fn=choices[index]
        start=time.perf_counter();value=fn(*values);mx.eval(value)
        samples.append(dict(variant=name,repetition=rep,wall_us=(time.perf_counter()-start)*1e6))
        assert bool(mx.all(value==reference).item()),name
out=dict(layer=a.layer,device=mx.device_info(),exact_outputs=True,samples=samples,
    sources={n:hashlib.sha256((ROOT/'experiments/north/quantized'/n).read_bytes()).hexdigest() for n in ['kernel.h','kernels.py','exact.py']})
(ROOT/a.output).write_text(json.dumps(out,indent=2)+'\n')
for name,_ in choices:print(name,round(statistics.median(s['wall_us'] for s in samples if s['variant']==name),3))
