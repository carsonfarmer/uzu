"""Exact component, publication, exactly-once, and occupancy stress checks."""
import argparse,hashlib,json,sys,time,statistics
from pathlib import Path
import numpy as np
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quantized'))
from reference import ROOT,load
from kernels import inputs
from exact import run_exact
from branch import run_scheduled
from persistent.prefetch import run_staged
from persistent.fast import run_fast
p=argparse.ArgumentParser()
p.add_argument('--layer',type=int,default=1)
p.add_argument('--mode',choices=['ready','staged','fast'],default='ready')
p.add_argument('--repetitions',type=int,default=15)
p.add_argument('--workers',type=int,nargs='+',default=[1,20,64,128,256])
p.add_argument('--output',default='experiments/north/persistent/check.json')
a=p.parse_args();model,_=load();layer=model.layers[a.layer];mlp=layer.mlp
cap=ROOT/('work/north-layer' if a.layer==1 else 'work/north-layer7')
meta=json.loads((cap/'metadata.json').read_text())
def bf(name,shape):return mx.array(np.fromfile(cap/(name+'.bin'),dtype=np.uint16)).view(mx.bfloat16).reshape(shape)
x=bf('x',(1,1,2048));ax=bf('ax',(1,1,4096));res=bf('residual',(1,1,2048))
ids=mx.array(meta['experts'],dtype=mx.uint32).reshape(1,1,8)
scores=mx.array(np.fromfile(cap/'scores.bin',dtype=np.float32)).reshape(1,1,8)
data=inputs(mlp,x,ax,ids,layer.self_attn.o_proj.weight)
mx.eval(*data,scores,res)
choices=[]
for workers in a.workers:
 for fine in [False,True]:
  def fn(x,ax,ids,scores,res,workers=workers,fine=fine):
   if a.mode=='fast':return run_fast([x,ax,ids,*data[3:]],scores,res,workers=workers,fine=fine,return_parts=True)
   if a.mode=='staged':return run_staged([x,ax,ids,*data[3:]],scores,res,workers=workers,prefetch=fine,return_parts=True)
   return run_scheduled([x,ax,ids,*data[3:]],scores,res,workers=workers,fine=fine,return_parts=True)
  label=('prefetch' if fine else 'after_ready') if a.mode=='staged' else ('fine' if fine else 'all_experts')
  choices.append((f'w{workers}_'+label,mx.compile(fn)))
checks=[];samples=[]
for rep in range(a.repetitions):
 # Repeated submissions use varying activations and selected experts. This
 # catches stale workspace flags and accidentally cached compiled results.
 xx=(x*(1+rep*.013)).astype(mx.bfloat16)
 ii=((ids+rep*7)%128).astype(mx.uint32)
 expected=run_exact([xx,ax,ii,*data[3:]],scores,res,workers=160,rows=16,return_parts=True)
 mx.eval(*expected)
 # Adjacent repetitions reverse the same order. Rotating on every repetition
 # and then reversing cancels out when there are exactly two variants.
 shift=(rep//2)%len(choices)
 order=choices[shift:]+choices[:shift]
 if rep%2:order=list(reversed(order))
 for name,fn in order:
  start=time.perf_counter();got=fn(xx,ax,ii,scores,res);mx.eval(*got)
  elapsed=(time.perf_counter()-start)*1e6
  unequal=[int(mx.sum(v.reshape(-1).view(mx.uint8)!=r.reshape(-1).view(mx.uint8)).item()) for v,r in zip(got[:4],expected)]
  state=np.array(got[4]);visits=state[128:128+672]
  row=dict(repetition=rep,variant=name,unequal_bytes=unequal,errors=int(state[105]),
      task_visits_min=int(visits.min()),task_visits_max=int(visits.max()),
      first_down_completed_up_tiles=int(state[107]),completed_up_tiles=int(state[108]),early_prefetch_tiles=int(state[110]))
  checks.append(row)
  assert unequal==[0]*4 and row['errors']==0 and np.all(visits==1),row
  if rep>1:samples.append(dict(repetition=rep,variant=name,wall_us=elapsed))
  print(row,flush=True)
out=dict(device=mx.device_info(),args=vars(a),checks=checks,samples=samples,
 sources={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in [Path(__file__),Path(__file__).with_name('branch.py'),Path(__file__).with_name('down.h'),Path(__file__).with_name('prefetch.py'),Path(__file__).with_name('fast.py'),ROOT/'experiments/north/quantized/kernel.h']})
(ROOT/a.output).write_text(json.dumps(out,indent=2)+'\n')
for name,_ in choices:print(name,statistics.median(s['wall_us'] for s in samples if s['variant']==name),flush=True)
