"""Compare quantized schedules and native MLX in the same runtime."""
import argparse,json,sys,time,statistics
from pathlib import Path
import mlx.core as mx
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference import ROOT,load
from kernels import inputs,run
p=argparse.ArgumentParser()
p.add_argument('--layer',type=int,default=1)
p.add_argument('--capture',default='work/north-layer')
p.add_argument('--output',default='experiments/north/quantized/results/pilot.json')
p.add_argument('--repetitions',type=int,default=31)
args=p.parse_args()
model,_=load()
layer=model.layers[args.layer];mlp=layer.mlp
capture=ROOT/args.capture;meta=json.loads((capture/'metadata.json').read_text())
import numpy as np
def bf(name,shape):
 return mx.array(np.fromfile(capture/(name+'.bin'),dtype=np.uint16)).view(mx.bfloat16).reshape(shape)
x=bf('x',(1,1,2048));ax=bf('ax',(1,1,4096));residual=bf('residual',(1,1,2048))
ids=mx.array(meta['experts'],dtype=mx.uint32).reshape(1,1,8)
scores=mx.array(np.fromfile(capture/'scores.bin',dtype=np.float32)).reshape(8)
data=inputs(mlp,x,ax,ids,layer.self_attn.o_proj.weight)
def native():
 routed=mlp.switch_mlp(x,ids)
 moe=(routed*scores.reshape(1,1,8,1)).sum(-2).astype(routed.dtype)
 return (layer.self_attn.o_proj(ax)+moe)+residual
choices=[('native',native),('phased',lambda:run(data,scores,residual,phased=True))]
for w in [20,64,128,256]:
 choices.append((f'static{w}',lambda w=w:run(data,scores,residual,workers=w)))
choices.append(('interleaved128',lambda:run(data,scores,residual,workers=128,interleave=True)))
mx.eval(*data,scores,residual)
ref=native();mx.eval(ref)
parts=run(data,scores,residual,phased=True,return_parts=True);mx.eval(*parts)
checks={}
for name,fn in choices:
 o=fn();mx.eval(o)
 if name!='native':
  assert bool(mx.all(o==parts[0]).item()),name
 d=o.astype(mx.float32)-ref.astype(mx.float32)
 checks[name]={'max_abs':float(mx.max(mx.abs(d)).item()),'relative_l2':float((mx.linalg.norm(d)/mx.linalg.norm(ref.astype(mx.float32))).item())}
 if name.startswith('static') or name.startswith('interleaved'):
  w=int(''.join(c for c in name if c.isdigit()))
  candidate=run(data,scores,residual,workers=w,interleave=name.startswith('interleaved'),return_parts=True);mx.eval(*candidate)
  assert all(bool(mx.all(a==b).item()) for a,b in zip(parts,candidate)),name
 for _ in range(8):mx.eval(fn())
samples=[]
for rep in range(args.repetitions):
 order=list(range(len(choices)));shift=rep%len(order);order=order[shift:]+order[:shift]
 if rep%2:order.reverse()
 for i in order:
  name,fn=choices[i]
  start=time.perf_counter();out=fn();mx.eval(out)
  samples.append(dict(variant=name,repetition=rep,wall_us=(time.perf_counter()-start)*1e6))
  assert bool(mx.all(out==(ref if name=='native' else parts[0])).item()),name
out={'layer':args.layer,'capture':args.capture,'checks':checks,'samples':samples,
 'device':mx.device_info(),'note':'Native affine 4bit/group64 weights stay packed; real captured routing fixed; same MLX process, interleaved variants. Wall includes lazy graph creation/evaluation, no host sampling. Operator selection still distinct from whole-model decode.'}
(ROOT/args.output).write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(checks,indent=2))
for name,_ in choices:print(name,statistics.median(s['wall_us'] for s in samples if s['variant']==name))
