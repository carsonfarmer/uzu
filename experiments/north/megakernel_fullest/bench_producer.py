"""Balanced real-matrix preparation cost: dedicated producer ring versus controls."""
import argparse,hashlib,json,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'additional'))
import mlx.core as mx
from reference import load,ROOT
from prep import run as baseline
from producer_prep import run
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--layers',type=int,nargs='+',default=[1,7,47]);p.add_argument('--runs',type=int,default=16);p.add_argument("--leader",action="store_true");a=p.parse_args();assert a.runs%8==0
model,_=load();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
 def save(r):f.write(json.dumps(r)+'\n');f.flush()
 save(dict(kind='provenance',args=vars(a),mlx=mx.__version__,device=mx.device_info(),sources={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__).resolve(),HERE/'producer_prep.py',HERE.parent/'additional/prep.py',HERE.parent/'additional/prep.h']},scope='Hot constituent wall time including submission/eval; real weights and synthetic activations; not end-to-end or GPU-duration attribution'))
 for index in a.layers:
  layer=model.layers[index];nw=layer.input_layernorm.weight;weights=[getattr(layer.self_attn,k+'_proj').weight for k in ('q','k','v')]+[layer.mlp.gate.weight]
  choices={}
  for rows,rr in ((64,8),(32,4)):
   def b(x,rows=rows,rr=rr):
    p,h=baseline(x,weights,rows,rr,nw);return [p[...,:4096],p[...,4096:4608],p[...,4608:5120],p[...,5120:],h]
   choices[f'prepared_{rows}_{rr}']=mx.compile(b)
  for depth in (1,2):
   def candidate(x,depth=depth,leader=a.leader):
    results=[run(x,w,nw,router=i==3,depth=depth,leader=a.leader) for i,w in enumerate(weights)]
    return [r[0] for r in results]+[results[0][1]]+[r[2] for r in results]
   choices[f'producer_depth{depth}']=mx.compile(candidate)
  mx.random.seed(1521+index);xs=[mx.random.normal((1,1,2048)).astype(mx.bfloat16) for _ in range(3)];mx.eval(xs)
  for name,fn in choices.items():
   for sample,x in enumerate(xs):
    h=layer.input_layernorm(x);refs=[h@w.T for w in weights]+[h];got=fn(x);mx.eval(refs,got)
    unequal=[int(mx.sum(r.view(mx.uint8)!=v.view(mx.uint8)).item()) for r,v in zip(refs,got)]
    errors=sum(int(mx.sum(v).item()) for v in got[5:]);row=dict(kind='correctness',layer=index,variant=name,sample=sample,unequal_bytes=unequal,errors=errors);save(row)
    assert not any(unequal) and errors==0,row
  for fn in choices.values():
   for x in xs:mx.eval(fn(x))
  names=list(choices)
  for rep in range(a.runs):
   order=names[rep%4:]+names[:rep%4]
   if (rep//4)%2:order.reverse()
   for pos,name in enumerate(order):
    start=time.perf_counter();v=choices[name](xs[rep%3]);mx.eval(v);elapsed=time.perf_counter()-start
    save(dict(kind='timing',layer=index,variant=name,repetition=rep,position=pos,wall_seconds=elapsed))
   print(index,rep,'complete',flush=True)
