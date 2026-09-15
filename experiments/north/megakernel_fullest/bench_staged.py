"""Consumed prefix cost screen; real weights, synthetic activations, no overlap claim."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'additional'))
import mlx.core as mx
from reference import load,ROOT
from prep import run as baseline
from staged_prep import run as staged

p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--layers',type=int,nargs='+',default=[1,7,47])
p.add_argument('--runs',type=int,default=14);a=p.parse_args()
assert a.runs%14==0
model,_=load()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
    def save(row):f.write(json.dumps(row)+'\n');f.flush()
    sources=[Path(__file__).resolve(),HERE/'staged_prep.py',HERE/'gpu_run.py',HERE.parent/'additional/prep.py',HERE.parent/'additional/prep.h',HERE.parent/'reference.py']
    save(dict(kind='provenance',args=vars(a),mlx=mx.__version__,device=mx.device_info(),
        sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
        scope='Hot primitive wall latency including host submission/eval. Real model weights and synthetic BF16 inputs. No physical or cross-layer overlap claim.'))
    for index in a.layers:
        layer=model.layers[index];nw=layer.input_layernorm.weight
        weights=[getattr(layer.self_attn,k+'_proj').weight for k in ('q','k','v')]+[layer.mlp.gate.weight]
        choices={'prepared':mx.compile(lambda x:baseline(x,weights,64,8,nw))}
        for rows,prefix in ((32,128),(32,256),(64,128)):
            for early in (False,True):
                choices[f'r{rows}_p{prefix}_early{int(early)}']=mx.compile(
                    lambda x,rows=rows,prefix=prefix,early=early:staged(x,weights,nw,rows,8,prefix,early))
        mx.random.seed(982+index)
        xs=[mx.random.normal((1,1,2048)).astype(mx.bfloat16) for _ in range(3)];mx.eval(xs)
        valid={}
        for name,fn in choices.items():
            ok=True
            for sample,x in enumerate(xs):
                h=layer.input_layernorm(x);ref=[mx.concatenate([h@w.T for w in weights],axis=-1),h]
                got=fn(x);mx.eval(ref,got)
                unequal=[int(mx.sum(r.view(mx.uint8)!=g.view(mx.uint8)).item()) for r,g in zip(ref,got)]
                save(dict(kind='correctness',layer=index,variant=name,sample=sample,unequal_bytes=unequal))
                ok=ok and not any(unequal)
            if ok:valid[name]=fn
            else:print('REJECT',index,name,flush=True)
        assert len(valid)==len(choices),'Retain failed evidence; fix before timing'
        for fn in choices.values():
            for x in xs:mx.eval(fn(x))
        names=list(choices)
        for rep in range(a.runs):
            shift=rep%len(names);order=names[shift:]+names[:shift]
            if (rep//len(names))%2:order.reverse()
            for position,name in enumerate(order):
                start=time.perf_counter();got=choices[name](xs[rep%3]);mx.eval(got)
                elapsed=time.perf_counter()-start
                save(dict(kind='timing',layer=index,variant=name,repetition=rep,order=position,wall_seconds=elapsed))
            print(index,rep,'complete',flush=True)
