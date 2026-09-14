"""Constituent comparison, not an end-to-end speed claim."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference import load
from prep import run

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--repetitions',type=int,default=31)
p.add_argument('--layers',type=int,nargs='+',default=[1,7,47])
a=p.parse_args()
model,_=load()
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
def bits(x):
    return np.array(x.view(mx.uint16)).tobytes()
with out.open('w') as f:
    def save(row):
        f.write(json.dumps(row)+'\n');f.flush()
    save(dict(kind='provenance',mlx=mx.__version__,device=mx.device_info(),args=vars(a),
        sources={str(q):hashlib.sha256(q.read_bytes()).hexdigest() for q in Path(__file__).parent.glob('*') if q.is_file()},
        scope='Real model weights, random BF16 inputs; wall latency including submission and eval. Hot resident constituent test, not decode timing.'))
    for index in a.layers:
        layer=model.layers[index]
        weights=[getattr(layer.self_attn,k+'_proj').weight for k in ('q','k','v')]+[layer.mlp.gate.weight]
        assert all(w.dtype==mx.bfloat16 for w in weights)
        mx.random.seed(710+index)
        xs=[mx.random.normal((1,1,2048)).astype(mx.bfloat16) for _ in range(3)]
        mx.eval(xs)
        def native(x):
            h=layer.input_layernorm(x)
            return [h@w.T for w in weights]+[h]
        choices={'native':native,'native_compiled':mx.compile(native)}
        for rows in (32,64,128,256):
            for rr in (4,8,16):
                for rms in (False,True):
                    def candidate(x,rows=rows,rr=rr,rms=rms):
                        if rms:
                            prep,h=run(x,weights,rows,rr,layer.input_layernorm.weight)
                        else:
                            h=layer.input_layernorm(x)
                            prep=run(h,weights,rows,rr)[0]
                        return [prep[...,:4096],prep[...,4096:4608],prep[...,4608:5120],prep[...,5120:],h]
                    choices[f'r{rows}_t{rr}_rms{int(rms)}']=mx.compile(candidate)
        valid={}
        for name,fn in choices.items():
            matches=[]
            for x in xs:
                ref=native(x);got=fn(x);mx.eval(ref,got)
                matches.append([bits(r)==bits(g) for r,g in zip(ref,got)])
            save(dict(kind='correctness',layer=index,variant=name,byte_equal=matches))
            if all(all(v) for v in matches):valid[name]=fn
            else:print('REJECT',index,name,matches,flush=True)
        for fn in valid.values():
            for _ in range(5):mx.eval(fn(xs[0]))
        samples={name:[] for name in valid}
        for rep in range(a.repetitions):
            names=list(valid);shift=rep%len(names);names=names[shift:]+names[:shift]
            if rep%2:names.reverse()
            for name in names:
                x=xs[rep%len(xs)]
                start=time.perf_counter();y=valid[name](x);mx.eval(y)
                us=(time.perf_counter()-start)*1e6
                samples[name].append(us)
                save(dict(kind='timing',layer=index,variant=name,repetition=rep,wall_us=us))
        medians={name:statistics.median(v) for name,v in samples.items()}
        print(index,sorted(medians.items(),key=lambda x:x[1]),flush=True)
        save(dict(kind='summary',layer=index,median_us=medians))
