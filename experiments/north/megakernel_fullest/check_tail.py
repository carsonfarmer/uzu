"""Bitwise intermediate and bounded-progress gate before any tail timing."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent));sys.path.insert(0,str(HERE.parent/'quantized'))
import mlx.core as mx
from reference import load,ROOT
from kernels import inputs
from exact import run_exact
from ready_tail import run,U

p=argparse.ArgumentParser();p.add_argument('--output',required=True)
p.add_argument('--layers',type=int,nargs='+',default=[1,7,47])
p.add_argument('--workers',type=int,nargs='+',default=[1,20,32,64])
p.add_argument('--storage',choices=['threadgroup','register'],default='threadgroup')
p.add_argument('--samples',type=int,default=3);a=p.parse_args()
model,_=load();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
    def save(row):f.write(json.dumps(row)+'\n');f.flush()
    sources=[Path(__file__).resolve(),HERE/'ready_tail.py',HERE/'staged_prep.py',HERE/'gpu_run.py',HERE.parent/'full_layer/tail_prep.py',HERE.parent/'persistent/down.h',HERE.parent/'quantized/kernel.h',HERE.parent/'additional/prep.h']
    save(dict(kind='provenance',args=vars(a),device=mx.device_info(),mlx=mx.__version__,sources={str(s.relative_to(ROOT)):hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},scope='Real weights; synthetic branch inputs; exact intermediate bytes and visit/progress gate, not decode timing'))
    for index in a.layers:
        layer=model.layers[index];next_layer=model.layers[index+1]
        mx.random.seed(719+index)
        for sample in range(a.samples):
            x=mx.random.normal((1,1,2048)).astype(mx.bfloat16)
            ax=mx.random.normal((1,1,4096)).astype(mx.bfloat16)
            residual=mx.random.normal((1,1,2048)).astype(mx.bfloat16)
            ids=mx.array([(j*13+sample*7)%128 for j in range(8)],mx.uint32).reshape(1,1,8)
            scores=mx.array([.1+j*.09 for j in range(8)],mx.float32).reshape(1,1,8)
            data=inputs(layer.mlp,x,ax,ids,layer.self_attn.o_proj.weight)
            o=run_exact(data,scores,residual,workers=160,rows=16);h=next_layer.input_layernorm(o)
            refs=[o,h,*[getattr(next_layer.self_attn,k+'_proj')(h) for k in ('q','k','v')],next_layer.mlp.gate(h)];mx.eval(refs)
            for workers in a.workers:
                for prefetch in (False,True):
                    got=run(data,scores,residual,next_layer,workers=workers,prefetch=prefetch,audit=True,storage=a.storage);mx.eval(got)
                    unequal=[int(mx.sum(r.view(mx.uint8)!=v.view(mx.uint8)).item()) for r,v in zip(refs,got)]
                    state=got[-1].tolist();total=U['FRONT_TASKS']+U['DOWN_TASKS']+U['JOIN_TASKS']+1+176
                    visits=state[U['VISITS']:U['VISITS']+total]
                    row=dict(kind='correctness',layer=index,sample=sample,workers=workers,prefetch=prefetch,unequal_bytes=unequal,errors=state[U['ERROR']],prep_completed=state[U['PREFETCH_DONE']],early_reservations=state[U['EARLY_ROUNDS']],visits_min=min(visits),visits_max=max(visits))
                    save(row);print(row,flush=True)
                    assert not any(unequal) and not row['errors'] and min(visits)==max(visits)==1 and row['prep_completed']==176,row
                    assert bool(row['early_reservations'])==prefetch,row
