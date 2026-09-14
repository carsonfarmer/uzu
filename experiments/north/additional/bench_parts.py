"""Screen branch constituents and head; report hot wall latency, not GPU time."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time
import mlx.core as mx
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE.parent))
sys.path.insert(0,str(HERE.parent/'quantized'))
from reference import load
from kernels import inputs
from exact import front,down,run_exact

p=argparse.ArgumentParser()
p.add_argument('--output',required=True)
p.add_argument('--layers',type=int,nargs='+',default=[1,7,47])
p.add_argument('--repetitions',type=int,default=51)
a=p.parse_args()
model,_=load()
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
with path.open('w') as f:
    def save(row):f.write(json.dumps(row)+'\n');f.flush()
    save(dict(kind='provenance',args=vars(a),mlx=mx.__version__,device=mx.device_info(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='Random BF16 activations, actual model weights, deterministic top-8 routing; hot constituent wall latency with evaluation. Not additive stage timing or end-to-end performance.'))
    for index in a.layers:
        layer=model.layers[index];mlp=layer.mlp
        mx.random.seed(992+index)
        x=mx.random.normal((1,1,2048)).astype(mx.bfloat16)
        ax=mx.random.normal((1,1,4096)).astype(mx.bfloat16)
        residual=mx.random.normal(x.shape).astype(x.dtype)
        gates=mlp.gate_act(mlp.gate(x).astype(mx.float32))
        ids=mx.argpartition(-gates,kth=7,axis=-1)[...,:8]
        scores=mx.take_along_axis(gates,ids,axis=-1)
        data=inputs(mlp,x,ax,ids,layer.self_attn.o_proj.weight)
        def native_front(x,ax):
            xx=mx.expand_dims(x,(-2,-3))
            u=mlp.switch_mlp.up_proj(xx,ids)
            g=mlp.switch_mlp.gate_proj(xx,ids)
            return [mlp.switch_mlp.activation(u,g).reshape(8,768),layer.self_attn.o_proj(ax)]
        def exact_front(x,ax):
            return front(inputs=[x,ax,*data[2:]],template=[('WORKERS',160),('INTERLEAVE',True)],grid=(160*256,1,1),threadgroup=(256,1,1),output_shapes=[(8,768),(1,1,2048)],output_dtypes=[mx.bfloat16,mx.bfloat16])
        hidden,attention=native_front(x,ax);mx.eval(hidden,attention)
        def native_down(h):
            d=mlp.switch_mlp.down_proj(h.reshape(1,1,8,1,768),ids).reshape(1,1,8,2048)
            moe=(d*scores[...,None]).sum(-2).astype(d.dtype)
            return attention+moe+residual
        def exact_down(h,rows):
            return down(inputs=[h,ids,*data[9:12],scores,attention,residual],template=[('ROWS',rows)],grid=(16384,1,1),threadgroup=(rows*8,1,1),output_shapes=[(1,1,2048),(8,2048)],output_dtypes=[mx.bfloat16,mx.bfloat16])[0]
        tests={
            'front_native':(mx.compile(native_front),(x,ax),'front'),
            'front_exact':(mx.compile(exact_front),(x,ax),'front'),
            'down_native':(mx.compile(native_down),(hidden,),'down'),
            'branch_native':(mx.compile(lambda x,ax: layer.self_attn.o_proj(ax)+(mlp.switch_mlp(x,ids)*scores[...,None]).sum(-2).astype(x.dtype)+residual),(x,ax),'branch'),
            'branch_exact':(mx.compile(lambda x,ax:run_exact([x,ax,*data[2:]],scores,residual,workers=160,rows=16)),(x,ax),'branch'),
            'head_native':(mx.compile(lambda x:model.model.embed_tokens.as_linear(model.model.norm(x))*model.model.args.logit_scale),(x,),'head')}
        for rows in (8,16,32):
            tests[f'down_exact_r{rows}']=(mx.compile(lambda h,rows=rows:exact_down(h,rows)),(hidden,),'down')
        reference={}
        for name,(fn,args,kind) in tests.items():
            got=fn(*args);mx.eval(got)
            values=got if isinstance(got,list) else [got]
            raw=[np.array(v.view(mx.uint16)).tobytes() for v in values]
            if kind not in reference:reference[kind]=raw
            equal=raw==reference[kind]
            save(dict(kind='correctness',layer=index,variant=name,bitwise=equal))
            assert equal,(index,name)
            for _ in range(5):mx.eval(fn(*args))
        samples={name:[] for name in tests}
        for rep in range(a.repetitions):
            order=list(tests);shift=rep%len(order);order=order[shift:]+order[:shift]
            if rep%2:order.reverse()
            for name in order:
                fn,args,_=tests[name]
                start=time.perf_counter();got=fn(*args);mx.eval(got)
                us=(time.perf_counter()-start)*1e6;samples[name].append(us)
                save(dict(kind='timing',layer=index,variant=name,repetition=rep,wall_us=us))
        medians={name:statistics.median(v) for name,v in samples.items()}
        save(dict(kind='summary',layer=index,median_us=medians))
        print(index,medians,flush=True)
