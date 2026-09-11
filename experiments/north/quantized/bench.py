"""Same-runtime, eager/compiled comparison using real packed North weights."""
import argparse,hashlib,json,sys,time,statistics,subprocess,platform
from pathlib import Path
import mlx.core as mx
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference import ROOT,load
from kernels import inputs,run

p=argparse.ArgumentParser()
p.add_argument('--layer',type=int,default=1)
p.add_argument('--capture',default='work/north-layer')
p.add_argument('--output',default='experiments/north/quantized/results/sweep.json')
p.add_argument('--repetitions',type=int,default=31)
p.add_argument('--chunks',type=int,nargs='+',default=[32,64,128])
p.add_argument('--selected',action='store_true',help='Frozen c64/interleaved128 candidate plus controls')
args=p.parse_args()
if args.selected:args.chunks=[64]
power_before=subprocess.check_output(['pmset','-g','therm'],text=True)
model,_=load()
layer=model.layers[args.layer];mlp=layer.mlp
capture=ROOT/args.capture;meta=json.loads((capture/'metadata.json').read_text())
assert meta['layer']==args.layer

def bf(name,shape):
    return mx.array(np.fromfile(capture/(name+'.bin'),dtype=np.uint16)).view(mx.bfloat16).reshape(shape)
x=bf('x',(1,1,2048));ax=bf('ax',(1,1,4096));residual=bf('residual',(1,1,2048))
ids=mx.array(meta['experts'],dtype=mx.uint32).reshape(1,1,8)
scores=mx.array(np.fromfile(capture/'scores.bin',dtype=np.float32)).reshape(8)
data=inputs(mlp,x,ax,ids,layer.self_attn.o_proj.weight)

def native(x,ax,residual,ids,scores):
    routed=mlp.switch_mlp(x,ids)
    moe=(routed*scores.reshape(1,1,8,1)).sum(-2).astype(routed.dtype)
    return (layer.self_attn.o_proj(ax)+moe)+residual

# Inputs are explicit compiler arguments: nothing is benchmarked as a closed
# constant graph. A changed-input check below also catches stale replay.
choices=[('native_eager',native,None),('native_compiled',mx.compile(native),None)]
for chunk in args.chunks:
    jobs=8*(768//chunk)
    configs=[('phased',dict(phased=True))]
    configs += [(f'static{w}',dict(workers=w)) for w in sorted(set([20,64,min(128,jobs+64),jobs+64]))]
    configs += [(f'interleaved{min(128,jobs+64)}',dict(workers=min(128,jobs+64),interleave=True))]
    for name,kw in configs:
        def candidate(x,ax,residual,ids,scores,kw=kw,chunk=chunk):
            return run([x,ax,ids,*data[3:]],scores,residual,chunk=chunk,**kw)
        choices += [(f'c{chunk}_{name}_eager',candidate,(chunk,kw)),
                    (f'c{chunk}_{name}_compiled',mx.compile(candidate),(chunk,kw))]

if args.selected:
    keep={'native_eager','native_compiled','c64_phased_compiled',
          'c64_static20_compiled','c64_static128_compiled','c64_interleaved128_compiled'}
    choices=[choice for choice in choices if choice[0] in keep]
values=(x,ax,residual,ids,scores)
mx.eval(*data,scores,residual)
ref=native(*values);mx.eval(ref)
parts={c:run(data,scores,residual,phased=True,return_parts=True,chunk=c) for c in args.chunks}
mx.eval(*[a for part in parts.values() for a in part])
checks={};expected={}
for name,fn,config in choices:
    o=fn(*values);mx.eval(o);expected[name]=o
    if config:
        chunk,kw=config
        assert bool(mx.all(o==parts[chunk][0]).item()),name
        candidate=run(data,scores,residual,return_parts=True,chunk=chunk,**kw)
        mx.eval(*candidate)
        assert all(bool(mx.all(a==b).item()) for a,b in zip(parts[chunk],candidate)),name
    d=o.astype(mx.float32)-ref.astype(mx.float32)
    checks[name]={'max_abs':float(mx.max(mx.abs(d)).item()),
        'relative_l2':float((mx.linalg.norm(d)/mx.linalg.norm(ref.astype(mx.float32))).item()),
        'finite':bool(mx.all(mx.isfinite(o)).item())}
    assert checks[name]['finite'],name
    changed=list(values);changed[0]=x*mx.array(0.75,dtype=x.dtype)
    alt=fn(*changed);mx.eval(alt)
    assert not bool(mx.all(alt==o).item()),name+' stale input'
    for _ in range(8):mx.eval(fn(*values))

samples=[]
for rep in range(args.repetitions):
    order=list(range(len(choices)));shift=rep%len(order);order=order[shift:]+order[:shift]
    if rep%2:order.reverse()
    for i in order:
        name,fn,_=choices[i]
        start=time.perf_counter();out=fn(*values);mx.eval(out)
        samples.append(dict(variant=name,repetition=rep,wall_us=(time.perf_counter()-start)*1e6))
        assert bool(mx.all(out==expected[name]).item()),name
out={'layer':args.layer,'capture':args.capture,'checks':checks,'samples':samples,
    'device':mx.device_info(),
    'mlx':mx.__version__,'macos':platform.mac_ver()[0],
    'thermal_before':power_before,
    'thermal_after':subprocess.check_output(['pmset','-g','therm'],text=True),
    'power_source':subprocess.check_output(['pmset','-g','batt'],text=True).splitlines()[0],
    'source_sha256':{f:hashlib.sha256(Path(__file__).with_name(f).read_bytes()).hexdigest() for f in ['bench.py','kernels.py','kernel.h']},
    'note':'Packed affine W4/group64 experts; fixed real routing; same MLX process and wall timing. Compiled functions take changing arrays. All schedules match their phased tile-size control bitwise, including intermediates. Compilation, model loading, warmup and checks excluded. Eager and compiled native outputs may differ through pointwise fusion. This is a selected branch, not whole-model decode.'}
path=ROOT/args.output;path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(out,indent=2)+'\n')
for name,_,_ in choices:
    print(name,round(statistics.median(s['wall_us'] for s in samples if s['variant']==name),3),checks[name],flush=True)
