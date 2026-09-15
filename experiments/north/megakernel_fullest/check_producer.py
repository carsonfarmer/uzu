"""Numerical/progress gate for dedicated loader and one/two-slot ring."""
import argparse,hashlib,json,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE.parent))
import mlx.core as mx
from reference import load,ROOT
from producer_prep import run
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--layers',type=int,nargs='+',default=[7]);a=p.parse_args()
model,_=load();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
with out.open('w') as f:
 def save(r):f.write(json.dumps(r)+'\n');f.flush()
 save(dict(kind='provenance',args=vars(a),mlx=mx.__version__,device=mx.device_info(),sources={str(q.relative_to(ROOT)):hashlib.sha256(q.read_bytes()).hexdigest() for q in [Path(__file__).resolve(),HERE/'producer_prep.py',HERE.parent/'additional/prep.py']},scope='Actual model matrices, synthetic activations, complete projection bytes; separate producer SIMD and exact consumers, no model speed claim'))
 for index in a.layers:
  layer=model.layers[index];mx.random.seed(1011+index);x=mx.random.normal((1,1,2048)).astype(mx.bfloat16);h=layer.input_layernorm(x)
  for name,module in [('q',layer.self_attn.q_proj),('k',layer.self_attn.k_proj),('v',layer.self_attn.v_proj),('router',layer.mlp.gate)]:
   ref=module(h);mx.eval(ref,h)
   for depth in (1,2):
    got,norm,errors=run(x,module.weight,layer.input_layernorm.weight,router=name=='router',depth=depth);mx.eval(got,norm,errors)
    row=dict(kind='correctness',layer=index,projection=name,depth=depth,unequal_bytes=int(mx.sum(got.view(mx.uint8)!=ref.view(mx.uint8)).item()),norm_unequal_bytes=int(mx.sum(norm.view(mx.uint8)!=h.view(mx.uint8)).item()),errors=int(mx.sum(errors).item()))
    save(row);print(row,flush=True);assert row['unequal_bytes']==row['norm_unequal_bytes']==row['errors']==0,row
