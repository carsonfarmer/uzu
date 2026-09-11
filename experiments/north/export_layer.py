"""Capture a real single-token North layer; export selected expert and O-proj weights."""
import argparse,json,time
from pathlib import Path
import mlx.core as mx
import numpy as np
from transformers import AutoTokenizer
from reference import ROOT,load
path=ROOT/'work/models/North-Mini-Code-1.0-4bit'
parser=argparse.ArgumentParser()
parser.add_argument('--layer',type=int,default=1)
parser.add_argument('--output',default='work/north-layer')
parser.add_argument('--prompt',default='Return only Python code for merging two sorted lists with an assertion.')
args=parser.parse_args()
model,config=load(path)
tok=AutoTokenizer.from_pretrained(str(path),local_files_only=True)
tok.chat_template=(path/'chat_template.jinja').read_text()
prompt=args.prompt
ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
cache=model.make_cache()
logits=model(mx.array([ids]),cache=cache).logits;mx.eval(logits)
token=int(mx.argmax(logits[0,-1]).item())
target=model.layers[args.layer];captured={}
layer_cls=type(target);old_layer=layer_cls.__call__
linear_cls=type(target.self_attn.o_proj);old_linear=linear_cls.__call__
mlp_cls=type(target.mlp);old_mlp=mlp_cls.__call__
def layer_call(self,x,*args,**kwargs):
 y=old_layer(self,x,*args,**kwargs)
 if self is target:captured.update(residual=x,output=y)
 return y
def linear_call(self,x,*args,**kwargs):
 y=old_linear(self,x,*args,**kwargs)
 if self is target.self_attn.o_proj:captured.update(ax=x,attention=y)
 return y
def mlp_call(self,x,*args,**kwargs):
 y=old_mlp(self,x,*args,**kwargs)
 if self is target.mlp:captured.update(x=x,moe=y)
 return y
layer_cls.__call__=layer_call;linear_cls.__call__=linear_call;mlp_cls.__call__=mlp_call
try:
 logits=model(mx.array([[token]]),cache=cache).logits;mx.eval(logits,*captured.values())
finally:
 layer_cls.__call__=old_layer;linear_cls.__call__=old_linear;mlp_cls.__call__=old_mlp
x=captured['x']
gates=mx.sigmoid(target.mlp.gate(x).astype(mx.float32))
indices=mx.argpartition(-gates,kth=7,axis=-1)[...,:8]
scores=mx.take_along_axis(gates,indices,-1).flatten()
selected=indices.flatten()
arrays=dict(captured,scores=scores,aw=target.self_attn.o_proj.weight)
for short,name in [('up','up_proj'),('gate','gate_proj'),('down','down_proj')]:
 m=getattr(target.mlp.switch_mlp,name)
 arrays[short]=mx.dequantize(m.weight[selected],m.scales[selected],m.biases[selected],
  group_size=m.group_size,bits=m.bits,mode=m.mode)
mx.eval(*arrays.values())
# Independent materialized-weight control; preserve the prototype's BF16 boundaries.
u=(x.reshape(1,1,2048) @ arrays['up'].swapaxes(-1,-2)).squeeze(1)
g=(x.reshape(1,1,2048) @ arrays['gate'].swapaxes(-1,-2)).squeeze(1)
act=(g.astype(mx.float32)*mx.sigmoid(g.astype(mx.float32))).astype(mx.bfloat16)
mx.eval(u,g,act)
h=(u.astype(mx.float32)*act.astype(mx.float32)).astype(mx.bfloat16)
value=(h[:,None,:] @ arrays['down'].swapaxes(-1,-2)).squeeze(1)
moe=(value.astype(mx.float32)*scores[:,None]).sum(0).astype(mx.bfloat16)
attn=captured['ax'] @ arrays['aw'].T
arrays['materialized_output']=(attn+moe)+captured['residual']
mx.eval(arrays['materialized_output'])
out=ROOT/args.output;out.mkdir(exist_ok=True)
metadata={'layer':args.layer,'prompt':prompt,'prompt_tokens':len(ids),'decode_token':token,'experts':selected.tolist(),'arrays':{},
 'note':'Captured single decode token. Selected affine 4bit expert weights dequantized/materialized to BF16 for initial scheduler experiment; original runtime continues to use quantized matmuls.'}
for name,a in arrays.items():
 if name=='scores':data=np.array(a.astype(mx.float32))
 else:data=np.array(a.astype(mx.bfloat16).view(mx.uint16))
 data.tofile(out/(name+'.bin'))
 metadata['arrays'][name]={'shape':list(a.shape),'source_dtype':str(a.dtype),'storage':'float32' if name=='scores' else 'bf16','bytes':data.nbytes}
(out/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
timings=[]
for repetition in range(-5,15):
 start=time.perf_counter()
 routed=target.mlp.switch_mlp(x,indices)
 reference_moe=(routed*scores.reshape(1,1,8,1)).sum(axis=-2).astype(routed.dtype)
 reference=target.self_attn.o_proj(captured['ax'])+reference_moe+captured['residual']
 mx.eval(reference)
 elapsed=time.perf_counter()-start
 if repetition>=0:timings.append(elapsed)
(out/'mlx-selected-branch-timing.json').write_text(json.dumps({'wall_seconds':timings,'max_abs_vs_captured':float(mx.max(mx.abs(reference-captured['output'])).item()),'note':'MLX quantized selected experts plus BF16 attention output projection and residual. Fixed routing/scores; lazy graph construction and evaluation included. Different weight representation/reduction from standalone Metal; not an isolated scheduler comparison.'},indent=2)+'\n')
print(json.dumps(metadata,indent=2),flush=True)
