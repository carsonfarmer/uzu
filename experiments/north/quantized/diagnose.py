"""Attribute discrepancies to activation vs projection/reduction arithmetic."""
import sys,json
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from reference import ROOT,load
from kernels import HEADER,NAMES,inputs,run
m,_=load();layer=m.layers[1];mlp=layer.mlp
cap=ROOT/'work/north-layer';meta=json.loads((cap/'metadata.json').read_text())
def bf(name,shape):return mx.array(np.fromfile(cap/(name+'.bin'),dtype=np.uint16)).view(mx.bfloat16).reshape(shape)
x=bf('x',(1,1,2048));ax=bf('ax',(1,1,4096));r=bf('residual',(1,1,2048))
ids=mx.array(meta['experts'],dtype=mx.uint32).reshape(1,1,8)
scores=mx.array(np.fromfile(cap/'scores.bin',dtype=np.float32))
data=inputs(mlp,x,ax,ids,layer.self_attn.o_proj.weight)
xx=mx.expand_dims(x,(-2,-3))
u=mlp.switch_mlp.up_proj(xx,ids);g=mlp.switch_mlp.gate_proj(xx,ids)
ref=mlp.switch_mlp.activation(u,g).reshape(8,768)
def delta(a,b):
    d=a.astype(mx.float32)-b.astype(mx.float32)
    return dict(max_abs=float(mx.max(mx.abs(d)).item()),relative_l2=float((mx.linalg.norm(d)/mx.linalg.norm(b.astype(mx.float32))).item()),unequal=int(mx.sum(a!=b).item()))
forms={'native_eager':(nn.silu(g)*u).reshape(8,768),
 'sigmoid_bf16':(g*mx.sigmoid(g)).astype(mx.bfloat16)*u,
 'float_silu_bf16':(g.astype(mx.float32)/(1+mx.exp(-g.astype(mx.float32)))).astype(mx.bfloat16)*u,
 'fused_float':(g.astype(mx.float32)*mx.sigmoid(g.astype(mx.float32))*u.astype(mx.float32)).astype(mx.bfloat16)}
q=run(data,scores,r,return_parts=True)
forms['custom_hidden']=q[1]
out={k:delta(v.reshape(8,768),ref) for k,v in forms.items()}
for label,refproj in [('u16',u),('g16',g)]:
    kernel=mx.fast.metal_kernel(name='north_probe_'+label,input_names=NAMES,
        output_names=['hidden'],header='#define NCHUNK 32\n'+HEADER.replace(
        'bfloat h=bfloat(float(u16)*float(a16));',f'bfloat h={label};'),
        source='''threadgroup float tile[32];
north_hidden(threadgroup_position_in_grid.x,simdgroup_index_in_threadgroup,
thread_index_in_simdgroup,x,ids,up,us,ub,gate,gs,gb,hidden,tile);''')
    probe=kernel(inputs=data,grid=(192*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(8,768)],output_dtypes=[mx.bfloat16])[0]
    out[label]=delta(probe,refproj.reshape(8,768))
for label,expression in [('float','float z=1.f/(1.f+exp(abs(float(v)))); bfloat sig=bfloat(float(v)<0?z:1.f-z);'),
                          ('bfloat','auto z=1/(1+exp(abs(v))); bfloat sig=(v<0)?z:1-z;')]:
    kernel=mx.fast.metal_kernel(name='north_probe_act_'+label,input_names=['up','gate'],
        output_names=['h'],source='uint i=thread_position_in_grid.x;bfloat v=gate[i];'+expression+
        'bfloat a=bfloat(float(v)*float(sig));h[i]=bfloat(float(a)*float(up[i]));')
    probe=kernel(inputs=[u,g],grid=(6144,1,1),threadgroup=(256,1,1),output_shapes=[(8,768)],output_dtypes=[mx.bfloat16])[0]
    out['act_'+label]=delta(probe,ref)
print(json.dumps(out,indent=2))
(ROOT/'experiments/north/quantized/results/activation-diagnostic.json').write_text(json.dumps(out,indent=2)+'\n')
