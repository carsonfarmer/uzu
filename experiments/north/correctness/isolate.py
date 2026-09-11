"""Compare every custom branch component on identical native-model inputs.

Hooks only observe the native forward pass; candidate results never feed the
reference's next layer. This identifies local errors without accumulated drift.
"""
import argparse,hashlib,json,sys
from pathlib import Path
import mlx.core as mx
from transformers import AutoTokenizer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'quantized'))
from reference import ROOT,load
from kernels import inputs,run,HEADER,NAMES
from exact import run_exact

p=argparse.ArgumentParser()
p.add_argument('--steps',type=int,default=4)
p.add_argument('--mode',choices=['split','exact'],default='split')
p.add_argument('--output',default='experiments/north/correctness/results/isolate-before.jsonl')
a=p.parse_args()
model,config=load();base=ROOT/'work/models/North-Mini-Code-1.0-4bit'
tok=AutoTokenizer.from_pretrained(str(base),local_files_only=True)
tok.chat_template=(base/'chat_template.jinja').read_text()
layer_ids={id(layer):i for i,layer in enumerate(model.layers) if i}
mlp_ids={id(layer.mlp):i for i,layer in enumerate(model.layers) if i}
attn_ids={id(layer.self_attn.o_proj):i for i,layer in enumerate(model.layers) if i}
layer_cls=type(model.layers[1]);mlp_cls=type(model.layers[1].mlp);linear_cls=type(model.layers[1].self_attn.o_proj)
old_layer,old_mlp,old_linear=layer_cls.__call__,mlp_cls.__call__,linear_cls.__call__
captured={}
def layer_call(self,x,*args,**kwargs):
    y=old_layer(self,x,*args,**kwargs)
    if id(self) in layer_ids:captured.setdefault(layer_ids[id(self)],{}).update(residual=x,output=y)
    return y
def mlp_call(self,x,*args,**kwargs):
    y=old_mlp(self,x,*args,**kwargs)
    if id(self) in mlp_ids:captured.setdefault(mlp_ids[id(self)],{}).update(x=x,moe=y)
    return y
def linear_call(self,x,*args,**kwargs):
    y=old_linear(self,x,*args,**kwargs)
    if id(self) in attn_ids:captured.setdefault(attn_ids[id(self)],{}).update(ax=x,attention=y)
    return y
reduce_down=mx.fast.metal_kernel(name='audit_partial_down',input_names=['p'],output_names=['d'],source='''
uint r=thread_position_in_grid.x;
float v=0;for(uint c=0;c<12;c++)v+=p[(r/2048*12+c)*2048+r%2048];
d[r]=bfloat(v);
''')
def delta(x,y):
    bitwise=int(mx.sum(x.reshape(-1).view(mx.uint8)!=y.reshape(-1).view(mx.uint8)).item())
    x=x.reshape(-1).astype(mx.float32);y=y.reshape(-1).astype(mx.float32);d=x-y
    return dict(unequal=int(mx.sum(x!=y).item()),unequal_bytes=bitwise,max_abs=float(mx.max(mx.abs(d)).item()),
        relative_l2=float((mx.linalg.norm(d)/mx.maximum(mx.linalg.norm(y),1e-20)).item()))
# Independent probe outputs identify up/gate errors separately from activation.
probes={}
for name,expr in [('up','u16'),('gate','g16')]:
    assert HEADER.count('bfloat h=bfloat(float(u16)*float(a16));')==1
    probes[name]=mx.fast.metal_kernel(name='audit_'+name,input_names=NAMES,output_names=['hidden'],
        header='#define NCHUNK 64\n'+HEADER.replace('bfloat h=bfloat(float(u16)*float(a16));',f'bfloat h={expr};'),
        source='''threadgroup float tile[64];
north_hidden(threadgroup_position_in_grid.x,simdgroup_index_in_threadgroup,
thread_index_in_simdgroup,x,ids,up,us,ub,gate,gs,gb,hidden,tile);''')

output=ROOT/a.output;output.parent.mkdir(parents=True,exist_ok=True)
with output.open('w') as f:
    def save(r):f.write(json.dumps(r)+'\n');f.flush()
    save(dict(kind='provenance',mlx=mx.__version__,steps=a.steps,mode=a.mode,
        kernel_sha256=hashlib.sha256(HEADER.encode()).hexdigest(),
        exact_sha256=hashlib.sha256((ROOT/'experiments/north/quantized/exact.py').read_bytes()).hexdigest()))
    prompts=['Return only Python code for merging two sorted lists. Include assertions for empty inputs and duplicate values.',
             'Return only Rust code for finding the first duplicate in a u32 slice, with tests for empty input and duplicates.']
    for prompt_id,prompt in enumerate(prompts):
        ids=tok.apply_chat_template([dict(role='user',content=prompt)],tokenize=True,return_dict=False,add_generation_prompt=True,reasoning=False,skip_thinking=True)
        cache=model.make_cache();logits=model(mx.array([ids]),cache=cache).logits
        token=int(mx.argmax(logits[0,-1]).item())
        for step in range(a.steps):
            captured.clear()
            layer_cls.__call__,mlp_cls.__call__,linear_cls.__call__=layer_call,mlp_call,linear_call
            try:
                logits=model(mx.array([[token]]),cache=cache).logits;mx.eval(logits)
                token=int(mx.argmax(logits[0,-1]).item())
            finally:
                layer_cls.__call__,mlp_cls.__call__,linear_cls.__call__=old_layer,old_mlp,old_linear
            assert len(captured)==48
            for index,layer in enumerate(model.layers[1:],1):
                c=captured[index];x,ax,res=c['x'],c['ax'],c['residual'];mlp=layer.mlp
                gates=mlp.gate_act(mlp.gate(x).astype(mx.float32))
                ids=mx.argpartition(-gates,kth=7,axis=-1)[...,:8]
                scores=mx.take_along_axis(gates,ids,axis=-1)
                xx=mx.expand_dims(x,(-2,-3))
                u=mlp.switch_mlp.up_proj(xx,ids);g=mlp.switch_mlp.gate_proj(xx,ids)
                h=mlp.switch_mlp.activation(u,g)
                d=mlp.switch_mlp.down_proj(h,ids).reshape(8,2048)
                native_moe=(d*scores.reshape(8,1)).sum(0).astype(mx.bfloat16).reshape(1,1,2048)
                data=inputs(mlp,x,ax,ids,layer.self_attn.o_proj.weight)
                if a.mode=='exact':
                    out,qh,qd,qa=run_exact(data,scores,res,return_parts=True)
                else:
                    out,qh,qp,qa=run(data,scores,res,chunk=64,workers=128,interleave=True,return_parts=True)
                    qd=reduce_down(inputs=[qp],grid=(8*2048,1,1),threadgroup=(256,1,1),output_shapes=[(8,2048)],output_dtypes=[mx.bfloat16])[0]
                # Native down-projection supplied with custom hidden values.
                dq=mlp.switch_mlp.down_proj(qh.reshape(h.shape),ids).reshape(8,2048)
                probe={k:fn(inputs=data,grid=(96*256,1,1),threadgroup=(256,1,1),output_shapes=[(8,768)],output_dtypes=[mx.bfloat16])[0] for k,fn in probes.items()}
                metrics={k:delta(q,v) for k,q,v in [('up',probe['up'],u),('gate',probe['gate'],g),('hidden',qh,h),
                    ('down_same_hidden',qd,dq),('down',qd,d),('attention',qa,c['attention']),
                    ('native_reconstruction',native_moe,c['moe']),('output',out,c['output'])]}
                save(dict(kind='layer',prompt=prompt_id,step=step,layer=index,metrics=metrics))
                if step==0 and prompt_id==0:
                    print(index,{k:v['unequal'] for k,v in metrics.items()},flush=True)
            print('completed',prompt_id,step,flush=True)
