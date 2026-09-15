"""Model adapter for ready tail experiments; normal attention/cache and exact last branch."""
import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.base import scaled_dot_product_attention
from kernels import inputs
from exact import run_exact
from prep import run as prepare
from ready_tail import run as tail

class Carrier:
    def __init__(self):self.value=None
    def take(self,index):
        assert self.value is not None and self.value[0]==index
        values=self.value[1];self.value=None;return values
    def put(self,index,values):
        assert self.value is None
        self.value=(index,values)

class CrossLayer(nn.Module):
    def __init__(self,original,index,next_layer,carrier,workers,prefix,prefetch,storage):
        super().__init__();self.original=original;self.index=index;self.next_layer=next_layer;self.carrier=carrier
        self.self_attn=original.self_attn;self.mlp=original.mlp
        if next_layer is None:
            def branch(h,ax,ids,scores,x):return (run_exact(inputs(self.mlp,h,ax,ids,self.self_attn.o_proj.weight),scores,x,workers=160,rows=16),)
        else:
            def branch(h,ax,ids,scores,x):return tail(inputs(self.mlp,h,ax,ids,self.self_attn.o_proj.weight),scores,x,next_layer,workers=workers,prefix=prefix,prefetch=prefetch,storage=storage)
        self.branch=mx.compile(branch)
        weights=[getattr(self.self_attn,k+'_proj').weight for k in ('q','k','v')]+[self.mlp.gate.weight]
        self.first_prep=mx.compile(lambda x:prepare(x,weights,64,8,original.input_layernorm.weight))
    def __call__(self,x,mask=None,cache=None):
        if x.shape!=(1,1,2048):return self.original(x,mask,cache)
        if self.index==1:
            p,h=self.first_prep(x);q,k,v,router=p[...,:4096],p[...,4096:4608],p[...,4608:5120],p[...,5120:]
        else:h,q,k,v,router=self.carrier.take(self.index)
        att=self.self_attn
        q=q.reshape(1,1,32,128).transpose(0,2,1,3);k=k.reshape(1,1,4,128).transpose(0,2,1,3);v=v.reshape(1,1,4,128).transpose(0,2,1,3)
        if att.use_sliding_window or att.force_rope:
            offset=0 if cache is None else cache.offset;q=att.rope(q,offset=offset);k=att.rope(k,offset=offset)
        if cache is not None:k,v=cache.update_and_fetch(k,v)
        ax=scaled_dot_product_attention(q,k,v,cache=cache,scale=att.scale,mask=mask).transpose(0,2,1,3).reshape(1,1,4096)
        gates=self.mlp.gate_act(router.astype(mx.float32));ids=mx.stop_gradient(mx.argpartition(-gates,kth=7,axis=-1)[...,:8]);scores=mx.take_along_axis(gates,ids,axis=-1)
        result=self.branch(h,ax,ids,scores,x)
        if self.next_layer is not None:self.carrier.put(self.index+1,result[1:])
        return result[0]

def path(original,workers=32,prefix=128,prefetch=False,storage="threadgroup"):
    carrier=Carrier()
    return [original[0]]+[CrossLayer(layer,i,original[i+1] if i+1<len(original) else None,carrier,workers,prefix,prefetch,storage) for i,layer in enumerate(original[1:],1)]
