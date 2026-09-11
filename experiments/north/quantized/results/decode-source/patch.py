"""Explicit, reversible single-token branch replacement for pinned North.

Prefill uses original layers. Mode 'native_compiled' controls for wrapper and
compiler effects; mode 'custom' tests the Metal branch. No global monkeypatch.
"""
import copy
import mlx.core as mx
import mlx.nn as nn
from kernels import inputs,run

class DecodeBranch(nn.Module):
    def __init__(self,original,mode,chunk=64,workers=128):
        super().__init__()
        self.original=original
        # The model consults this attribute when choosing its attention mask.
        self.self_attn=original.self_attn
        unprojected=copy.copy(original.self_attn)
        unprojected.o_proj=nn.Identity()
        assert unprojected.o_proj is not original.self_attn.o_proj
        self.unprojected=unprojected
        mlp=original.mlp
        assert mlp.shared_experts is None
        assert mlp.top_k==8 and not mlp.norm_topk_prob
        assert mlp.switch_mlp.up_proj.weight.shape[0]==128
        self.mlp=mlp
        self.mode=mode
        # Validate specialization before tracing.
        inputs(mlp,mx.zeros((1,1,2048),mx.bfloat16),
               mx.zeros((1,1,4096),mx.bfloat16),mx.zeros((1,1,8),mx.uint32),
               original.self_attn.o_proj.weight)
        def branch(h,ax,residual,ids,scores):
            if mode=='native_compiled':
                routed=mlp.switch_mlp(h,ids)
                moe=(routed*scores[...,None]).sum(-2).astype(routed.dtype)
                return original.self_attn.o_proj(ax)+moe+residual
            return run(inputs(mlp,h,ax,ids,original.self_attn.o_proj.weight),
                       scores,residual,chunk=chunk,workers=workers,interleave=True)
        assert mode in ('native_compiled','custom')
        self.branch=mx.compile(branch)

    def __call__(self,x,mask=None,cache=None):
        if x.shape!=(1,1,2048):
            return self.original(x,mask,cache)
        h=self.original.input_layernorm(x)
        ax=self.unprojected(h,mask,cache)
        gates=self.mlp.gate_act(self.mlp.gate(h).astype(mx.float32))
        ids=mx.stop_gradient(mx.argpartition(-gates,kth=7,axis=-1)[...,:8])
        scores=mx.take_along_axis(gates,ids,axis=-1)
        return self.branch(h,ax,x,ids,scores)


def variants(model,chunk=64,workers=128):
    original=list(model.layers)
    return {'original':original,**{
        mode:[original[0]]+[DecodeBranch(layer,mode,chunk,workers) for layer in original[1:]]
        for mode in ('native_compiled','custom')}}
