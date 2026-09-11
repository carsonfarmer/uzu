"""Explicit, reversible single-token branch replacement for pinned North.

Prefill uses original layers. Mode 'native_compiled' controls for wrapper and
compiler effects; mode 'custom' tests the Metal branch. No global monkeypatch.
"""
import copy
import mlx.core as mx
import mlx.nn as nn
from kernels import inputs,run
from exact import run_exact
from persistent.branch import run_scheduled
from persistent.prefetch import run_staged
from persistent.fast import run_fast

class DecodeBranch(nn.Module):
    def __init__(self,original,mode,chunk=64,workers=128,rows=32):
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
            if mode=='exact':
                return run_exact(inputs(mlp,h,ax,ids,original.self_attn.o_proj.weight),
                                 scores,residual,workers=workers,rows=rows)
            if mode=='scheduled':
                return run_scheduled(inputs(mlp,h,ax,ids,original.self_attn.o_proj.weight),
                                     scores,residual,workers=workers)
            if mode=='fast':
                return run_fast(inputs(mlp,h,ax,ids,original.self_attn.o_proj.weight),
                                scores,residual,workers=workers)
            if mode in ('prefetch','staged'):
                return run_staged(inputs(mlp,h,ax,ids,original.self_attn.o_proj.weight),
                                  scores,residual,workers=workers,prefetch=mode=='prefetch')
            return run(inputs(mlp,h,ax,ids,original.self_attn.o_proj.weight),
                       scores,residual,chunk=chunk,workers=workers,interleave=True)
        assert mode in ('native_compiled','custom','exact','scheduled','prefetch','staged','fast')
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


def variants(model,chunk=64,workers=128,modes=('native_compiled','custom'),rows=32,schedule_workers=64):
    original=list(model.layers)
    return {'original':original,**{
        mode:[original[0]]+[DecodeBranch(layer,mode,chunk,schedule_workers if mode in ('scheduled','prefetch','staged','fast') else workers,rows) for layer in original[1:]]
        for mode in modes}}
