"""Opt-in North batch-one fusion. Multi-token prefill keeps the original layers."""
import copy
import mlx.core as mx
import mlx.nn as nn
from inputs import inputs
from fusion import run_exact
from prep import run
from mlx_vlm.models.base import scaled_dot_product_attention

class FusedLayer(nn.Module):
    def __init__(self,original):
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
        # Validate specialization before tracing.
        inputs(mlp,mx.zeros((1,1,2048),mx.bfloat16),
               mx.zeros((1,1,4096),mx.bfloat16),mx.zeros((1,1,8),mx.uint32),
               original.self_attn.o_proj.weight)
        def branch(h,ax,residual,ids,scores):
            return run_exact(inputs(mlp,h,ax,ids,original.self_attn.o_proj.weight),
                             scores,residual,workers=160,rows=16)
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


class PreparedLayer(FusedLayer):
    def __init__(self,original,rows=32,router_rows=4,normalize=False):
        super().__init__(original)
        weights=[getattr(original.self_attn,k+'_proj').weight for k in ('q','k','v')]+[original.mlp.gate.weight]
        def prepare(x):
            if normalize:
                return run(x,weights,rows,router_rows,original.input_layernorm.weight)
            h=original.input_layernorm(x)
            return run(h,weights,rows,router_rows)+[h]
        self.prepare=mx.compile(prepare)

    def __call__(self,x,mask=None,cache=None):
        if x.shape!=(1,1,2048):return self.original(x,mask,cache)
        prepared,h=self.prepare(x)
        att=self.original.self_attn
        q=prepared[...,:4096].reshape(1,1,32,128).transpose(0,2,1,3)
        k=prepared[...,4096:4608].reshape(1,1,4,128).transpose(0,2,1,3)
        v=prepared[...,4608:5120].reshape(1,1,4,128).transpose(0,2,1,3)
        if att.use_sliding_window or att.force_rope:
            offset=0 if cache is None else cache.offset
            q=att.rope(q,offset=offset);k=att.rope(k,offset=offset)
        if cache is not None:k,v=cache.update_and_fetch(k,v)
        ax=scaled_dot_product_attention(q,k,v,cache=cache,scale=att.scale,mask=mask)
        ax=ax.transpose(0,2,1,3).reshape(1,1,4096)
        gates=self.mlp.gate_act(prepared[...,5120:].astype(mx.float32))
        ids=mx.stop_gradient(mx.argpartition(-gates,kth=7,axis=-1)[...,:8])
        scores=mx.take_along_axis(gates,ids,axis=-1)
        return self.branch(h,ax,x,ids,scores)


def variants(model):
    """Keep the first dense layer native; specialize the 48 MoE layers."""
    original=list(model.layers)
    assert len(original)==49
    return {
        'original': original,
        'fused': [original[0]]+[FusedLayer(layer) for layer in original[1:]],
        'prepared': [original[0]]+[PreparedLayer(layer,64,8,True) for layer in original[1:]],
    }
