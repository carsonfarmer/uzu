"""Experimental preparation fusion around the unchanged exact branch control."""
import mlx.core as mx
from patch import DecodeBranch
from prep import run
from mlx_vlm.models.base import scaled_dot_product_attention


class PreparedBranch(DecodeBranch):
    def __init__(self,original,rows=32,router_rows=4,normalize=False):
        super().__init__(original,'exact',workers=160,rows=16)
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
