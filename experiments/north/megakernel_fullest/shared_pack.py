"""Share packed weight storage with the ordinary model for matched comparisons."""
import gc
import mlx.core as mx
from whole_pass.pack import pack_weights

def pack_shared(model):
    packed=pack_weights(model,start=1,count=48)
    for i,layer in enumerate(model.layers[1:]):
        pairs=[(layer.input_layernorm,'weight','norm_w'),(layer.mlp.gate,'weight','rw')]
        pairs += [(getattr(layer.self_attn,n+'_proj'),'weight',k) for n,k in [('q','qw'),('k','kw'),('v','vw'),('o','aw')]]
        for module,names in [('up_proj',('up','us','ub')),('gate_proj',('gate','gs','gb')),('down_proj',('down','ds','db'))]:
            pairs += [(getattr(layer.mlp.switch_mlp,module),field,name) for field,name in zip(('weight','scales','biases'),names)]
        for module,field,name in pairs:
            value=packed[name][i];old=getattr(module,field)
            assert value.shape==old.shape and value.dtype==old.dtype
            setattr(module,field,value)
    # Prefix is tiny relative to the MoE weights; rebind it too, retaining one layout.
    first=model.layers[0]
    groups={'prefix_dense':[(first.input_layernorm,'weight')]+[(getattr(first.self_attn,n+'_proj'),'weight') for n in ('q','k','v','o')],
            'prefix_q':[(getattr(first.mlp,n+'_proj'),'weight') for n in ('up','gate','down')],
            'prefix_meta':[(getattr(first.mlp,n+'_proj'),field) for n in ('up','gate','down') for field in ('scales','biases')]}
    for name,fields in groups.items():
        offset=0
        for module,field in fields:
            old=getattr(module,field);size=old.size
            setattr(module,field,packed[name][offset:offset+size].reshape(old.shape));offset+=size
        assert offset==packed[name].size
    del old,pairs,groups,fields
    gc.collect();mx.synchronize();mx.clear_cache()
    return packed
