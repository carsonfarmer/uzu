"""Same phase-queue primitives split at layer boundaries; absolute RoPE pattern preserved."""
_KERNEL=None

def run(weights,x,key_cache,value_cache,position,offset=0,workers=32,do_head=False,do_prefix=False):
    global _KERNEL
    import mlx.core as mx
    from whole_pass import kernel as old
    from whole_pass.pack import WEIGHT_NAMES,HEAD_NAMES,PREFIX_NAMES
    layers=weights['norm_w'].shape[0];capacity=key_cache.shape[2]
    assert key_cache.shape==value_cache.shape and key_cache.shape[0]==layers+int(do_prefix)
    if _KERNEL is None:
        source=old.source.replace('((layer+1)%4)','((layer+1+LAYER_OFFSET)%4)')
        assert source!=old.source
        _KERNEL=mx.fast.metal_kernel(name='north_cut_phase_queue',input_names=['x_in','key_cache','value_cache','params',*WEIGHT_NAMES,*HEAD_NAMES,*PREFIX_NAMES,'sigmoid_table'],output_names=['key_out','value_out','workspace','logits'],header=old.header,source=source)
    outputs=_KERNEL(inputs=[x,key_cache,value_cache,mx.array([position,capacity],mx.uint32),*[weights[n] for n in WEIGHT_NAMES],*[weights[n] for n in HEAD_NAMES],*[weights[n] for n in PREFIX_NAMES],old.sigmoid_lut()],
        template=[('NLAYERS',layers),('WORKERS',workers),('FINE',False),('SAFE_QUEUE',True),('PREFETCH_STAGES',0),('PREP_ROWS',64),('OPROJ_ROWS',64),('ROUTER_ROWS',8),('DO_HEAD',do_head),('LM_ROWS',512),('DO_PREFIX',do_prefix),('LAYER_OFFSET',offset)],
        grid=(workers*256,1,1),threadgroup=(256,1,1),output_shapes=[key_cache.shape,value_cache.shape,(old.SIZE,),(1,1,262144)],output_dtypes=[mx.bfloat16,mx.bfloat16,mx.uint32,mx.bfloat16],init_value=0)
    raw=outputs[2].view(mx.bfloat16);out=raw[old.X*2:old.NORM*2].reshape(1,1,2048)
    return out,outputs[0],outputs[1],outputs[2][:3],outputs[3]

class CutState:
    def __init__(self,weights,caches,position,cuts,capacity):
        from whole_pass.pack import WEIGHT_NAMES,pack_caches
        assert cuts in (1,2,4,8) and 48%cuts==0
        self.position=position;self.capacity=capacity;self.parts=[]
        count=48//cuts
        for block in range(cuts):
            start=block*count;prefix=block==0;head=block==cuts-1
            subset={name:(value[start:start+count] if name in WEIGHT_NAMES else value) for name,value in weights.items()}
            keys,values=pack_caches(caches,start=0 if prefix else start+1,count=count+int(prefix),capacity=capacity)
            self.parts.append(dict(weights=subset,keys=keys,values=values,offset=start,prefix=prefix,head=head))
    def advance(self,h,workers=32,retain=False):
        import mlx.core as mx
        if self.position>=self.capacity:
            extra=self.capacity;self.capacity*=2
            for p in self.parts:
                shape=(p['keys'].shape[0],4,extra,128)
                p['keys']=mx.concatenate([p['keys'],mx.zeros(shape,mx.bfloat16)],axis=2)
                p['values']=mx.concatenate([p['values'],mx.zeros(shape,mx.bfloat16)],axis=2)
        states=[]
        for p in self.parts:
            h,p['keys'],p['values'],state,logits=run(p['weights'],h,p['keys'],p['values'],self.position,offset=p['offset'],workers=workers,do_head=p['head'],do_prefix=p['prefix'])
            if retain:states.append(state)
        self.position+=1
        return logits,states
