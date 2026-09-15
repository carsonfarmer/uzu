"""Validate the pinned affine-W4/BF16 geometry before compiling."""
from pathlib import Path
import mlx.core as mx

HEADER = Path(__file__).with_name("kernel.h").read_text()
NAMES = ["x", "ax", "ids", "up", "us", "ub", "gate", "gs", "gb", "down", "ds", "db", "aw"]

def inputs(mlp,x,ax,ids,aw):
    assert x.shape==(1,1,2048) and ax.shape==(1,1,4096) and ids.size==8
    weights=[]
    for name in ['up_proj','gate_proj','down_proj']:
        m=getattr(mlp.switch_mlp,name)
        assert (m.bits,m.group_size,m.mode)==(4,64,'affine')
        assert m.scales.dtype==mx.bfloat16 and m.biases.dtype==mx.bfloat16
        rows,cols=(2048,768) if name=='down_proj' else (768,2048)
        assert m.weight.dtype==mx.uint32 and m.weight.shape==(128,rows,cols//8)
        assert m.scales.shape==m.biases.shape==(128,rows,cols//64)
        weights += [m.weight,m.scales,m.biases]
    assert x.dtype==mx.bfloat16 and ax.dtype==mx.bfloat16
    assert aw.shape==(2048,4096) and aw.dtype==mx.bfloat16
    return [x,ax,ids.astype(mx.uint32),*weights,aw]

