"""Opt-in packed affine-W4 North branch; fixed batch-one geometry."""
from functools import lru_cache
from pathlib import Path
import mlx.core as mx

HEADER = Path(__file__).with_name('kernel.h').read_text()
NAMES = ['x','ax','ids','up','us','ub','gate','gs','gb','down','ds','db','aw']
DTYPES = [mx.bfloat16, mx.float32, mx.bfloat16]

@lru_cache(None)
def kernels(chunk):
    assert chunk in (32,64,128)
    nc = 768 // chunk
    jobs = 8 * nc
    total = jobs + 64
    header = f'#define NCHUNK {chunk}\n' + HEADER
    threads = f'''uint tid=thread_position_in_threadgroup.x;
uint lane=thread_index_in_simdgroup;
uint sg=simdgroup_index_in_threadgroup;
uint group=threadgroup_position_in_grid.x;
threadgroup float tile[{chunk}];
'''
    body = f'''
if(job<{jobs}){{
 north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 north_down(job,tid,ids,down,ds,db,tile,partial);
}}else north_attention(job-{jobs},sg,lane,ax,aw,attention);
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
'''
    # Round-robin interleave equal numbers from the branches, then the remainder.
    paired = min(jobs,64)
    mapping = f'((task<{paired*2}) ? ((task%2)?{jobs}+task/2:task/2) : '
    mapping += (f'task-{paired}' if jobs>=64 else f'task') + ')'
    mixed = mx.fast.metal_kernel(name=f'north_q_mixed_{chunk}',input_names=NAMES,
        output_names=['hidden','partial','attention'],header=header,
        source=threads+f'''for(uint task=group;task<{total};task+=WORKERS) {{
uint job=INTERLEAVE ? {mapping} : task;
'''+body+'}')
    up = mx.fast.metal_kernel(name=f'north_q_up_{chunk}',input_names=NAMES,
        output_names=['hidden'],header=header,
        source=threads+'north_hidden(group,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);')
    down = mx.fast.metal_kernel(name=f'north_q_down_{chunk}',input_names=NAMES+['hidden'],
        output_names=['partial'],header=header,source=threads+f'''
if(tid<{chunk})tile[tid]=float(hidden[(group/{nc})*768+(group%{nc})*{chunk}+tid]);
threadgroup_barrier(mem_flags::mem_threadgroup);
north_down(group,tid,ids,down,ds,db,tile,partial);
''')
    attn = mx.fast.metal_kernel(name=f'north_q_attn_{chunk}',input_names=['ax','aw'],
        output_names=['attention'],header=header,
        source=threads+'north_attention(group,sg,lane,ax,aw,attention);')
    join = mx.fast.metal_kernel(name=f'north_q_join_{chunk}',
        input_names=['partial','attention','scores','residual'],output_names=['out'],source=f'''
uint r=thread_position_in_grid.x;
float sum=0;
for(uint e=0;e<8;e++) {{
 float value=0;
 for(uint c=0;c<{nc};c++)value+=partial[(e*{nc}+c)*2048+r];
 sum+=float(bfloat(value))*scores[e];
}}
bfloat ff=bfloat(sum);
out[r]=bfloat(float(bfloat(float(attention[r])+float(ff)))+float(residual[r]));
''')
    return mixed,up,down,attn,join


def inputs(mlp,x,ax,ids,aw):
    assert x.shape==(1,1,2048) and ax.shape==(1,1,4096) and ids.size==8
    weights=[]
    for name in ['up_proj','gate_proj','down_proj']:
        m=getattr(mlp.switch_mlp,name)
        assert (m.bits,m.group_size,m.mode)==(4,64,'affine')
        assert m.scales.dtype==mx.bfloat16 and m.biases.dtype==mx.bfloat16
        weights += [m.weight,m.scales,m.biases]
    assert x.dtype==mx.bfloat16 and ax.dtype==mx.bfloat16
    assert aw.shape==(2048,4096) and aw.dtype==mx.bfloat16
    return [x,ax,ids.astype(mx.uint32),*weights,aw]


def run(data,scores,residual,workers=128,interleave=False,phased=False,
        return_parts=False,chunk=32):
    mixed,up,down,attn,join=kernels(chunk)
    nc=768//chunk; jobs=8*nc
    shapes=[(8,768),(8,nc,2048),(1,1,2048)]
    if phased:
        h=up(inputs=data,grid=(jobs*256,1,1),threadgroup=(256,1,1),
            output_shapes=[shapes[0]],output_dtypes=[DTYPES[0]])[0]
        p=down(inputs=data+[h],grid=(jobs*256,1,1),threadgroup=(256,1,1),
            output_shapes=[shapes[1]],output_dtypes=[DTYPES[1]])[0]
        a=attn(inputs=[data[1],data[-1]],grid=(64*256,1,1),threadgroup=(256,1,1),
            output_shapes=[shapes[2]],output_dtypes=[DTYPES[2]])[0]
    else:
        assert 0<workers<=jobs+64
        h,p,a=mixed(inputs=data,template=[('WORKERS',workers),('INTERLEAVE',interleave)],
            grid=(workers*256,1,1),threadgroup=(256,1,1),
            output_shapes=shapes,output_dtypes=DTYPES)
    out=join(inputs=[p,a,scores,residual],grid=(2048,1,1),threadgroup=(256,1,1),
        output_shapes=[(1,1,2048)],output_dtypes=[mx.bfloat16])[0]
    return (out,h,p,a) if return_parts else out
