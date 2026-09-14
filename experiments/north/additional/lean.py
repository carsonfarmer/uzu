"""Same exact arithmetic and launch geometry, with unused front bindings removed."""
import mlx.core as mx
from kernels import HEADER,NAMES,inputs
from exact import down
from patch import DecodeBranch

KEEP=[0,1,2,3,4,5,6,7,8,12]
front=mx.fast.metal_kernel(name='north_exact_lean_front',input_names=[NAMES[i] for i in KEEP],
    output_names=['hidden','attention'],header='#define NCHUNK 64\n'+HEADER,
    source='''
uint group=threadgroup_position_in_grid.x;
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
threadgroup float tile[64];
for(uint task=group;task<160;task+=160) {
 uint job=(task<128)?((task%2)?96+task/2:task/2):task-64;
 if(job<96)north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
 else north_attention(job-96,sg,lane,ax,aw,attention);
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
''')

def run_lean(data,scores,residual):
    h,a=front(inputs=[data[i] for i in KEEP],grid=(160*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(8,768),(1,1,2048)],output_dtypes=[mx.bfloat16,mx.bfloat16])
    return down(inputs=[h,data[2],*data[9:12],scores,a,residual],template=[('ROWS',16)],
        grid=(16384,1,1),threadgroup=(128,1,1),output_shapes=[(1,1,2048),(8,2048)],
        output_dtypes=[mx.bfloat16,mx.bfloat16])[0]

class LeanBranch(DecodeBranch):
    def __init__(self,original):
        super().__init__(original,'exact',workers=160,rows=16)
        def branch(h,ax,residual,ids,scores):
            return run_lean(inputs(original.mlp,h,ax,ids,original.self_attn.o_proj.weight),scores,residual)
        self.branch=mx.compile(branch)
