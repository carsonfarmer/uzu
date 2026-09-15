"""Independent QKV/router tasks; no inter-threadgroup communication.

Retains MLX's distinct narrow-router and QKV reduction orders. The optional
RMS variant redundantly normalizes in each group to eliminate a dispatch.
Arithmetic is adapted from Apple MLX; see NOTICE.md.
"""
from pathlib import Path
import mlx.core as mx

HEADER = Path(__file__).with_suffix('.h').read_text()
NORM = '''
threadgroup bfloat normalized[2048];
threadgroup float sums[32];
threadgroup float inv;
uint tid=thread_position_in_threadgroup.x;
if(tid<32)sums[tid]=0;
threadgroup_barrier(mem_flags::mem_threadgroup);
float a=0,b=0;
uint p0=tid*4,p1=(tid+256)*4;
for(uint j=0;j<4;j++){float v=float(x[p0+j]);a+=v*v;}
for(uint j=0;j<4;j++){float v=float(x[p1+j]);b+=v*v;}
a=simd_sum(a);b=simd_sum(b);
if(lane==0){sums[sg]=a;sums[sg+8]=b;}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==0){float total=simd_sum(sums[lane]);if(lane==0)inv=metal::precise::rsqrt(total/2048.0f+1e-6f);}
threadgroup_barrier(mem_flags::mem_threadgroup);
for(uint j=0;j<4;j++){
 normalized[p0+j]=nw[p0+j]*bfloat(float(x[p0+j])*inv);
 normalized[p1+j]=nw[p1+j]*bfloat(float(x[p1+j])*inv);
}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(threadgroup_position_in_grid.x==0)for(uint j=0;j<8;j++)norm[tid*8+j]=normalized[tid*8+j];
'''

def make_kernel(normalize):
    header=HEADER
    if normalize:
        header=header.replace('device const bfloat* x,', 'threadgroup const bfloat* x,')
    return mx.fast.metal_kernel(
        name='north_independent_prep'+('_rms' if normalize else ''),
        input_names=['x','qw','kw','vw','rw']+(['nw'] if normalize else []),
        output_names=['prep']+(['norm'] if normalize else []),
        header=header,
        source='uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;\n'
        + (NORM if normalize else '') + '''
threadgroup float scratch[64];
uint job=threadgroup_position_in_grid.x;
north_next_prep_tiled(job,ROWS,ROUTER_ROWS,sg,lane,'''
        + ('normalized' if normalize else 'x') + ''',qw,kw,vw,rw,prep,scratch);
''')

KERNELS={False:make_kernel(False),True:make_kernel(True)}

def run(x, weights, rows=32, router_rows=4, norm_weight=None):
    normalize=norm_weight is not None
    assert rows in (32,64,128,256,512) and router_rows in (4,8,16,32)
    outputs=KERNELS[normalize](
        inputs=[x,*weights]+([norm_weight] if normalize else []),
        template=[('ROWS',rows),('ROUTER_ROWS',router_rows)],
        grid=((5120//rows+128//router_rows)*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(1,1,5248)]+([(1,1,2048)] if normalize else []),
        output_dtypes=[mx.bfloat16]*(2 if normalize else 1))
    return outputs
