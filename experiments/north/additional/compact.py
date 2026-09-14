"""Omit the unused routed diagnostic output; preserve exact MLX-derived math.

Arithmetic is copied from ../quantized/exact.py; see ../quantized/NOTICE.md.
"""
import mlx.core as mx
from exact import front
from kernels import inputs
from layers import PreparedBranch

SOURCE = r'''
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint first=threadgroup_position_in_grid.x*ROWS+sg*4;
float combined[4]={0};
for(uint slot=0;slot<8;slot++) {
 uint expert=ids[slot];float acc[4]={0};
 for(uint k=lane*8;k<768;k+=256) {
  device const bfloat* xp=hidden+slot*768+k;
  float v[8],xsum=0;
  for(uint j=0;j<8;j+=4) {
   xsum+=float(xp[j]+xp[j+1]+xp[j+2]+xp[j+3]);
   v[j]=float(xp[j]);v[j+1]=float(xp[j+1])/16.0f;
   v[j+2]=float(xp[j+2])/256.0f;v[j+3]=float(xp[j+3])/4096.0f;
  }
  for(uint rr=0;rr<4;rr++) {
   uint row=expert*2048+first+rr,si=row*12+k/64;
   device const ushort* w=(device const ushort*)(down+row*96+k/8);
   float sum=0;
   for(uint j=0;j<2;j++)sum+=(v[j*4]*(w[j]&15)+v[j*4+1]*(w[j]&240)+v[j*4+2]*(w[j]&3840)+v[j*4+3]*(w[j]&61440));
   acc[rr]+=float(ds[si])*sum+float(db[si])*xsum;
  }
 }
 for(uint rr=0;rr<4;rr++) {
  float value=simd_sum(acc[rr]);
  if(lane==0) {
   bfloat d=bfloat(value);
   volatile float product=float(d)*scores[slot];
   combined[rr]+=product;
  }
 }
}
if(lane==0)for(uint rr=0;rr<4;rr++) {
 uint row=first+rr;
 bfloat moe=bfloat(combined[rr]);
 out[row]=bfloat(float(bfloat(float(attention[row])+float(moe)))+float(residual[row]));
}
'''

down=mx.fast.metal_kernel(name='north_compact_exact_down',
    input_names=['hidden','ids','down','ds','db','scores','attention','residual'],
    output_names=['out'],source=SOURCE)

def run_compact(data,scores,residual):
    h,a=front(inputs=data,template=[('WORKERS',160),('INTERLEAVE',True)],
        grid=(160*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(8,768),(1,1,2048)],output_dtypes=[mx.bfloat16,mx.bfloat16])
    return down(inputs=[h,data[2],*data[9:12],scores,a,residual],template=[('ROWS',16)],
        grid=(16384,1,1),threadgroup=(128,1,1),
        output_shapes=[(1,1,2048)],output_dtypes=[mx.bfloat16])[0]

class CompactPrepared(PreparedBranch):
    def __init__(self,original):
        super().__init__(original,64,8,True)
        def branch(h,ax,residual,ids,scores):
            return run_compact(inputs(original.mlp,h,ax,ids,original.self_attn.o_proj.weight),scores,residual)
        self.branch=mx.compile(branch)
