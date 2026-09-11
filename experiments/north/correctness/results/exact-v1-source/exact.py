"""Two-dispatch North branch with native accumulation order.

First dispatch mixes independent up/gate/activation and attention projection
jobs. Second computes complete expert down dots and combines experts/residual.
No split-K reassociation. This is the correctness/performance anchor before
introducing inter-threadgroup dependencies or weight-load overlap.
"""
import mlx.core as mx
from kernels import HEADER,NAMES,inputs

front=mx.fast.metal_kernel(name='north_exact_front',input_names=NAMES,
    output_names=['hidden','attention'],header='#define NCHUNK 64\n'+HEADER,
    source='''
uint group=threadgroup_position_in_grid.x;
uint lane=thread_index_in_simdgroup,sg=simdgroup_index_in_threadgroup;
threadgroup float tile[64];
for(uint task=group;task<160;task+=WORKERS) {
 uint job=INTERLEAVE ? ((task<128)?((task%2)?96+task/2:task/2):task-64) : task;
 if(job<96)north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
 else north_attention(job-96,sg,lane,ax,aw,attention);
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
''')

down=mx.fast.metal_kernel(name='north_exact_down',
    input_names=['hidden','ids','down','ds','db','scores','attention','residual'],
    output_names=['out','routed'],source='''
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint first=threadgroup_position_in_grid.x*32+sg*4;
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
   bfloat d=bfloat(value);routed[slot*2048+first+rr]=d;
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
''')

def run_exact(data,scores,residual,workers=128,interleave=True,return_parts=False):
    h,a=front(inputs=data,template=[('WORKERS',workers),('INTERLEAVE',interleave)],
        grid=(workers*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(8,768),(1,1,2048)],output_dtypes=[mx.bfloat16,mx.bfloat16])
    out,d=down(inputs=[h,data[2],*data[9:12],scores,a,residual],
        grid=(64*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(1,1,2048),(8,2048)],output_dtypes=[mx.bfloat16,mx.bfloat16])
    return (out,h,d,a) if return_parts else out
