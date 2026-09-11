"""A/B test: stage immutable down weights before/after activation readiness.

Both variants use the same claimed task order and threadgroup-memory GEMV.
Workers holding an unready down tile execute front tasks, so weight staging
does not prevent producers from making progress. This tests the mechanism on
MoE down; it is not yet next-layer QKV/router prefetch as in Cohere's design.
"""
import mlx.core as mx
from persistent.branch import header,join,SIZE,HIDDEN,ATTENTION,ROUTED,STATE

stage_header=header+'''
inline void stage_down(uint expert,uint first,uint tid,
 device const uint* down,device const bfloat* ds,device const bfloat* db,
 threadgroup uint* wb,threadgroup bfloat* sb,threadgroup bfloat* bb) {
 for(uint i=tid;i<3072;i+=256)wb[i]=down[(expert*2048+first)*96+i];
 for(uint i=tid;i<384;i+=256) {
  sb[i]=ds[(expert*2048+first)*12+i];bb[i]=db[(expert*2048+first)*12+i];
 }
}
inline void staged_down(uint slot,uint first,uint sg,uint lane,
 coherent(device) device const bfloat* hidden,
 threadgroup const uint* wb,threadgroup const bfloat* sb,threadgroup const bfloat* bb,
 device bfloat* routed) {
 float acc[4]={0};
 for(uint k=lane*8;k<768;k+=256) {
  coherent(device) device const bfloat* xp=hidden+slot*768+k;
  float v[8],xsum=0;
  for(uint j=0;j<8;j+=4) {
   xsum+=float(xp[j]+xp[j+1]+xp[j+2]+xp[j+3]);
   v[j]=float(xp[j]);v[j+1]=float(xp[j+1])/16.0f;
   v[j+2]=float(xp[j+2])/256.0f;v[j+3]=float(xp[j+3])/4096.0f;
  }
  for(uint rr=0;rr<4;rr++) {
   uint row=sg*4+rr,si=row*12+k/64;
   threadgroup const ushort* w=(threadgroup const ushort*)(wb+row*96+k/8);
   float sum=0;
   for(uint j=0;j<2;j++)sum+=(v[j*4]*(w[j]&15)+v[j*4+1]*(w[j]&240)+v[j*4+2]*(w[j]&3840)+v[j*4+3]*(w[j]&61440));
   acc[rr]+=float(sb[si])*sum+float(bb[si])*xsum;
  }
 }
 for(uint rr=0;rr<4;rr++) {
  float value=simd_sum(acc[rr]);
  if(lane==0)routed[slot*2048+first+sg*4+rr]=bfloat(value);
 }
}
'''

staged=mx.fast.metal_kernel(name='north_staged_branch',input_names=['x','ax','ids','up','us','ub','gate','gs','gb','down','ds','db','aw'],
 output_names=['workspace'],header=stage_header,source='''
uint tid=thread_position_in_threadgroup.x;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
device atomic_uint* state=(device atomic_uint*)workspace;
coherent(device) device bfloat* hidden=(coherent(device) device bfloat*)(workspace+1024);
device bfloat* attention=(device bfloat*)(workspace+4096);
device bfloat* routed=(device bfloat*)(workspace+5120);
threadgroup float tile[64];
threadgroup uint wb[3072];threadgroup bfloat sb[384],bb[384];
threadgroup uint pending,task;
while(true) {
 if(tid==0)pending=atomic_fetch_add_explicit(state+109,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(pending>=512)break;
 uint slot=pending/64,first=pending%64*32;
 if(PREFETCH) {
  stage_down(ids[slot],first,tid,down,ds,db,wb,sb,bb);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid==0) {
   bool early=false;
   for(uint j=0;j<12;j++)early=early || atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed)==0;
   if(early)atomic_fetch_add_explicit(state+110,1u,memory_order_relaxed);
  }
 }
 uint idle=0;
 while(true) {
  if(tid==0) {
   bool ready=true;
   for(uint j=0;j<12;j++)ready=ready && atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed)==1;
   task=0xffffffffu;
   if(ready)task=0xfffffffeu;
   else {
    uint front=atomic_load_explicit(state,memory_order_relaxed);
    if(front<160) {
     front=atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
     if(front<160)task=front;
    }
   }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(task==0xfffffffeu)break;
  if(task==0xffffffffu) {
   if(++idle>=1000000) {
    if(tid==0)atomic_fetch_add_explicit(state+105,1u,memory_order_relaxed);
    break;
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   continue;
  }
  idle=0;
  uint job=(task<128)?((task%2)?96+task/2:task/2):task-64;
  if(job<96) {
   north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
   threadgroup_barrier(mem_flags::mem_device);
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(tid==0) {
    atomic_fetch_add_explicit(state+108,1u,memory_order_relaxed);
    atomic_store_explicit(state+1+job,1u,memory_order_relaxed);
   }
  } else north_attention(job-96,sg,lane,ax,aw,attention);
  if(tid==0)atomic_fetch_add_explicit(state+128+task,1u,memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
 }
 uint published=0;
 for(uint j=0;j<12;j++)published+=atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed);
 atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
 if(published!=12 && tid==0)atomic_fetch_add_explicit(state+105,1u,memory_order_relaxed);
 if(!PREFETCH) {
  stage_down(ids[slot],first,tid,down,ds,db,wb,sb,bb);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(tid==0) {
  uint expected=0;
  if(atomic_compare_exchange_weak_explicit(state+106,&expected,1u,memory_order_relaxed,memory_order_relaxed))
   atomic_store_explicit(state+107,atomic_load_explicit(state+108,memory_order_relaxed),memory_order_relaxed);
 }
 staged_down(slot,first,sg,lane,hidden,wb,sb,bb,routed);
 if(tid==0)atomic_fetch_add_explicit(state+128+160+pending,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
}
// All hidden tiles may become ready before the last attention task is claimed.
// Drain independent front work before exiting, even after all down jobs issued.
while(true) {
 if(tid==0)task=atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(task>=160)break;
 uint job=(task<128)?((task%2)?96+task/2:task/2):task-64;
 if(job<96) {
  north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0) {
   atomic_fetch_add_explicit(state+108,1u,memory_order_relaxed);
   atomic_store_explicit(state+1+job,1u,memory_order_relaxed);
  }
 } else north_attention(job-96,sg,lane,ax,aw,attention);
 if(tid==0)atomic_fetch_add_explicit(state+128+task,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_device|mem_flags::mem_threadgroup);
}
''')

def run_staged(data,scores,residual,workers=64,prefetch=True,return_parts=False):
 workspace=staged(inputs=data,template=[('PREFETCH',prefetch)],grid=(workers*256,1,1),
  threadgroup=(256,1,1),output_shapes=[(SIZE,)],output_dtypes=[mx.uint32],init_value=0)[0]
 out=join(inputs=[workspace,scores,residual],grid=(2048,1,1),threadgroup=(256,1,1),
  output_shapes=[(1,1,2048)],output_dtypes=[mx.bfloat16])[0]
 if not return_parts:return out
 raw=workspace.view(mx.bfloat16)
 return out,raw[HIDDEN*2:ATTENTION*2].reshape(8,768),raw[ROUTED*2:].reshape(8,2048),raw[ATTENTION*2:ROUTED*2].reshape(1,1,2048),workspace[:STATE]
