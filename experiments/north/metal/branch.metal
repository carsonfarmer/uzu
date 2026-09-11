#include <metal_stdlib>
using namespace metal;
constant uint D=2048,H=768,A=4096,E=8,CHUNKS=24,EXPERT_TASKS=192,TASKS=256;
struct Config { uint phase, workers, kind, unused; };
#define BUFFERS \
 device const bfloat* x [[buffer(0)]], device const bfloat* residual [[buffer(1)]], \
 device const bfloat* ax [[buffer(2)]], device const bfloat* up [[buffer(3)]], \
 device const bfloat* gate [[buffer(4)]], device const bfloat* down [[buffer(5)]], \
 device const bfloat* aw [[buffer(6)]], device const float* scores [[buffer(7)]], \
 device bfloat* hidden [[buffer(8)]], device float* partial [[buffer(9)]], \
 device bfloat* attention [[buffer(10)]], device bfloat* output [[buffer(11)]], \
 device atomic_uint* head [[buffer(12)]], device atomic_uint* visits [[buffer(13)]], \
 constant Config& cfg [[buffer(14)]], uint group [[threadgroup_position_in_grid]], \
 uint tid [[thread_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]], \
 uint sg [[simdgroup_index_in_threadgroup]]

inline void make_hidden(uint job,uint sg,uint lane,device const bfloat* x,
 device const bfloat* up,device const bfloat* gate,device bfloat* hidden,threadgroup float* tile) {
 uint expert=job/CHUNKS,chunk=job%CHUNKS;
 for(uint rr=sg;rr<32;rr+=8) {
  uint row=chunk*32+rr,base=(expert*H+row)*D;
  float u=0,g=0;
  for(uint k=lane;k<D;k+=32) { u+=float(x[k])*float(up[base+k]); g+=float(x[k])*float(gate[base+k]); }
  u=simd_sum(u);g=simd_sum(g);
  if(lane==0) {
   bfloat ub=bfloat(u),gb=bfloat(g);
   bfloat act=bfloat(float(gb)/(1.0f+exp(-float(gb))));
   bfloat h=bfloat(float(ub)*float(act));
   hidden[expert*H+row]=h;tile[rr]=float(h);
  }
 }
}
inline void make_partial(uint job,uint tid,device const bfloat* down,
 threadgroup const float* tile,device float* partial) {
 uint expert=job/CHUNKS,chunk=job%CHUNKS;
 for(uint row=tid;row<D;row+=256) {
  float sum=0;
  for(uint k=0;k<32;k++) sum+=float(down[(expert*D+row)*H+chunk*32+k])*tile[k];
  partial[job*D+row]=sum;
 }
}
inline void make_attention(uint block,uint sg,uint lane,device const bfloat* ax,
 device const bfloat* aw,device bfloat* attention) {
 for(uint rr=sg;rr<32;rr+=8) {
  uint row=block*32+rr;
  float sum=0;
  for(uint k=lane;k<A;k+=32)sum+=float(ax[k])*float(aw[row*A+k]);
  sum=simd_sum(sum);
  if(lane==0) attention[row]=bfloat(sum);
 }
}
inline void job_body(uint job,uint tid,uint sg,uint lane,device const bfloat* x,
 device const bfloat* ax,device const bfloat* up,device const bfloat* gate,
 device const bfloat* down,device const bfloat* aw,device bfloat* hidden,
 device float* partial,device bfloat* attention,threadgroup float* tile) {
 if(job<EXPERT_TASKS) {
  make_hidden(job,sg,lane,x,up,gate,hidden,tile);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  make_partial(job,tid,down,tile,partial);
 } else make_attention(job-EXPERT_TASKS,sg,lane,ax,aw,attention);
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
}
kernel void branch(BUFFERS) {
 threadgroup float tile[32];
 threadgroup uint claimed;
 if(cfg.phase==0) {
  make_hidden(group,sg,lane,x,up,gate,hidden,tile);
 } else if(cfg.phase==1) {
  if(tid<32)tile[tid]=float(hidden[(group/CHUNKS)*H+(group%CHUNKS)*32+tid]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  make_partial(group,tid,down,tile,partial);
 } else if(cfg.phase==2) {
  make_attention(group,sg,lane,ax,aw,attention);
 } else if(cfg.phase==3) {
  uint row=group*256+tid;
  float sum=0;
  for(uint expert=0;expert<E;expert++){
   float value=0;
   for(uint chunk=0;chunk<CHUNKS;chunk++)value+=partial[(expert*CHUNKS+chunk)*D+row];
   sum+=float(bfloat(value))*scores[expert];
  }
  bfloat ff=bfloat(sum);
  output[row]=bfloat(float(bfloat(float(attention[row])+float(ff)))+float(residual[row]));
 } else if(cfg.phase==4) {
  // One ready local chain per group; phase 5 can include attention tasks.
  uint mapped=cfg.kind ? ((group%4==3)?EXPERT_TASKS+group/4:(group/4)*3+group%4) : group;
  job_body(mapped,tid,sg,lane,x,ax,up,gate,down,aw,hidden,partial,attention,tile);
  if(tid==0)atomic_fetch_add_explicit(visits+mapped,1u,memory_order_relaxed);
 } else if(cfg.phase==5) {
  for(uint job=group;job<TASKS;job+=cfg.workers) {
   uint mapped=cfg.kind ? ((job%4==3)?EXPERT_TASKS+job/4:(job/4)*3+job%4) : job;
   job_body(mapped,tid,sg,lane,x,ax,up,gate,down,aw,hidden,partial,attention,tile);
   if(tid==0)atomic_fetch_add_explicit(visits+mapped,1u,memory_order_relaxed);
  }
 } else if(cfg.phase==6) {
  while(true) {
   if(tid==0)claimed=atomic_fetch_add_explicit(head,1u,memory_order_relaxed);
   threadgroup_barrier(mem_flags::mem_threadgroup);
   uint job=claimed;
   if(job>=TASKS)break;
   // Interleave one attention tile for every three expert tiles.
   uint mapped=(job%4==3)?EXPERT_TASKS+job/4:(job/4)*3+job%4;
   job_body(mapped,tid,sg,lane,x,ax,up,gate,down,aw,hidden,partial,attention,tile);
   if(tid==0)atomic_fetch_add_explicit(visits+mapped,1u,memory_order_relaxed);
   threadgroup_barrier(mem_flags::mem_threadgroup);
  }
 }
}
