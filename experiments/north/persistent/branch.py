"""Experimental ready-work scheduler for the exact North W4 branch.

Metal 3.2 coherent buffers and device-scoped sequential fences publish hidden
tiles. Workers claim only ready down jobs, so progress does not require every
threadgroup to be resident. An explicit grid-wide readiness ablation uses the
same arithmetic and work allocator. Workspace initialization and final join
remain separate dispatches and must be included in timing.
"""
from pathlib import Path
import mlx.core as mx
from kernels import HEADER,NAMES

STATE=1024
HIDDEN=STATE
ATTENTION=HIDDEN+3072
ROUTED=ATTENTION+1024
SIZE=ROUTED+8192

header='#define NCHUNK 64\n'+HEADER.replace('device bfloat* hidden,','coherent(device) device bfloat* hidden,')
header+='\n'+Path(__file__).with_name('down.h').read_text()
SCHEDULE_SOURCE='''
uint tid=thread_position_in_threadgroup.x;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint group=threadgroup_position_in_grid.x;
device atomic_uint* state=(device atomic_uint*)workspace;
coherent(device) device bfloat* hidden=(coherent(device) device bfloat*)(workspace+1024);
device bfloat* attention=(device bfloat*)(workspace+4096);
device bfloat* routed=(device bfloat*)(workspace+5120);
threadgroup float tile[64];
threadgroup uint task;
uint rotation=group%8,idle=0;
while(true) {
 if(tid==0) {
  task=0xffffffffu;
  bool all_claimed=true;
  bool grid_ready=true;
  if(!FINE)for(uint j=1;j<=96;j++)
   grid_ready=grid_ready && atomic_load_explicit(state+j,memory_order_relaxed)==1;
  for(uint ee=0;ee<8;ee++) {
   uint slot=(rotation+ee)%8;
   uint claimed=atomic_load_explicit(state+97+slot,memory_order_relaxed);
   if(claimed>=64)continue;
   all_claimed=false;
   bool ready=grid_ready;
   for(uint j=0;j<12;j++)ready=ready && atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed)==1;
   if(ready) {
    uint rowjob=atomic_fetch_add_explicit(state+97+slot,1u,memory_order_relaxed);
    if(rowjob<64){task=160+slot*64+rowjob;break;}
   }
  }
  if(task==0xffffffffu) {
   uint front=atomic_load_explicit(state,memory_order_relaxed);
   if(front<160) {
    front=atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
    if(front<160)task=front;
   }
   if(task==0xffffffffu && all_claimed)task=0xfffffffeu;
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
 if(task<160) {
  uint job=INTERLEAVE ? ((task<128)?((task%2)?96+task/2:task/2):task-64) : task;
  if(job<96) {
   north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
   threadgroup_barrier(mem_flags::mem_device);
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(tid==0) {
    atomic_fetch_add_explicit(state+108,1u,memory_order_relaxed);
    atomic_store_explicit(state+1+job,1u,memory_order_relaxed);
   }
  } else north_attention(job-96,sg,lane,ax,aw,attention);
 } else {
  uint job=task-160,slot=job/64;
  // Every consumer thread observes every publication before reading hidden.
  uint published=0;
  for(uint j=0;j<12;j++)published+=atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(published!=12 && tid==0)atomic_fetch_add_explicit(state+105,1u,memory_order_relaxed);
  if(tid==0) {
   uint expected=0;
   if(atomic_compare_exchange_weak_explicit(state+106,&expected,1u,memory_order_relaxed,memory_order_relaxed))
    atomic_store_explicit(state+107,atomic_load_explicit(state+108,memory_order_relaxed),memory_order_relaxed);
  }
  north_complete_down(slot,job%64*32+sg*4,lane,hidden,ids,down,ds,db,routed);
 }
 if(tid==0)atomic_fetch_add_explicit(state+128+task,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 rotation=(rotation+1)%8;
}
'''
scheduled=mx.fast.metal_kernel(name='north_ready_branch',input_names=NAMES,
    output_names=['workspace'],header=header,source=SCHEDULE_SOURCE)

join=mx.fast.metal_kernel(name='north_ready_join',
    input_names=['workspace','scores','residual'],output_names=['out'],source='''
uint row=thread_position_in_grid.x;
device const bfloat* attention=(device const bfloat*)(workspace+4096);
device const bfloat* routed=(device const bfloat*)(workspace+5120);
float value=0;
for(uint slot=0;slot<8;slot++) {
 volatile float product=float(routed[slot*2048+row])*scores[slot];
 value+=product;
}
bfloat moe=bfloat(value);
out[row]=workspace[105] ? bfloat(NAN) :
 bfloat(float(bfloat(float(attention[row])+float(moe)))+float(residual[row]));
''')

def run_scheduled(data,scores,residual,workers=64,fine=True,interleave=True,return_parts=False):
    workspace=scheduled(inputs=data,template=[('FINE',fine),('INTERLEAVE',interleave)],
        grid=(workers*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(SIZE,)],output_dtypes=[mx.uint32],init_value=0)[0]
    out=join(inputs=[workspace,scores,residual],grid=(2048,1,1),threadgroup=(256,1,1),
        output_shapes=[(1,1,2048)],output_dtypes=[mx.bfloat16])[0]
    if not return_parts:return out
    raw=workspace.view(mx.bfloat16)
    return out,raw[HIDDEN*2:ATTENTION*2].reshape(8,768),raw[ROUTED*2:].reshape(8,2048),raw[ATTENTION*2:ROUTED*2].reshape(1,1,2048),workspace[:STATE]
