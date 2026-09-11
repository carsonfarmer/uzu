"""Share dependency acquisition and hidden loads across a consumer group.

Thread 0 observes all producer flags and acquires their device writes. A
device-memory threadgroup barrier carries those writes to all consumers;
coherent loads stage the activation once into threadgroup memory. The dot
product retains the exact arithmetic of the audited reference implementation.
"""
import mlx.core as mx
from persistent.branch import SCHEDULE_SOURCE,header,join,SIZE,HIDDEN,ATTENTION,ROUTED,STATE
from kernels import NAMES

def replace_once(source,before,after):
    assert source.count(before)==1,before
    return source.replace(before,after)

fast_header=replace_once(header,'coherent(device) device const bfloat* hidden,device const uint* ids,',
    'threadgroup const bfloat* hidden,device const uint* ids,')
fast_header=replace_once(fast_header,'coherent(device) device const bfloat* xp=hidden+slot*768+k;',
    'threadgroup const bfloat* xp=hidden+k;')
source=replace_once(SCHEDULE_SOURCE,'threadgroup float tile[64];',
    'threadgroup float tile[64];\nthreadgroup bfloat hidden_tile[768];')
source=replace_once(source,'uint rotation=group%8,idle=0;',
    'uint rotation=group%8,idle=0,ready_mask=0;\nbool all_seen=false;')
source=replace_once(source,'''  bool grid_ready=true;
  if(!FINE)for(uint j=1;j<=96;j++)
   grid_ready=grid_ready && atomic_load_explicit(state+j,memory_order_relaxed)==1;''',
'''  bool grid_ready=true;
  if(!FINE && !all_seen)for(uint j=1;j<=96;j++)
   grid_ready=grid_ready && atomic_load_explicit(state+j,memory_order_relaxed)==1;
  all_seen=grid_ready;''')
source=replace_once(source,'''   bool ready=grid_ready;
   for(uint j=0;j<12;j++)ready=ready && atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed)==1;''',
'''   bool ready=(ready_mask & (1u<<slot))!=0;
   if(!ready && grid_ready) {
    ready=true;
    for(uint j=0;j<12;j++)ready=ready && atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed)==1;
    if(ready)ready_mask|=1u<<slot;
   }''')
source=replace_once(source,'''  // Every consumer thread observes every publication before reading hidden.
  uint published=0;
  for(uint j=0;j<12;j++)published+=atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(published!=12 && tid==0)atomic_fetch_add_explicit(state+105,1u,memory_order_relaxed);''',
'''  // One acquiring thread publishes visibility through the group barrier.
  if(tid==0) {
   uint published=0;
   for(uint j=0;j<12;j++)published+=atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed);
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(published!=12)atomic_fetch_add_explicit(state+105,1u,memory_order_relaxed);
  }
  threadgroup_barrier(mem_flags::mem_device);
  for(uint j=tid;j<768;j+=256)hidden_tile[j]=hidden[slot*768+j];
  threadgroup_barrier(mem_flags::mem_threadgroup);''')
source=replace_once(source,'north_complete_down(slot,job%64*32+sg*4,lane,hidden,ids,down,ds,db,routed);',
    'north_complete_down(slot,job%64*32+sg*4,lane,hidden_tile,ids,down,ds,db,routed);')
source=replace_once(source,'atomic_fetch_add_explicit(state+108,1u,memory_order_relaxed);',
    'if(AUDIT)atomic_fetch_add_explicit(state+108,1u,memory_order_relaxed);')
source=replace_once(source,'if(tid==0) {\n   uint expected=0;',
    'if(tid==0 && AUDIT) {\n   uint expected=0;')
source=replace_once(source,'if(tid==0)atomic_fetch_add_explicit(state+128+task,1u,memory_order_relaxed);',
    'if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+128+task,1u,memory_order_relaxed);')
kernel=mx.fast.metal_kernel(name='north_ready_shared_acquire',input_names=NAMES,
    output_names=['workspace'],header=fast_header,source=source)

def run_fast(data,scores,residual,workers=64,fine=True,return_parts=False):
 workspace=kernel(inputs=data,template=[('FINE',fine),('INTERLEAVE',True),('AUDIT',return_parts)],
    grid=(workers*256,1,1),threadgroup=(256,1,1),output_shapes=[(SIZE,)],output_dtypes=[mx.uint32],init_value=0)[0]
 out=join(inputs=[workspace,scores,residual],grid=(2048,1,1),threadgroup=(256,1,1),
    output_shapes=[(1,1,2048)],output_dtypes=[mx.bfloat16])[0]
 if not return_parts:return out
 raw=workspace.view(mx.bfloat16)
 return out,raw[HIDDEN*2:ATTENTION*2].reshape(8,768),raw[ROUTED*2:].reshape(8,2048),raw[ATTENTION*2:ROUTED*2].reshape(1,1,2048),workspace[:STATE]
