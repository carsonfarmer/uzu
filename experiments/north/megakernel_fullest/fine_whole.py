"""Whole-model ready DAG replacing global MoE phase boundaries, same primitives.

Each layer has440 task states. Completed states are never reset within a decode,
so a delayed worker cannot claim a reused state from an earlier layer. Prefix,
cache carry, and head retain the established phase queue. No prefetch surrogate.
"""
HEADER=r'''
inline bool north_node_done(device atomic_uint* tasks,uint node) {
 return atomic_load_explicit(tasks+node,memory_order_relaxed)==2;
}
inline uint north_front_node(uint logical) {return 136+(logical<32?logical*2:logical+32);}
inline bool north_node_ready(device atomic_uint* t,uint node) {
 if(node==0)return true;
 if(node<97)return north_node_done(t,0);
 if(node<136) {
  uint p=node-97;
  if(p<32)return north_node_done(t,1+p*2)&&north_node_done(t,2+p*2);
  if(p<36)return north_node_done(t,65+(p-32)*2)&&north_node_done(t,66+(p-32)*2);
  if(p<38){for(uint j=0;j<4;j++)if(!north_node_done(t,73+(p-36)*4+j))return false;return true;}
  for(uint j=81;j<97;j++)if(!north_node_done(t,j))return false;
  return true;
 }
 if(node<264) {
  uint task=node-136,job=task<64?(task%2?96+task/2:task/2):task-32;
  if(job<96)return north_node_done(t,135);
  uint head=job-96;return north_node_done(t,97+head)&&north_node_done(t,129+head/8)&&north_node_done(t,133+head/16);
 }
 if(node<424) {
  uint task=node-264;
  if(task<128){uint slot=task/16;for(uint j=0;j<12;j++)if(!north_node_done(t,north_front_node(slot*12+j)))return false;return true;}
  for(uint h=0;h<32;h++)if(!north_node_done(t,137+h*2))return false;
  return true;
 }
 uint row=node-424;
 for(uint slot=0;slot<8;slot++)if(!north_node_done(t,264+slot*16+row))return false;
 return north_node_done(t,392+row*2)&&north_node_done(t,393+row*2);
}
inline uint north_node_phase(uint n){return n<1?0:(n<97?1:(n<136?2:(n<264?3:(n<424?4:5))));}
inline uint north_node_job(uint n){return n<1?n:(n<97?n-1:(n<136?n-97:(n<264?n-136:(n<424?n-264:n-424))));}
inline uint north_claim_node(device atomic_uint* t,uint cursor) {
 for(uint j=0;j<440;j++) {
  uint n=(cursor+j)%440;
  if(atomic_load_explicit(t+n,memory_order_relaxed)!=0 || !north_node_ready(t,n))continue;
  uint expected=0;
  if(atomic_compare_exchange_weak_explicit(t+n,&expected,1u,memory_order_relaxed,memory_order_relaxed))return n;
 }
 return 0xffffffffu;
}
'''

def build(header,source,size):
    s=source
    s=s.replace('threadgroup uint dynamic_task,queue_stage,queue_task,queue_tasks;',
        f'threadgroup uint dynamic_task,queue_stage,queue_task,queue_tasks,fine_node,fine_layer;\nuint fine_cursor=group%440,idle=0;\ndevice atomic_uint* fine_states=(device atomic_uint*)(workspace+{size});')
    needle='   queue_task=(queue_stage<total_stages)?north_queue_claim(state,queue_stage,queue_tasks):0xffffffffu;'
    assert s.count(needle)==1
    s=s.replace(needle,'''   fine_node=0xffffffffu;
   if(queue_stage>=moe_start && queue_stage<head_start) {
    fine_layer=(queue_stage-moe_start)/6;
    fine_node=north_claim_node(fine_states+fine_layer*512,fine_cursor);
    queue_task=fine_node==0xffffffffu?fine_node:north_node_job(fine_node);
    if(fine_node!=0xffffffffu)queue_stage=moe_start+fine_layer*6+north_node_phase(fine_node);
   } else queue_task=(queue_stage<total_stages)?north_queue_claim(state,queue_stage,queue_tasks):0xffffffffu;''')
    needle='   if(tid==0)north_queue_finish(state,queue_stage,queue_tasks);'
    assert s.count(needle)==1
    s=s.replace(needle,'''   if(tid==0) {
    if(fine_node!=0xffffffffu) {
     device atomic_uint* t=fine_states+fine_layer*512;
     atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
     atomic_store_explicit(t+fine_node,2u,memory_order_relaxed);
     uint complete=atomic_fetch_add_explicit(t+511,1u,memory_order_relaxed)+1;
     if(complete==440)atomic_store_explicit(state,moe_start+(fine_layer+1)*6,memory_order_relaxed);
     fine_cursor=(fine_node+1)%440;
    } else north_queue_finish(state,queue_stage,queue_tasks);
   }
   idle=0;''')
    needle='   while(atomic_load_explicit(state,memory_order_relaxed)==queue_stage) {}'
    assert s.count(needle)==1
    s=s.replace(needle,'''   // Return to the ready scanner; do not wait for an entire layer here.
   if(++idle>1000000)atomic_store_explicit(state+3,1u,memory_order_relaxed);''')
    # Finite diagnostic abort shared by every group, inspected by numerical gate.
    s=s.replace('  if(queue_stage>=total_stages)break;','  if(queue_stage>=total_stages || atomic_load_explicit(state+3,memory_order_relaxed)!=0)break;',1)
    return header+HEADER,s

_KERNELS={}

def run(weights,x,key_cache,value_cache,position,workers=32,do_head=True,do_prefix=True,scheduler="scan"):
    import mlx.core as mx
    from whole_pass import kernel as old
    from whole_pass.pack import WEIGHT_NAMES,HEAD_NAMES,PREFIX_NAMES
    layers=weights['norm_w'].shape[0];capacity=key_cache.shape[2]
    assert key_cache.shape==value_cache.shape and key_cache.shape[0]==layers+int(do_prefix)
    assert scheduler in ("scan","affinity","progress","prefetch_early","prefetch_late")
    if scheduler not in _KERNELS:
        builder=build
        if scheduler=="affinity":
            from affinity_whole import build as builder
        if scheduler=="progress":
            from progress_whole import build as builder
        if scheduler.startswith("prefetch_"):
            from prefetch_whole import build as prefetch_builder
            builder=lambda h,s,z:prefetch_builder(h,s,z,early=scheduler=="prefetch_early")
        h,s=builder(old.header,old.source,old.SIZE)
        _KERNELS[scheduler]=mx.fast.metal_kernel(name='north_whole_ready_dag_'+scheduler,input_names=['x_in','key_cache','value_cache','params',*WEIGHT_NAMES,*HEAD_NAMES,*PREFIX_NAMES,'sigmoid_table'],output_names=['key_out','value_out','workspace','logits'],header=h,source=s)
    outputs=_KERNELS[scheduler](inputs=[x,key_cache,value_cache,mx.array([position,capacity],mx.uint32),*[weights[n] for n in WEIGHT_NAMES],*[weights[n] for n in HEAD_NAMES],*[weights[n] for n in PREFIX_NAMES],old.sigmoid_lut()],
        template=[('NLAYERS',layers),('WORKERS',workers),('FINE',False),('SAFE_QUEUE',True),('PREFETCH_STAGES',0),('PREP_ROWS',64),('OPROJ_ROWS',64),('ROUTER_ROWS',8),('DO_HEAD',do_head),('LM_ROWS',512),('DO_PREFIX',do_prefix)],
        grid=(workers*256,1,1),threadgroup=(256,1,1),output_shapes=[key_cache.shape,value_cache.shape,(old.SIZE+layers*512,),(1,1,262144)],output_dtypes=[mx.bfloat16,mx.bfloat16,mx.uint32,mx.bfloat16],init_value=0)
    raw=outputs[2].view(mx.bfloat16);out=raw[old.X*2:old.NORM*2].reshape(1,1,2048)
    return out,outputs[0],outputs[1],outputs[2],outputs[3]
