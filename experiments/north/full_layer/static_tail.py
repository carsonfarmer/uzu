"""Static-barrier control for the cross-layer persistent kernel.

This keeps the same one-dispatch layer tail and next-layer preparation as
tail_prep.py, but each row tile computes all experts after every front task is
finished. It isolates the cost of fine per-expert dependency scheduling.
"""

import mlx.core as mx

from kernels import NAMES
from full_layer.tail_prep import (
    ATTENTION,
    HIDDEN,
    NEXT_NAMES,
    NORM,
    OUT,
    PREP,
    ROUTED,
    SIZE,
    STATE,
    header,
)


FRONT_TASKS = 160
DOWN_JOIN_TASKS = 128
PREP_TASKS = 48
TOTAL_TASKS = FRONT_TASKS + DOWN_JOIN_TASKS + 1 + PREP_TASKS

FRONT_DONE = 1
DOWN_JOIN_CLAIM = 2
DOWN_JOIN_DONE = 3
NORM_CLAIM = 4
NORM_READY = 5
PREP_CLAIM = 6
ERROR = 7
VISITS = 1024

source = f'''
uint tid=thread_position_in_threadgroup.x;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
device atomic_uint* state=(device atomic_uint*)workspace;
coherent(device) device bfloat* hidden=(coherent(device) device bfloat*)(workspace+{HIDDEN});
coherent(device) device bfloat* attention=(coherent(device) device bfloat*)(workspace+{ATTENTION});
coherent(device) device bfloat* routed=(coherent(device) device bfloat*)(workspace+{ROUTED});
coherent(device) device bfloat* out=(coherent(device) device bfloat*)(workspace+{OUT});
coherent(device) device bfloat* norm=(coherent(device) device bfloat*)(workspace+{NORM});
coherent(device) device bfloat* prep=(coherent(device) device bfloat*)(workspace+{PREP});
threadgroup float tile[64];
threadgroup bfloat hidden_tile[768];
threadgroup float norm_sums[32];
threadgroup float norm_inv[1];
threadgroup uint task;
uint idle=0;
bool front_seen=false,norm_seen=false;
while(true) {{
 if(tid==0) {{
  task=0xffffffffu;
  if(!front_seen)front_seen=atomic_load_explicit(state+{FRONT_DONE},memory_order_relaxed)=={FRONT_TASKS};
  if(front_seen) {{
   uint rowjob=atomic_fetch_add_explicit(state+{DOWN_JOIN_CLAIM},1u,memory_order_relaxed);
   if(rowjob<{DOWN_JOIN_TASKS})task={FRONT_TASKS}+rowjob;
  }}
  if(task==0xffffffffu && atomic_load_explicit(state+{DOWN_JOIN_DONE},memory_order_relaxed)=={DOWN_JOIN_TASKS}) {{
   uint expected=0;
   if(atomic_compare_exchange_weak_explicit(state+{NORM_CLAIM},&expected,1u,
      memory_order_relaxed,memory_order_relaxed))task={FRONT_TASKS + DOWN_JOIN_TASKS};
  }}
  if(!norm_seen)norm_seen=atomic_load_explicit(state+{NORM_READY},memory_order_relaxed)==1;
  if(task==0xffffffffu && norm_seen) {{
   uint job=atomic_fetch_add_explicit(state+{PREP_CLAIM},1u,memory_order_relaxed);
   if(job<{PREP_TASKS})task={FRONT_TASKS + DOWN_JOIN_TASKS + 1}+job;
  }}
  if(task==0xffffffffu) {{
   uint front=atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
   if(front<{FRONT_TASKS})task=front;
  }}
  if(task==0xffffffffu && atomic_load_explicit(state+{PREP_CLAIM},memory_order_relaxed)>={PREP_TASKS})
   task=0xfffffffeu;
 }}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(task==0xfffffffeu)break;
 if(task==0xffffffffu) {{
  if(++idle>=1000000) {{
   if(tid==0)atomic_fetch_add_explicit(state+{ERROR},1u,memory_order_relaxed);
   break;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  continue;
 }}
 idle=0;

 if(task<{FRONT_TASKS}) {{
  uint job=(task<128)?((task%2)?96+task/2:task/2):task-64;
  if(job<96)north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
  else {{
   uint rowjob=job-96;
   north_attention(rowjob,sg,lane,ax,aw,attention);
  }}
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_fetch_add_explicit(state+{FRONT_DONE},1u,memory_order_relaxed);
 }} else if(task<{FRONT_TASKS + DOWN_JOIN_TASKS}) {{
  uint rowjob=task-{FRONT_TASKS};
  if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  for(uint slot=0;slot<8;slot++) {{
   for(uint j=tid;j<768;j+=256)hidden_tile[j]=hidden[slot*768+j];
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if(sg<4)north_complete_down(slot,rowjob*16+sg*4,lane,hidden_tile,ids,down,ds,db,routed);
   threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  }}
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid<16) {{
   uint row=rowjob*16+tid;float value=0;
   for(uint slot=0;slot<8;slot++) {{
    volatile float product=float(routed[slot*2048+row])*scores[slot];
    value+=product;
   }}
   bfloat moe=bfloat(value);
   out[row]=bfloat(float(bfloat(float(attention[row])+float(moe)))+float(residual[row]));
  }}
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_fetch_add_explicit(state+{DOWN_JOIN_DONE},1u,memory_order_relaxed);
 }} else if(task=={FRONT_TASKS + DOWN_JOIN_TASKS}) {{
  if(tid<32)norm_sums[tid]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  float acc0=0,acc1=0;uint p0=tid*4,p1=(tid+256)*4;
  for(uint j=0;j<4;j++) {{float v=float(out[p0+j]);acc0+=v*v;}}
  for(uint j=0;j<4;j++) {{float v=float(out[p1+j]);acc1+=v*v;}}
  acc0=simd_sum(acc0);acc1=simd_sum(acc1);
  if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0) {{
   float total=simd_sum(norm_sums[lane]);
   if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint j=0;j<4;j++)norm[p0+j]=next_norm_w[p0+j]*bfloat(float(out[p0+j])*norm_inv[0]);
  for(uint j=0;j<4;j++)norm[p1+j]=next_norm_w[p1+j]*bfloat(float(out[p1+j])*norm_inv[0]);
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_store_explicit(state+{NORM_READY},1u,memory_order_relaxed);
 }} else {{
  uint job=task-{FRONT_TASKS + DOWN_JOIN_TASKS + 1};
  if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  north_next_gemv(job,sg,lane,norm,next_qw,next_kw,next_vw,next_rw,prep,tile);
 }}
 if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{VISITS}+task,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
}}
'''

kernel = mx.fast.metal_kernel(
    name="north_cross_layer_static_tail",
    input_names=NAMES + ["scores", "residual"] + NEXT_NAMES,
    output_names=["workspace"],
    header=header,
    source=source,
)


def run_static_tail(data, scores, residual, next_layer, workers=64, return_state=False):
    attn = next_layer.self_attn
    kernel_inputs = data + [
        scores,
        residual,
        next_layer.input_layernorm.weight,
        attn.q_proj.weight,
        attn.k_proj.weight,
        attn.v_proj.weight,
        next_layer.mlp.gate.weight,
    ]
    workspace = kernel(
        inputs=kernel_inputs,
        template=[("AUDIT", return_state)],
        grid=(workers * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(SIZE,)],
        output_dtypes=[mx.uint32],
        init_value=0,
    )[0]
    raw = workspace.view(mx.bfloat16)
    out = raw[OUT * 2 : NORM * 2].reshape(1, 1, 2048)
    norm = raw[NORM * 2 : PREP * 2].reshape(1, 1, 2048)
    p = raw[PREP * 2 :]
    values = (
        out,
        norm,
        p[:4096].reshape(1, 1, 4096),
        p[4096:4608].reshape(1, 1, 512),
        p[4608:5120].reshape(1, 1, 512),
        p[5120:].reshape(1, 1, 128),
    )
    return (*values, workspace[:STATE]) if return_state else values
