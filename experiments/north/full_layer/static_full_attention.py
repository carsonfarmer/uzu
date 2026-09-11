"""Static per-worker full-attention schedule with three device-wide phase barriers."""

import mlx.core as mx

from kernels import inputs as branch_inputs
from full_layer.full_attention import (
    ATTENTION,
    AX,
    DOWN_TASKS,
    FRONT_TASKS,
    FULL_NAMES,
    HIDDEN,
    JOIN_TASKS,
    NEXT_NAMES,
    NORM,
    OUT,
    PREP,
    PREP_TASKS,
    ROUTED,
    SIZE,
    STATE,
    TOTAL_TASKS,
    VISITS,
    header,
)


source = f'''
uint tid=thread_position_in_threadgroup.x;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint group=threadgroup_position_in_grid.x;
device atomic_uint* state=(device atomic_uint*)workspace;
coherent(device) device bfloat* hidden=(coherent(device) device bfloat*)(workspace+{HIDDEN});
coherent(device) device bfloat* ax=(coherent(device) device bfloat*)(workspace+{AX});
coherent(device) device bfloat* attention=(coherent(device) device bfloat*)(workspace+{ATTENTION});
coherent(device) device bfloat* routed=(coherent(device) device bfloat*)(workspace+{ROUTED});
coherent(device) device bfloat* out=(coherent(device) device bfloat*)(workspace+{OUT});
coherent(device) device bfloat* norm=(coherent(device) device bfloat*)(workspace+{NORM});
coherent(device) device bfloat* prep=(coherent(device) device bfloat*)(workspace+{PREP});
threadgroup float tile[64];threadgroup bfloat hidden_tile[768];
threadgroup float norm_sums[32];threadgroup float norm_inv[1];
threadgroup float sdpa_outputs[1024],sdpa_max[32],sdpa_sum[32];

// Static host-style list: every group owns a fixed strided subset.
for(uint task=group;task<{FRONT_TASKS};task+=WORKERS) {{
 uint job=(task<64)?((task%2)?96+task/2:task/2):task-32;
 if(job<96)north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
 else north_sdpa_task(job-96,sg,lane,q,kc,vc,params,ax,sdpa_outputs,sdpa_max,sdpa_sum);
 if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{VISITS}+task,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup);
}}
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
if(tid==0){{
 atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
 while(atomic_load_explicit(state,memory_order_relaxed)<WORKERS){{}}
 atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
}}
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);

for(uint local=group;local<{DOWN_TASKS + 16};local+=WORKERS) {{
 uint task=(local<{DOWN_TASKS})?({FRONT_TASKS}+16+local):({FRONT_TASKS}+local-{DOWN_TASKS});
 if(local<{DOWN_TASKS}) {{
  uint slot=local/16,rowjob=local%16;
  for(uint j=tid;j<768;j+=256)hidden_tile[j]=hidden[slot*768+j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint phase=0;phase<4;phase++)
   north_complete_down(slot,rowjob*128+phase*32+sg*4,lane,hidden_tile,ids,down,ds,db,routed);
 }} else {{
  uint rowjob=local-{DOWN_TASKS};
  for(uint phase=0;phase<4;phase++)north_attention(rowjob*4+phase,sg,lane,ax,aw,attention);
 }}
 if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{VISITS}+task,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup);
}}
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
if(tid==0){{
 atomic_fetch_add_explicit(state+1,1u,memory_order_relaxed);
 while(atomic_load_explicit(state+1,memory_order_relaxed)<WORKERS){{}}
 atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
}}
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);

for(uint rowjob=group;rowjob<{JOIN_TASKS};rowjob+=WORKERS) {{
 for(uint row=rowjob*128+tid;row<(rowjob+1)*128;row+=256) {{
  float value=0;
  for(uint slot=0;slot<8;slot++){{volatile float product=float(routed[slot*2048+row])*scores[slot];value+=product;}}
  bfloat moe=bfloat(value);out[row]=bfloat(float(bfloat(float(attention[row])+float(moe)))+float(residual[row]));
 }}
 if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{VISITS + FRONT_TASKS + 16 + DOWN_TASKS}+rowjob,1u,memory_order_relaxed);
}}

if(DO_PREP) {{
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
 if(tid==0){{
  atomic_fetch_add_explicit(state+2,1u,memory_order_relaxed);
  while(atomic_load_explicit(state+2,memory_order_relaxed)<WORKERS){{}}
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
 }}
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);

 if(group==0) {{
  if(tid<32)norm_sums[tid]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc0=0,acc1=0;
  uint p0=tid*4,p1=(tid+256)*4;
  for(uint j=0;j<4;j++){{float value=float(out[p0+j]);acc0+=value*value;}}
  for(uint j=0;j<4;j++){{float value=float(out[p1+j]);acc1+=value*value;}}
  acc0=simd_sum(acc0);acc1=simd_sum(acc1);
  if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){{float total=simd_sum(norm_sums[lane]);if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint j=0;j<4;j++)norm[p0+j]=next_norm_w[p0+j]*bfloat(float(out[p0+j])*norm_inv[0]);
  for(uint j=0;j<4;j++)norm[p1+j]=next_norm_w[p1+j]*bfloat(float(out[p1+j])*norm_inv[0]);
  if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{VISITS + FRONT_TASKS + 16 + DOWN_TASKS + JOIN_TASKS},1u,memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_store_explicit(state+3,1u,memory_order_relaxed);
 }} else if(tid==0) {{
  while(atomic_load_explicit(state+3,memory_order_relaxed)==0){{}}
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
 }}
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);

 for(uint job=group;job<{PREP_TASKS};job+=WORKERS) {{
  north_next_prep(job,sg,lane,norm,next_qw,next_kw,next_vw,next_rw,prep,tile);
  if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{VISITS + FRONT_TASKS + 16 + DOWN_TASKS + JOIN_TASKS + 1}+job,1u,memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }}
}}
'''


kernel = mx.fast.metal_kernel(
    name="north_full_attention_static_layer",
    input_names=FULL_NAMES + ["scores", "residual"] + NEXT_NAMES,
    output_names=["workspace"],
    header=header,
    source=source,
)


def run_static_full_attention(
    mlp,
    x,
    q,
    key_cache,
    value_cache,
    params,
    ids,
    attention_weight,
    scores,
    residual,
    next_layer,
    workers=32,
    do_prep=True,
    return_state=False,
):
    validation = branch_inputs(mlp, x, mx.zeros((1, 1, 4096), mx.bfloat16), ids, attention_weight)
    kernel_inputs = [x, q, key_cache, value_cache, params, ids, *validation[3:], scores, residual]
    attn = next_layer.self_attn
    kernel_inputs += [
        next_layer.input_layernorm.weight,
        attn.q_proj.weight,
        attn.k_proj.weight,
        attn.v_proj.weight,
        next_layer.mlp.gate.weight,
    ]
    workspace = kernel(
        inputs=kernel_inputs,
        template=[("WORKERS", workers), ("DO_PREP", do_prep), ("AUDIT", return_state)],
        grid=(workers * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[(SIZE,)],
        output_dtypes=[mx.uint32],
        init_value=0,
    )[0]
    raw = workspace.view(mx.bfloat16)
    out = raw[OUT * 2 : NORM * 2].reshape(1, 1, 2048)
    if not do_prep:
        return (out, workspace[:STATE]) if return_state else out
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
    if return_state:
        return (
            *values,
            workspace[:STATE],
            raw[AX * 2 : ATTENTION * 2].reshape(1, 32, 1, 128),
            raw[HIDDEN * 2 : AX * 2].reshape(8, 768),
            raw[ATTENTION * 2 : ROUTED * 2].reshape(1, 1, 2048),
            raw[ROUTED * 2 : OUT * 2].reshape(8, 2048),
        )
    return values
