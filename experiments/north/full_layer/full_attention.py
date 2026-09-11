"""Persistent one-pass attention, MoE branch, join, and next-layer preparation."""

from pathlib import Path

import mlx.core as mx

from kernels import HEADER, NAMES, inputs as branch_inputs


STATE = 4096
HIDDEN = STATE
AX = HIDDEN + 3072
ATTENTION = AX + 2048
ROUTED = ATTENTION + 1024
OUT = ROUTED + 8192
NORM = OUT + 1024
PREP = NORM + 1024
SIZE = PREP + 2624

FRONT_TASKS = 128
OPROJ_TASKS = 16
DOWN_TASKS = 128
JOIN_TASKS = 16
PREP_TASKS = 48
TOTAL_TASKS = FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS + JOIN_TASKS + 1 + PREP_TASKS

HIDDEN_READY = 1
DOWN_CLAIM = 97
SDPA_DONE = 105
OPROJ_CLAIM = 106
ATTN_READY = 107
DOWN_READY = ATTN_READY + OPROJ_TASKS
JOIN_CLAIM = DOWN_READY + DOWN_TASKS
JOIN_DONE = JOIN_CLAIM + JOIN_TASKS
NORM_CLAIM = JOIN_DONE + 1
NORM_READY = NORM_CLAIM + 1
PREP_CLAIM = NORM_READY + 1
ERROR = PREP_CLAIM + 1
VISITS = 512

NEXT_NAMES = ["next_norm_w", "next_qw", "next_kw", "next_vw", "next_rw"]
FULL_NAMES = ["x", "q", "kc", "vc", "params", "ids"] + NAMES[3:]

header = "#define NCHUNK 64\n" + HEADER
header = header.replace("device bfloat* hidden,", "coherent(device) device bfloat* hidden,")
header = header.replace(
    "device const bfloat* ax,device const bfloat* aw,device bfloat* attention)",
    "coherent(device) device const bfloat* ax,device const bfloat* aw,coherent(device) device bfloat* attention)",
)
header += "\n" + Path(__file__).resolve().parents[1].joinpath("persistent/down.h").read_text()
header = header.replace(
    "coherent(device) device const bfloat* hidden,device const uint* ids,",
    "threadgroup const bfloat* hidden,device const uint* ids,",
).replace(
    "coherent(device) device const bfloat* xp=hidden+slot*768+k;",
    "threadgroup const bfloat* xp=hidden+k;",
).replace(
    "device bfloat* routed) {",
    "coherent(device) device bfloat* routed) {",
)
header += r'''
inline void north_sdpa_task(
 uint head,uint sg,uint lane,device const bfloat* q,
 device const bfloat* kc,device const bfloat* vc,constant const uint* params,
 coherent(device) device bfloat* out,threadgroup float* outputs,
 threadgroup float* max_scores,threadgroup float* sum_scores) {
 uint n=params[0],kh_stride=params[1],vh_stride=params[2],kv_head=head/8;
 device const bfloat* qp=q+head*128+lane*4;
 float qv[4],kv[4],ov[4][4];
 for(uint j=0;j<4;j++)qv[j]=0.08838834764831845f*float(qp[j]);
 for(uint part=0;part<4;part++)for(uint j=0;j<4;j++)ov[part][j]=0;
 for(uint part=0;part<4;part++) {
  uint virtual_sg=sg+part*8;
  device const bfloat* kp=kc+kv_head*kh_stride+virtual_sg*128+lane*4;
  device const bfloat* vp=vc+kv_head*vh_stride+virtual_sg*128+lane*4;
  float max_score=-3.402823466e+38f,sum_exp=0;
  for(uint i=virtual_sg;i<n;i+=32) {
   for(uint j=0;j<4;j++)kv[j]=float(kp[j]);
   float score=0;for(uint j=0;j<4;j++)score+=qv[j]*kv[j];
   score=simd_sum(score);
   float next_max=max(max_score,score);
   float factor=fast::exp(max_score-next_max),exp_score=fast::exp(score-next_max);
   max_score=next_max;sum_exp=sum_exp*factor+exp_score;
   for(uint j=0;j<4;j++)ov[part][j]=ov[part][j]*factor+exp_score*float(vp[j]);
   kp+=4096;vp+=4096;
  }
  if(lane==0){max_scores[virtual_sg]=max_score;sum_scores[virtual_sg]=sum_exp;}
 }
 threadgroup_barrier(mem_flags::mem_threadgroup);
 float max_score=max_scores[lane];float next_max=simd_max(max_score);
 float factor=fast::exp(max_score-next_max);
 float sum_exp=simd_sum(sum_scores[lane]*factor);
 for(uint j=0;j<4;j++) {
  for(uint part=0;part<4;part++)outputs[lane*32+sg+part*8]=ov[part][j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint part=0;part<4;part++) {
   uint virtual_sg=sg+part*8;
   float value=simd_sum(outputs[virtual_sg*32+lane]*factor);
   value=sum_exp==0?value:(value/sum_exp);
   if(lane==0)out[head*128+virtual_sg*4+j]=bfloat(value);
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }
}

inline void north_next_prep(
 uint job,uint sg,uint lane,coherent(device) device const bfloat* x,
 device const bfloat* qw,device const bfloat* kw,
 device const bfloat* vw,device const bfloat* rw,
 coherent(device) device bfloat* prep,threadgroup float* scratch) {
 if(job>=40) {
  uint block=(job-40)*4;
  for(uint phase=0;phase<4;phase++) {
   uint row=(block+phase)*4;float acc[4]={0};
   if(sg<8)for(uint k=sg*128+lane*4;k<2048;k+=1024) {
    float v[4];for(uint j=0;j<4;j++)v[j]=float(x[k+j]);
    for(uint rr=0;rr<4;rr++)
     for(uint j=0;j<4;j++)acc[rr]+=float(rw[(row+rr)*2048+k+j])*v[j];
   }
   if(sg<8)for(uint rr=0;rr<4;rr++) {
    for(ushort offset=16;offset>=1;offset>>=1)acc[rr]+=simd_shuffle_down(acc[rr],offset);
    if(lane==0)scratch[sg*8+rr]=acc[rr];
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if(sg==0 && lane==0)for(uint rr=0;rr<4;rr++) {
    float total=acc[rr];for(uint part=1;part<8;part++)total+=scratch[part*8+rr];
    prep[5120+row+rr]=bfloat(total);
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return;
 }
 if(sg>=8)return;
 device const bfloat* w;coherent(device) device bfloat* y;uint block;
 if(job<32){w=qw;y=prep;block=job;}
 else if(job<36){w=kw;y=prep+4096;block=job-32;}
 else {w=vw;y=prep+4608;block=job-36;}
 for(uint phase=0;phase<4;phase++) {
  uint row=(block*4+phase)*32+sg*4;float acc[4]={0};
  for(uint k=lane*4;k<2048;k+=128) {
   float v[4];for(uint j=0;j<4;j++)v[j]=float(x[k+j]);
   for(uint rr=0;rr<4;rr++)for(uint j=0;j<4;j++)acc[rr]+=float(w[(row+rr)*2048+k+j])*v[j];
  }
  for(uint rr=0;rr<4;rr++) {
   for(ushort offset=16;offset>=1;offset>>=1)acc[rr]+=simd_shuffle_down(acc[rr],offset);
   if(lane==0)y[row+rr]=bfloat(acc[rr]);
  }
 }
}
'''

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
threadgroup uint task;
uint rotation=group%8,idle=0,ready_mask=0;bool sdpa_seen=false,norm_seen=false;
while(true) {{
 if(tid==0) {{
  task=0xffffffffu;
  if(FRONT_FIRST) {{
   uint front=atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
   if(front<{FRONT_TASKS})task=front;
  }}
  for(uint ee=0;ee<8 && task==0xffffffffu;ee++) {{
   uint slot=(rotation+ee)%8,claimed=atomic_load_explicit(state+{DOWN_CLAIM}+slot,memory_order_relaxed);
   if(claimed>=16)continue;
   bool ready=(ready_mask&(1u<<slot))!=0;
   if(!ready) {{
    ready=true;for(uint j=0;j<12;j++)ready=ready&&atomic_load_explicit(state+{HIDDEN_READY}+slot*12+j,memory_order_relaxed)==1;
    if(ready)ready_mask|=1u<<slot;
   }}
   if(ready) {{
    uint rowjob=atomic_fetch_add_explicit(state+{DOWN_CLAIM}+slot,1u,memory_order_relaxed);
    if(rowjob<16)task={FRONT_TASKS + OPROJ_TASKS}+slot*16+rowjob;
   }}
  }}
  if(!sdpa_seen)sdpa_seen=atomic_load_explicit(state+{SDPA_DONE},memory_order_relaxed)==32;
  if(task==0xffffffffu && sdpa_seen) {{
   uint job=atomic_fetch_add_explicit(state+{OPROJ_CLAIM},1u,memory_order_relaxed);
   if(job<{OPROJ_TASKS})task={FRONT_TASKS}+job;
  }}
  for(uint rowjob=rotation;rowjob<16 && task==0xffffffffu;rowjob+=8) {{
   if(atomic_load_explicit(state+{JOIN_CLAIM}+rowjob,memory_order_relaxed))continue;
   bool ready=atomic_load_explicit(state+{ATTN_READY}+rowjob,memory_order_relaxed)==1;
   for(uint slot=0;slot<8;slot++)ready=ready&&atomic_load_explicit(state+{DOWN_READY}+slot*16+rowjob,memory_order_relaxed)==1;
   if(ready) {{
    uint expected=0;
    if(atomic_compare_exchange_weak_explicit(state+{JOIN_CLAIM}+rowjob,&expected,1u,memory_order_relaxed,memory_order_relaxed))
     task={FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS}+rowjob;
   }}
  }}
  uint joined=atomic_load_explicit(state+{JOIN_DONE},memory_order_relaxed);
  if(task==0xffffffffu && DO_PREP && joined==16) {{
   uint expected=0;
   if(atomic_compare_exchange_weak_explicit(state+{NORM_CLAIM},&expected,1u,memory_order_relaxed,memory_order_relaxed))
    task={FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS + JOIN_TASKS};
  }}
  if(DO_PREP && !norm_seen)norm_seen=atomic_load_explicit(state+{NORM_READY},memory_order_relaxed)==1;
  if(task==0xffffffffu && DO_PREP && norm_seen) {{
   uint job=atomic_fetch_add_explicit(state+{PREP_CLAIM},1u,memory_order_relaxed);
   if(job<{PREP_TASKS})task={FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS + JOIN_TASKS + 1}+job;
  }}
  if(task==0xffffffffu && !FRONT_FIRST) {{
   uint front=atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
   if(front<{FRONT_TASKS})task=front;
  }}
  if(task==0xffffffffu && ((DO_PREP && atomic_load_explicit(state+{PREP_CLAIM},memory_order_relaxed)>={PREP_TASKS})
     || (!DO_PREP && joined==16)))task=0xfffffffeu;
 }}
 threadgroup_barrier(mem_flags::mem_threadgroup);
 if(task==0xfffffffeu)break;
 if(task==0xffffffffu) {{
  if(++idle>=1000000){{if(tid==0)atomic_fetch_add_explicit(state+{ERROR},1u,memory_order_relaxed);break;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);continue;
 }}
 idle=0;
 if(task<{FRONT_TASKS}) {{
  uint job=(task<64)?((task%2)?96+task/2:task/2):task-32;
  if(job<96) {{
   north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
   threadgroup_barrier(mem_flags::mem_device);
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(tid==0)atomic_store_explicit(state+{HIDDEN_READY}+job,1u,memory_order_relaxed);
  }} else {{
   north_sdpa_task(job-96,sg,lane,q,kc,vc,params,ax,sdpa_outputs,sdpa_max,sdpa_sum);
   threadgroup_barrier(mem_flags::mem_device);
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(tid==0)atomic_fetch_add_explicit(state+{SDPA_DONE},1u,memory_order_relaxed);
  }}
 }} else if(task<{FRONT_TASKS + OPROJ_TASKS}) {{
  uint rowjob=task-{FRONT_TASKS};
  if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if(sg<8)for(uint phase=0;phase<4;phase++)north_attention(rowjob*4+phase,sg,lane,ax,aw,attention);
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_store_explicit(state+{ATTN_READY}+rowjob,1u,memory_order_relaxed);
 }} else if(task<{FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS}) {{
  uint job=task-{FRONT_TASKS + OPROJ_TASKS},slot=job/16,rowjob=job%16;
  if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  for(uint j=tid;j<768;j+=256)hidden_tile[j]=hidden[slot*768+j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg<8)for(uint phase=0;phase<4;phase++)
   north_complete_down(slot,rowjob*128+phase*32+sg*4,lane,hidden_tile,ids,down,ds,db,routed);
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_store_explicit(state+{DOWN_READY}+job,1u,memory_order_relaxed);
 }} else if(task<{FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS + JOIN_TASKS}) {{
  uint rowjob=task-{FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS};
  if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if(tid<128) {{
   uint row=rowjob*128+tid;float value=0;
   for(uint slot=0;slot<8;slot++){{volatile float product=float(routed[slot*2048+row])*scores[slot];value+=product;}}
   bfloat moe=bfloat(value);out[row]=bfloat(float(bfloat(float(attention[row])+float(moe)))+float(residual[row]));
  }}
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_fetch_add_explicit(state+{JOIN_DONE},1u,memory_order_relaxed);
 }} else if(task=={FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS + JOIN_TASKS}) {{
  if(tid<32)norm_sums[tid]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  float acc0=0,acc1=0;
  if(tid<256) {{
   uint p0=tid*4,p1=(tid+256)*4;
   for(uint j=0;j<4;j++){{float value=float(out[p0+j]);acc0+=value*value;}}
   for(uint j=0;j<4;j++){{float value=float(out[p1+j]);acc1+=value*value;}}
  }}
  if(tid<256) {{acc0=simd_sum(acc0);acc1=simd_sum(acc1);if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){{float total=simd_sum(norm_sums[lane]);if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid<256) {{
   uint p0=tid*4,p1=(tid+256)*4;
   for(uint j=0;j<4;j++)norm[p0+j]=next_norm_w[p0+j]*bfloat(float(out[p0+j])*norm_inv[0]);
   for(uint j=0;j<4;j++)norm[p1+j]=next_norm_w[p1+j]*bfloat(float(out[p1+j])*norm_inv[0]);
  }}
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_store_explicit(state+{NORM_READY},1u,memory_order_relaxed);
 }} else {{
  uint job=task-{FRONT_TASKS + OPROJ_TASKS + DOWN_TASKS + JOIN_TASKS + 1};
  if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  north_next_prep(job,sg,lane,norm,next_qw,next_kw,next_vw,next_rw,prep,tile);
 }}
 if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{VISITS}+task,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);rotation=(rotation+1)%8;
}}
'''

kernel = mx.fast.metal_kernel(
    name="north_full_attention_layer",
    input_names=FULL_NAMES + ["scores", "residual"] + NEXT_NAMES,
    output_names=["workspace"],
    header=header,
    source=source,
)


def run_full_attention(mlp, x, q, key_cache, value_cache, params, ids, attention_weight,
                       scores, residual, next_layer, workers=64, return_state=False,
                       front_first=False, do_prep=True):
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
        template=[("AUDIT", return_state), ("FRONT_FIRST", front_first), ("DO_PREP", do_prep)],
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
        attention_debug = raw[AX * 2 : ATTENTION * 2].reshape(1, 32, 1, 128)
        hidden_debug = raw[HIDDEN * 2 : AX * 2].reshape(8, 768)
        projected_debug = raw[ATTENTION * 2 : ROUTED * 2].reshape(1, 1, 2048)
        routed_debug = raw[ROUTED * 2 : OUT * 2].reshape(8, 2048)
        return (
            *values,
            workspace[:STATE],
            attention_debug,
            hidden_debug,
            projected_debug,
            routed_debug,
        )
    return values
