"""Persistent branch tail plus exact next-layer RMSNorm/QKV/router preparation."""

from pathlib import Path

import mlx.core as mx

from kernels import HEADER, NAMES


# Workspace offsets are uint32 elements. Tensor comments use bfloat16 elements.
STATE = 4096
HIDDEN = STATE                 # 8 * 768 bf16 = 3072 uint
ATTENTION = HIDDEN + 3072      # 2048 bf16 = 1024 uint
ROUTED = ATTENTION + 1024      # 8 * 2048 bf16 = 8192 uint
OUT = ROUTED + 8192            # 2048 bf16 = 1024 uint
NORM = OUT + 1024              # 2048 bf16 = 1024 uint
PREP = NORM + 1024             # (4096 + 512 + 512 + 128) bf16 = 2624 uint
SIZE = PREP + 2624

FRONT_TASKS = 112
DOWN_TASKS = 128
JOIN_TASKS = 16
PREP_TASKS = 47
PREFETCH_GROUPS = 48
TOTAL_TASKS = FRONT_TASKS + DOWN_TASKS + JOIN_TASKS + 1 + PREP_TASKS

DOWN_READY = 128
ATTN_READY = DOWN_READY + DOWN_TASKS
JOIN_CLAIM = ATTN_READY + JOIN_TASKS
JOIN_DONE = JOIN_CLAIM + JOIN_TASKS
NORM_CLAIM = JOIN_DONE + 1
NORM_READY = NORM_CLAIM + 1
PREP_CLAIM = NORM_READY + 1
PREFETCH_DONE = PREP_CLAIM + 1
EARLY_ROUNDS = PREFETCH_DONE + 1
ERROR = EARLY_ROUNDS + 1
VISITS = 1024

NEXT_NAMES = ["next_norm_w", "next_qw", "next_kw", "next_vw", "next_rw"]

header = "#define NCHUNK 64\n" + HEADER
header = header.replace(
    "device bfloat* hidden,",
    "coherent(device) device bfloat* hidden,",
)
header = header.replace(
    "device const bfloat* ax,device const bfloat* aw,device bfloat* attention)",
    "device const bfloat* ax,device const bfloat* aw,coherent(device) device bfloat* attention)",
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
inline void north_next_gemv(
 uint job,uint sg,uint lane,coherent(device) device const bfloat* x,
 device const bfloat* qw,device const bfloat* kw,
 device const bfloat* vw,device const bfloat* rw,
 coherent(device) device bfloat* prep,threadgroup float* scratch) {
 if(job>=39) {
  // MLX selects BM=1/BN=8 for the narrow 128-row router. Eight SIMD
  // groups split K, then group zero adds those partials in ascending order.
  uint block=(job-39)*4;
  for(uint phase=0;phase<4;phase++) {
   uint row=(block+phase)*4;float acc[4]={0};
   for(uint k=sg*128+lane*4;k<2048;k+=1024) {
    float v[4];for(uint j=0;j<4;j++)v[j]=float(x[k+j]);
    for(uint rr=0;rr<4;rr++)
     for(uint j=0;j<4;j++)acc[rr]+=float(rw[(row+rr)*2048+k+j])*v[j];
   }
   for(uint rr=0;rr<4;rr++) {
    for(ushort offset=16;offset>=1;offset>>=1)
     acc[rr]+=simd_shuffle_down(acc[rr],offset);
    if(lane==0)scratch[sg*8+rr]=acc[rr];
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if(sg==0 && lane==0)for(uint rr=0;rr<4;rr++) {
    float total=acc[rr];
    for(uint part=1;part<8;part++)total+=scratch[part*8+rr];
    prep[5120+row+rr]=bfloat(total);
   }
   threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  return;
 }
 device const bfloat* w;
 coherent(device) device bfloat* y;
 uint block;
 if(job<31){w=qw;y=prep;block=job;}
 else if(job<35){w=kw;y=prep+4096;block=job-31;}
 else {w=vw;y=prep+4608;block=job-35;}
 // Prefetched workers own Q rows 0..191. The remaining 122 Q blocks use
 // thirty full four-block tasks and one two-block tail task.
 uint phases=(job==30)?2:4;
 uint first_block=(job<31)?(6+block*4):(block*4);
 for(uint phase=0;phase<phases;phase++) {
  uint row=(first_block+phase)*32+sg*4;
  float acc[4]={0};
  for(uint k=lane*4;k<2048;k+=128) {
   float v[4];for(uint j=0;j<4;j++)v[j]=float(x[k+j]);
   for(uint rr=0;rr<4;rr++)
    for(uint j=0;j<4;j++)acc[rr]+=float(w[(row+rr)*2048+k+j])*v[j];
  }
  for(uint rr=0;rr<4;rr++) {
   for(ushort offset=16;offset>=1;offset>>=1)
    acc[rr]+=simd_shuffle_down(acc[rr],offset);
   if(lane==0)y[row+rr]=bfloat(acc[rr]);
  }
 }
}
inline void north_prefetched_q(
 uint group,uint sg,uint lane,coherent(device) device const bfloat* x,
 threadgroup const bfloat* w,coherent(device) device bfloat* prep) {
 if(sg>0)return;
 uint row=group*4;float acc[4]={0};
 for(uint k=lane*4;k<2048;k+=128) {
  float v[4];for(uint j=0;j<4;j++)v[j]=float(x[k+j]);
  for(uint rr=0;rr<4;rr++)
   for(uint j=0;j<4;j++)acc[rr]+=float(w[rr*2048+k+j])*v[j];
 }
 for(uint rr=0;rr<4;rr++) {
  for(ushort offset=16;offset>=1;offset>>=1)
   acc[rr]+=simd_shuffle_down(acc[rr],offset);
  if(lane==0)prep[row+rr]=bfloat(acc[rr]);
 }
}
'''

source = f'''
uint tid=thread_position_in_threadgroup.x;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint group=threadgroup_position_in_grid.x;
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
threadgroup bfloat prefetched_q[8192];
threadgroup uint task;
uint rotation=group%8,idle=0,ready_mask=0,prefetch_round=0;
bool all_hidden_seen=false,norm_seen=false;
bool prefetched_done=group>={PREFETCH_GROUPS};
while(true) {{
 if(tid==0) {{
  task=0xffffffffu;

  // A down tile needs only its own expert's hidden activation.
  for(uint ee=0;ee<8 && task==0xffffffffu;ee++) {{
   uint slot=(rotation+ee)%8;
   uint claimed=atomic_load_explicit(state+97+slot,memory_order_relaxed);
   if(claimed>=16)continue;
   bool ready=(ready_mask&(1u<<slot))!=0;
   if(!ready) {{
    ready=true;
    for(uint j=0;j<12;j++)ready=ready && atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed)==1;
    if(ready)ready_mask|=1u<<slot;
   }}
   if(ready) {{
    uint rowjob=atomic_fetch_add_explicit(state+97+slot,1u,memory_order_relaxed);
    if(rowjob<16)task={FRONT_TASKS}+slot*16+rowjob;
   }}
  }}

  // Join a row tile once attention and all eight expert results exist.
  for(uint rowjob=rotation;rowjob<16 && task==0xffffffffu;rowjob+=8) {{
   if(atomic_load_explicit(state+{JOIN_CLAIM}+rowjob,memory_order_relaxed))continue;
   bool ready=atomic_load_explicit(state+{ATTN_READY}+rowjob,memory_order_relaxed)==1;
   for(uint slot=0;slot<8;slot++)
    ready=ready && atomic_load_explicit(state+{DOWN_READY}+slot*16+rowjob,memory_order_relaxed)==1;
   if(ready) {{
    uint expected=0;
    if(atomic_compare_exchange_weak_explicit(state+{JOIN_CLAIM}+rowjob,&expected,1u,
       memory_order_relaxed,memory_order_relaxed))task={FRONT_TASKS + DOWN_TASKS}+rowjob;
   }}
  }}

  uint joined=atomic_load_explicit(state+{JOIN_DONE},memory_order_relaxed);
  if(task==0xffffffffu && joined==16) {{
   uint expected=0;
   if(atomic_compare_exchange_weak_explicit(state+{NORM_CLAIM},&expected,1u,
      memory_order_relaxed,memory_order_relaxed))task={FRONT_TASKS + DOWN_TASKS + JOIN_TASKS};
  }}

  if(!norm_seen)norm_seen=atomic_load_explicit(state+{NORM_READY},memory_order_relaxed)==1;
  if(task==0xffffffffu && norm_seen && !prefetched_done)task=0xfffffffdu;
  if(task==0xffffffffu && norm_seen) {{
   uint job=atomic_fetch_add_explicit(state+{PREP_CLAIM},1u,memory_order_relaxed);
   if(job<{PREP_TASKS})task={FRONT_TASKS + DOWN_TASKS + JOIN_TASKS + 1}+job;
  }}

  if(task==0xffffffffu) {{
   uint front=atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
   if(front<{FRONT_TASKS})task=front;
  }}
  if(task==0xffffffffu && atomic_load_explicit(state+{PREP_CLAIM},memory_order_relaxed)>={PREP_TASKS}
     && atomic_load_explicit(state+{PREFETCH_DONE},memory_order_relaxed)>={PREFETCH_GROUPS})
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

 if(task==0xfffffffdu) {{
  // The control reaches this point with zero rounds loaded; PREFETCH carries
  // up to four rounds across earlier tasks while the activation is unready.
  while(prefetch_round<4) {{
   uint base=prefetch_round*2048+tid*8;
   for(uint j=0;j<8;j++)prefetched_q[base+j]=next_qw[group*8192+base+j];
   prefetch_round++;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  north_prefetched_q(group,sg,lane,norm,prefetched_q,prep);
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0) {{
   prefetched_done=true;
   atomic_fetch_add_explicit(state+{PREFETCH_DONE},1u,memory_order_relaxed);
  }}
 }} else if(task<{FRONT_TASKS}) {{
  uint job=INTERLEAVE ? ((task<32)?((task%2)?96+task/2:task/2):task-16) : task;
  if(job<96) {{
   north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
   threadgroup_barrier(mem_flags::mem_device);
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(tid==0)atomic_store_explicit(state+1+job,1u,memory_order_relaxed);
  }} else {{
   uint rowjob=job-96;
   for(uint phase=0;phase<4;phase++)north_attention(rowjob*4+phase,sg,lane,ax,aw,attention);
   threadgroup_barrier(mem_flags::mem_device);
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(tid==0)atomic_store_explicit(state+{ATTN_READY}+rowjob,1u,memory_order_relaxed);
  }}
 }} else if(task<{FRONT_TASKS + DOWN_TASKS}) {{
  uint job=task-{FRONT_TASKS},slot=job/16,rowjob=job%16;
  if(tid==0) {{
   uint published=0;
   for(uint j=0;j<12;j++)published+=atomic_load_explicit(state+1+slot*12+j,memory_order_relaxed);
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(published!=12)atomic_fetch_add_explicit(state+{ERROR},1u,memory_order_relaxed);
  }}
  threadgroup_barrier(mem_flags::mem_device);
  for(uint j=tid;j<768;j+=256)hidden_tile[j]=hidden[slot*768+j];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint phase=0;phase<4;phase++)
   north_complete_down(slot,rowjob*128+phase*32+sg*4,lane,hidden_tile,ids,down,ds,db,routed);
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_store_explicit(state+{DOWN_READY}+job,1u,memory_order_relaxed);
 }} else if(task<{FRONT_TASKS + DOWN_TASKS + JOIN_TASKS}) {{
  uint rowjob=task-{FRONT_TASKS + DOWN_TASKS};
  if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  if(tid<128) {{
   uint row=rowjob*128+tid;float value=0;
   for(uint slot=0;slot<8;slot++) {{
    volatile float product=float(routed[slot*2048+row])*scores[slot];
    value+=product;
   }}
   bfloat moe=bfloat(value);
   out[row]=bfloat(float(bfloat(float(attention[row])+float(moe)))+float(residual[row]));
  }}
  threadgroup_barrier(mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0)atomic_fetch_add_explicit(state+{JOIN_DONE},1u,memory_order_relaxed);
 }} else if(task=={FRONT_TASKS + DOWN_TASKS + JOIN_TASKS}) {{
  if(tid<32)norm_sums[tid]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  float acc0=0,acc1=0;
  uint p0=tid*4,p1=(tid+256)*4;
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
  uint job=task-{FRONT_TASKS + DOWN_TASKS + JOIN_TASKS + 1};
  if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  threadgroup_barrier(mem_flags::mem_device);
  north_next_gemv(job,sg,lane,norm,next_qw,next_kw,next_vw,next_rw,prep,tile);
 }}
 if(task!=0xfffffffdu && group<{PREFETCH_GROUPS} && PREFETCH && prefetch_round<4
    && atomic_load_explicit(state+{NORM_READY},memory_order_relaxed)==0) {{
  uint base=prefetch_round*2048+tid*8;
  for(uint j=0;j<8;j++)prefetched_q[base+j]=next_qw[group*8192+base+j];
  prefetch_round++;
  if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{EARLY_ROUNDS},1u,memory_order_relaxed);
 }}
 if(task!=0xfffffffdu && tid==0 && AUDIT)atomic_fetch_add_explicit(state+{VISITS}+task,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 rotation=(rotation+1)%8;
}}
'''

kernel = mx.fast.metal_kernel(
    name="north_cross_layer_tail_prep",
    input_names=NAMES + ["scores", "residual"] + NEXT_NAMES,
    output_names=["workspace"],
    header=header,
    source=source,
)


def run_tail_prep(data, scores, residual, next_layer, workers=64, prefetch=False, return_state=False):
    assert workers >= PREFETCH_GROUPS
    attn = next_layer.self_attn
    inputs = data + [
        scores,
        residual,
        next_layer.input_layernorm.weight,
        attn.q_proj.weight,
        attn.k_proj.weight,
        attn.v_proj.weight,
        next_layer.mlp.gate.weight,
    ]
    workspace = kernel(
        inputs=inputs,
        template=[("INTERLEAVE", True), ("PREFETCH", prefetch), ("AUDIT", return_state)],
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
    prepared = (
        p[:4096].reshape(1, 1, 4096),
        p[4096:4608].reshape(1, 1, 512),
        p[4608:5120].reshape(1, 1, 512),
        p[5120:].reshape(1, 1, 128),
    )
    return (out, norm, *prepared, workspace[:STATE]) if return_state else (out, norm, *prepared)
