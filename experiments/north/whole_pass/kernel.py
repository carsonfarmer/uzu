"""Packed multi-layer North decode path in one persistent Metal dispatch."""

import mlx.core as mx

from full_layer.full_attention import header as layer_header
from whole_pass.pack import HEAD_NAMES, PREFIX_NAMES, WEIGHT_NAMES
from whole_pass.primitives import sigmoid_lut


STATE = 4096
X = STATE
NORM = X + 1024
PREP = NORM + 1024
AX = PREP + 2624
HIDDEN = AX + 2048
ATTENTION = HIDDEN + 3072
ROUTED = ATTENTION + 1024
IDS = ROUTED + 8192
SCORES = IDS + 8
PREFIX_DEBUG = SCORES + 8
PREFIX_ATTN_DEBUG = PREFIX_DEBUG + 1024
PREFIX_MLP_DEBUG = PREFIX_ATTN_DEBUG + 1024
PREFIX_HIDDEN_DEBUG = PREFIX_MLP_DEBUG + 1024
SIZE = PREFIX_HIDDEN_DEBUG + 1536

FRONT_TASKS = 128
SECOND_TASKS = 144
JOIN_TASKS = 16
PREP_TASKS = 48

header = layer_header.replace(
    " device const bfloat* x,device const uint* ids,",
    " coherent(device) device const bfloat* x,device const uint* ids,",
) + r'''
inline void north_grid_barrier(
 device atomic_uint* state,uint counter,uint tid,uint workers) {
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
 atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
 if(tid==0) {
  atomic_fetch_add_explicit(state+counter,1u,memory_order_relaxed);
  while(atomic_load_explicit(state+counter,memory_order_relaxed)<workers) {}
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
 }
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
}

inline uint north_queue_claim(
 device atomic_uint* state,uint stage,uint tasks) {
 uint base=stage*4096u;
 while(true) {
  uint cursor=atomic_load_explicit(state+1,memory_order_relaxed);
  if(cursor<base) {
   atomic_compare_exchange_weak_explicit(
    state+1,&cursor,base,memory_order_relaxed,memory_order_relaxed);
   continue;
  }
  if(cursor>=base+tasks)return 0xffffffffu;
  uint next=cursor+1;
  if(atomic_compare_exchange_weak_explicit(
      state+1,&cursor,next,memory_order_relaxed,memory_order_relaxed))
   return cursor-base;
 }
}

inline void north_queue_finish(
 device atomic_uint* state,uint stage,uint tasks) {
 atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
 uint base=stage*4096u;
 while(true) {
  uint done=atomic_load_explicit(state+2,memory_order_relaxed);
  if(done<base) {
   atomic_compare_exchange_weak_explicit(
    state+2,&done,base,memory_order_relaxed,memory_order_relaxed);
   continue;
  }
  uint next=done+1;
  if(atomic_compare_exchange_weak_explicit(
      state+2,&done,next,memory_order_relaxed,memory_order_relaxed)) {
   if(next-base==tasks) {
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    atomic_store_explicit(state,stage+1,memory_order_relaxed);
   }
   return;
  }
 }
}

inline void north_sdpa_packed(
 uint head,uint sg,uint lane,uint n,uint head_stride,
 coherent(device) device const bfloat* q,device const bfloat* kc,device const bfloat* vc,
 coherent(device) device bfloat* out,threadgroup float* outputs,
 threadgroup float* max_scores,threadgroup float* sum_scores) {
 uint kv_head=head/8;
 coherent(device) device const bfloat* qp=q+head*128+lane*4;
 float qv[4],kv[4],ov[4][4];
 for(uint j=0;j<4;j++)qv[j]=0.08838834764831845f*float(qp[j]);
 for(uint part=0;part<4;part++)for(uint j=0;j<4;j++)ov[part][j]=0;
 for(uint part=0;part<4;part++) {
  uint virtual_sg=sg+part*8;
  device const bfloat* kp=kc+kv_head*head_stride+virtual_sg*128+lane*4;
  device const bfloat* vp=vc+kv_head*head_stride+virtual_sg*128+lane*4;
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

inline void north_next_prep_tiled(
 uint job,uint rows,uint router_rows,uint sg,uint lane,coherent(device) device const bfloat* x,
 device const bfloat* qw,device const bfloat* kw,
 device const bfloat* vw,device const bfloat* rw,
 coherent(device) device bfloat* prep,threadgroup float* scratch) {
 uint qjobs=4096/rows,kvjobs=512/rows,dense_jobs=qjobs+2*kvjobs;
 if(job>=dense_jobs) {
  uint router_phases=router_rows/4,block=(job-dense_jobs)*router_phases;
  for(uint phase=0;phase<router_phases;phase++) {
   uint row=(block+phase)*4;float acc[4]={0};
   if(sg<8)for(uint k=sg*128+lane*4;k<2048;k+=1024) {
    float v[4];for(uint j=0;j<4;j++)v[j]=float(x[k+j]);
    for(uint rr=0;rr<4;rr++)for(uint j=0;j<4;j++)acc[rr]+=float(rw[(row+rr)*2048+k+j])*v[j];
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
 if(job<qjobs){w=qw;y=prep;block=job;}
 else if(job<qjobs+kvjobs){w=kw;y=prep+4096;block=job-qjobs;}
 else {w=vw;y=prep+4608;block=job-qjobs-kvjobs;}
 uint phases=rows/32;
 for(uint phase=0;phase<phases;phase++) {
  uint row=(block*phases+phase)*32+sg*4;float acc[4]={0};
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

inline void north_lm_head(
 uint job,uint rows,uint sg,uint lane,coherent(device) device const bfloat* x,
 device const uint* lm_w,device const bfloat* lm_s,device const bfloat* lm_b,
 device bfloat* logits) {
 uint first=job*rows;
 for(uint rr=sg*4;rr<rows;rr+=32) {
  uint row=first+rr;float acc[4]={0};
  for(uint k=lane*16;k<2048;k+=512) {
   float v[16],xsum=0;
   for(uint j=0;j<16;j+=4) {
    xsum+=float(x[k+j]+x[k+j+1]+x[k+j+2]+x[k+j+3]);
    v[j]=float(x[k+j]);v[j+1]=float(x[k+j+1])/16.0f;
    v[j+2]=float(x[k+j+2])/256.0f;v[j+3]=float(x[k+j+3])/4096.0f;
   }
   for(uint rowoff=0;rowoff<4;rowoff++) {
    uint r=row+rowoff,si=r*32+k/64;
    device const ushort* w=(device const ushort*)(lm_w+r*256+k/8);
    float dot=0;
    for(uint j=0;j<4;j++)dot+=(v[j*4]*(w[j]&15)+v[j*4+1]*(w[j]&240)+v[j*4+2]*(w[j]&3840)+v[j*4+3]*(w[j]&61440));
    acc[rowoff]+=float(lm_s[si])*dot+float(lm_b[si])*xsum;
   }
  }
  for(uint rowoff=0;rowoff<4;rowoff++) {
   float value=simd_sum(acc[rowoff]);
   if(lane==0)logits[row+rowoff]=bfloat(value);
  }
 }
}

inline void north_prefix_hidden(
 uint job,uint sg,uint lane,coherent(device) device const bfloat* x,
 device const uint* up,device const bfloat* us,device const bfloat* ub,
 device const uint* gate,device const bfloat* gs,device const bfloat* gb,
 coherent(device) device bfloat* hidden) {
 uint first=job*128;
 for(uint rr=sg*4;rr<128;rr+=32) {
  uint row=first+rr;float uu[4]={0},gg[4]={0};
  for(uint k=lane*16;k<2048;k+=512) {
   float v[16],xsum=0;
   for(uint j=0;j<16;j+=4) {
    xsum+=float(x[k+j]+x[k+j+1]+x[k+j+2]+x[k+j+3]);
    v[j]=float(x[k+j]);v[j+1]=float(x[k+j+1])/16.0f;
    v[j+2]=float(x[k+j+2])/256.0f;v[j+3]=float(x[k+j+3])/4096.0f;
   }
   for(uint rowoff=0;rowoff<4;rowoff++) {
    uint r=row+rowoff,si=r*32+k/64;
    device const ushort* uw=(device const ushort*)(up+r*256+k/8);
    device const ushort* gw=(device const ushort*)(gate+r*256+k/8);
    float ud=0,gd=0;
    for(uint j=0;j<4;j++) {
     ud+=(v[j*4]*(uw[j]&15)+v[j*4+1]*(uw[j]&240)+v[j*4+2]*(uw[j]&3840)+v[j*4+3]*(uw[j]&61440));
     gd+=(v[j*4]*(gw[j]&15)+v[j*4+1]*(gw[j]&240)+v[j*4+2]*(gw[j]&3840)+v[j*4+3]*(gw[j]&61440));
    }
    uu[rowoff]+=float(us[si])*ud+float(ub[si])*xsum;
    gg[rowoff]+=float(gs[si])*gd+float(gb[si])*xsum;
   }
  }
  for(uint rowoff=0;rowoff<4;rowoff++) {
   float u=simd_sum(uu[rowoff]),g=simd_sum(gg[rowoff]);
   if(lane==0) {
    bfloat u16=bfloat(u),g16=bfloat(g);auto z=1/(1+exp(abs(g16)));
    bfloat sigmoid16=(g16<0)?z:1-z;
    hidden[row+rowoff]=bfloat(float(u16)*float(g16*sigmoid16));
   }
  }
 }
}

inline void north_prefix_down(
 uint job,uint sg,uint lane,coherent(device) device const bfloat* hidden,
 device const uint* down,device const bfloat* ds,device const bfloat* db,
 coherent(device) device bfloat* out) {
 uint first=job*64;
 for(uint phase=0;phase<2;phase++) {
  uint row=first+phase*32+sg*4;float acc[4]={0};
  for(uint k=lane*16;k<3072;k+=512) {
   coherent(device) device const bfloat* xp=hidden+k;
   float v[16],xsum=0;
   for(uint j=0;j<16;j+=4) {
    xsum+=float(xp[j]+xp[j+1]+xp[j+2]+xp[j+3]);
    v[j]=float(xp[j]);v[j+1]=float(xp[j+1])/16.0f;
    v[j+2]=float(xp[j+2])/256.0f;v[j+3]=float(xp[j+3])/4096.0f;
   }
   for(uint rr=0;rr<4;rr++) {
    uint r=row+rr,si=r*48+k/64;
    device const ushort* w=(device const ushort*)(down+r*384+k/8);
    float dot=0;
    for(uint j=0;j<4;j++)dot+=(v[j*4]*(w[j]&15)+v[j*4+1]*(w[j]&240)+v[j*4+2]*(w[j]&3840)+v[j*4+3]*(w[j]&61440));
    acc[rr]+=float(ds[si])*dot+float(db[si])*xsum;
   }
  }
  for(uint rr=0;rr<4;rr++){{float value=simd_sum(acc[rr]);if(lane==0)out[row+rr]=bfloat(value);}}
 }
}
'''

source = f'''
uint tid=thread_position_in_threadgroup.x;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint group=threadgroup_position_in_grid.x;
uint global_tid=group*256+tid,global_stride=WORKERS*256;
device atomic_uint* state=(device atomic_uint*)workspace;
coherent(device) device bfloat* x=(coherent(device) device bfloat*)(workspace+{X});
coherent(device) device bfloat* norm=(coherent(device) device bfloat*)(workspace+{NORM});
coherent(device) device bfloat* prep=(coherent(device) device bfloat*)(workspace+{PREP});
coherent(device) device bfloat* ax=(coherent(device) device bfloat*)(workspace+{AX});
coherent(device) device bfloat* hidden=(coherent(device) device bfloat*)(workspace+{HIDDEN});
coherent(device) device bfloat* attention=(coherent(device) device bfloat*)(workspace+{ATTENTION});
coherent(device) device bfloat* routed=(coherent(device) device bfloat*)(workspace+{ROUTED});
coherent(device) device bfloat* prefix_debug=(coherent(device) device bfloat*)(workspace+{PREFIX_DEBUG});
coherent(device) device bfloat* prefix_attn_debug=(coherent(device) device bfloat*)(workspace+{PREFIX_ATTN_DEBUG});
coherent(device) device bfloat* prefix_mlp_debug=(coherent(device) device bfloat*)(workspace+{PREFIX_MLP_DEBUG});
coherent(device) device bfloat* prefix_hidden_debug=(coherent(device) device bfloat*)(workspace+{PREFIX_HIDDEN_DEBUG});
device uint* ids=workspace+{IDS};device float* scores=(device float*)(workspace+{SCORES});
uint position=params[0],capacity=params[1],length=position+1,head_stride=capacity*128;
threadgroup float tile[64];threadgroup bfloat hidden_tile[768];
threadgroup float norm_sums[32];threadgroup float norm_inv[1];
threadgroup float sdpa_outputs[1024],sdpa_max[32],sdpa_sum[32];
threadgroup uint dynamic_task,queue_stage,queue_task,queue_tasks;

if(SAFE_QUEUE) {{
 constexpr uint init_tasks=32;
 constexpr uint prefix_post_tasks=38;
 constexpr uint prep_tasks=5120/PREP_ROWS+128/ROUTER_ROWS;
 constexpr uint second_tasks=128+2048/OPROJ_ROWS;
 constexpr uint moe_start=1+DO_PREFIX*6;
 constexpr uint head_start=moe_start+NLAYERS*6;
 constexpr uint total_stages=head_start+DO_HEAD*2;
 while(true) {{
  if(tid==0) {{
   queue_stage=atomic_load_explicit(state,memory_order_relaxed);
   uint phase=(queue_stage>=moe_start && queue_stage<head_start)?((queue_stage-moe_start)%6):0;
   queue_tasks=0;
   if(queue_stage==0)queue_tasks=init_tasks;
   else if(DO_PREFIX && queue_stage<moe_start) {{
    uint p=queue_stage-1;
    queue_tasks=(p==0)?1:((p==1)?64:((p==2)?prefix_post_tasks:((p==3)?64:((p==4)?64:16))));
   }} else if(queue_stage<head_start) {{
    if(phase==0)queue_tasks=1;
    else if(phase==1)queue_tasks=prep_tasks;
    else if(phase==2)queue_tasks=39;
    else if(phase==3)queue_tasks={FRONT_TASKS};
    else if(phase==4)queue_tasks=second_tasks;
    else {{
     uint layer=(queue_stage-moe_start)/6;
     queue_tasks=16+((PREFETCH_STAGES>0 && layer+1<NLAYERS)?prep_tasks:0);
    }}
   }} else if(DO_HEAD && queue_stage<total_stages)
    queue_tasks=(queue_stage==head_start)?1:(262144/LM_ROWS);
   queue_task=(queue_stage<total_stages)?north_queue_claim(state,queue_stage,queue_tasks):0xffffffffu;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(queue_stage>=total_stages)break;
  if(queue_task!=0xffffffffu) {{
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
   if(queue_stage==0) {{
    ulong cache_prefix=ulong(NLAYERS+DO_PREFIX)*4ul*ulong(position)*128ul;
    for(ulong logical=ulong(queue_task)*256ul+tid;logical<cache_prefix;logical+=ulong(init_tasks)*256ul) {{
     uint dim=logical%128ul;ulong z=logical/128ul;
     uint pos=z%position;z/=position;uint head=z%4ul;uint layer=z/4ul;
     ulong target=((ulong(layer)*4ul+head)*capacity+pos)*128ul+dim;
     key_out[target]=key_cache[target];value_out[target]=value_cache[target];
    }}
    if(queue_task==0)for(uint i=tid;i<2048;i+=256)x[i]=x_in[i];
   }} else if(DO_PREFIX && queue_stage<moe_start) {{
    uint p=queue_stage-1;
    device const bfloat* pnorm=prefix_dense;
    device const bfloat* pqw=prefix_dense+2048;
    device const bfloat* pkw=pqw+8388608;
    device const bfloat* pvw=pkw+1048576;
    device const bfloat* paw=pvw+1048576;
    device const uint* pup=prefix_q;
    device const uint* pgate=prefix_q+786432;
    device const uint* pdown=prefix_q+1572864;
    device const bfloat* pus=prefix_meta;
    device const bfloat* pub=prefix_meta+98304;
    device const bfloat* pgs=prefix_meta+196608;
    device const bfloat* pgb=prefix_meta+294912;
    device const bfloat* pds=prefix_meta+393216;
    device const bfloat* pdb=prefix_meta+491520;
    if(p==0) {{
     if(tid<32)norm_sums[tid]=0;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     float acc0=0,acc1=0;uint p0=tid*4,p1=(tid+256)*4;
     for(uint j=0;j<4;j++){{float value=float(x[p0+j]);acc0+=value*value;}}
     for(uint j=0;j<4;j++){{float value=float(x[p1+j]);acc1+=value*value;}}
     acc0=simd_sum(acc0);acc1=simd_sum(acc1);
     if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if(sg==0){{float total=simd_sum(norm_sums[lane]);if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);}}
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for(uint j=0;j<4;j++)norm[p0+j]=pnorm[p0+j]*bfloat(float(x[p0+j])*norm_inv[0]);
     for(uint j=0;j<4;j++)norm[p1+j]=pnorm[p1+j]*bfloat(float(x[p1+j])*norm_inv[0]);
    }} else if(p==1) {{
     if(queue_task<40)north_next_prep_tiled(queue_task,128,16,sg,lane,norm,pqw,pkw,pvw,pqw,prep,tile);
     else {{
      uint block=queue_task-40;
      north_prefix_hidden(block,sg,lane,norm,pup,pus,pub,pgate,pgs,pgb,hidden);
      threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
      for(uint i=tid;i<128;i+=256)prefix_hidden_debug[block*128+i]=hidden[block*128+i];
     }}
    }} else if(p==2) {{
     if(queue_task<32)for(uint item=queue_task*64+tid;item<(queue_task+1)*64;item+=256) {{
      uint pair=item%64,head=item/64,index=head*128+pair*2;
      float x1=float(prep[index]),x2=float(prep[index+1]);
      float inv_freq=metal::exp2(-(float(pair)/64.0f)*15.609640474436812f);
      float theta=float(position)*inv_freq,c=metal::fast::cos(theta),s=metal::fast::sin(theta);
      prep[index]=bfloat(x1*c-x2*s);prep[index+1]=bfloat(x1*s+x2*c);
     }} else if(queue_task<36)for(uint item=(queue_task-32)*64+tid;item<(queue_task-31)*64;item+=256) {{
      uint pair=item%64,head=item/64,index=head*128+pair*2;
      float x1=float(prep[4096+index]),x2=float(prep[4096+index+1]);
      float inv_freq=metal::exp2(-(float(pair)/64.0f)*15.609640474436812f);
      float theta=float(position)*inv_freq,c=metal::fast::cos(theta),s=metal::fast::sin(theta);
      ulong target=(ulong(head)*capacity+position)*128ul+pair*2;
      key_out[target]=bfloat(x1*c-x2*s);key_out[target+1]=bfloat(x1*s+x2*c);
     }} else for(uint item=(queue_task-36)*256+tid;item<(queue_task-35)*256;item+=256) {{
      uint head=item/128,dim=item%128;
      value_out[(ulong(head)*capacity+position)*128ul+dim]=prep[4608+item];
     }}
    }} else if(p==3) {{
     if(queue_task%2==0)north_sdpa_packed(queue_task/2,sg,lane,length,head_stride,prep,key_out,value_out,ax,sdpa_outputs,sdpa_max,sdpa_sum);
     else north_prefix_down(queue_task/2,sg,lane,hidden,pdown,pds,pdb,routed);
    }} else if(p==4)north_attention(queue_task,sg,lane,ax,paw,attention);
    else {{
     uint first=queue_task*128;
     for(uint row=first+tid;row<first+128;row+=256) {{
      x[row]=bfloat(float(bfloat(float(attention[row])+float(routed[row])))+float(x[row]));
      prefix_debug[row]=x[row];prefix_attn_debug[row]=attention[row];prefix_mlp_debug[row]=routed[row];
     }}
    }}
   }} else if(queue_stage<head_start) {{
    uint relative=queue_stage-moe_start,layer=relative/6,phase=relative%6;
    device const bfloat* lnorm=norm_w+ulong(layer)*2048ul;
    device const bfloat* lqw=qw+ulong(layer)*8388608ul;
    device const bfloat* lkw=kw+ulong(layer)*1048576ul;
    device const bfloat* lvw=vw+ulong(layer)*1048576ul;
    device const bfloat* lrw=rw+ulong(layer)*262144ul;
    device const bfloat* law=aw+ulong(layer)*8388608ul;
    device const uint* lup=up+ulong(layer)*25165824ul;
    device const bfloat* lus=us+ulong(layer)*3145728ul;
    device const bfloat* lub=ub+ulong(layer)*3145728ul;
    device const uint* lgate=gate+ulong(layer)*25165824ul;
    device const bfloat* lgs=gs+ulong(layer)*3145728ul;
    device const bfloat* lgb=gb+ulong(layer)*3145728ul;
    device const uint* ldown=down+ulong(layer)*25165824ul;
    device const bfloat* lds=ds+ulong(layer)*3145728ul;
    device const bfloat* ldb=db+ulong(layer)*3145728ul;
    ulong cache_layer=ulong(layer)+DO_PREFIX;
    device const bfloat* layer_keys=key_out+cache_layer*4ul*capacity*128ul;
    device const bfloat* layer_values=value_out+cache_layer*4ul*capacity*128ul;
    bool use_rope=((layer+1)%4)!=0;
    if(phase==0) {{
     if(tid<32)norm_sums[tid]=0;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     float acc0=0,acc1=0;uint p0=tid*4,p1=(tid+256)*4;
     for(uint j=0;j<4;j++){{float value=float(x[p0+j]);acc0+=value*value;}}
     for(uint j=0;j<4;j++){{float value=float(x[p1+j]);acc1+=value*value;}}
     acc0=simd_sum(acc0);acc1=simd_sum(acc1);
     if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if(sg==0){{float total=simd_sum(norm_sums[lane]);if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);}}
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for(uint j=0;j<4;j++)norm[p0+j]=lnorm[p0+j]*bfloat(float(x[p0+j])*norm_inv[0]);
     for(uint j=0;j<4;j++)norm[p1+j]=lnorm[p1+j]*bfloat(float(x[p1+j])*norm_inv[0]);
    }} else if(phase==1)
     north_next_prep_tiled(queue_task,PREP_ROWS,ROUTER_ROWS,sg,lane,norm,lqw,lkw,lvw,lrw,prep,tile);
    else if(phase==2) {{
     if(queue_task<32)for(uint item=queue_task*64+tid;item<(queue_task+1)*64;item+=256) {{
      uint pair=item%64,head=item/64,index=head*128+pair*2;
      float x1=float(prep[index]),x2=float(prep[index+1]);
      if(use_rope) {{
       float inv_freq=metal::exp2(-(float(pair)/64.0f)*15.609640474436812f);
       float theta=float(position)*inv_freq,c=metal::fast::cos(theta),s=metal::fast::sin(theta);
       prep[index]=bfloat(x1*c-x2*s);prep[index+1]=bfloat(x1*s+x2*c);
      }}
     }} else if(queue_task<36)for(uint item=(queue_task-32)*64+tid;item<(queue_task-31)*64;item+=256) {{
      uint pair=item%64,head=item/64,index=head*128+pair*2;
      float x1=float(prep[4096+index]),x2=float(prep[4096+index+1]);
      ulong target=(cache_layer*4ul+head)*capacity*128ul+ulong(position)*128ul+pair*2;
      if(use_rope) {{
       float inv_freq=metal::exp2(-(float(pair)/64.0f)*15.609640474436812f);
       float theta=float(position)*inv_freq,c=metal::fast::cos(theta),s=metal::fast::sin(theta);
       key_out[target]=bfloat(x1*c-x2*s);key_out[target+1]=bfloat(x1*s+x2*c);
      }} else {{key_out[target]=prep[4096+index];key_out[target+1]=prep[4096+index+1];}}
     }} else if(queue_task<38)for(uint item=(queue_task-36)*256+tid;item<(queue_task-35)*256;item+=256) {{
      uint head=item/128,dim=item%128;
      value_out[(cache_layer*4ul+head)*capacity*128ul+ulong(position)*128ul+dim]=prep[4608+item];
     }} else if(tid==0) {{
      float best[8];uint best_ids[8];device const ushort* bits=(device const ushort*)(prep+5120);
      for(uint j=0;j<8;j++){{best[j]=-3.402823466e+38f;best_ids[j]=0xffffffffu;}}
      for(uint expert=0;expert<128;expert++) {{
       float score=sigmoid_table[bits[expert]];uint place=8;
       for(uint j=0;j<8;j++)if(score>best[j]){{place=j;break;}}
       if(place<8){{for(uint j=7;j>place;j--){{best[j]=best[j-1];best_ids[j]=best_ids[j-1];}}best[place]=score;best_ids[place]=expert;}}
      }}
      for(uint j=0;j<8;j++){{ids[j]=best_ids[j];scores[j]=best[j];}}
     }}
    }} else if(phase==3) {{
     uint job=(queue_task<64)?((queue_task%2)?96+queue_task/2:queue_task/2):queue_task-32;
     if(job<96)north_hidden(job,sg,lane,norm,ids,lup,lus,lub,lgate,lgs,lgb,hidden,tile);
     else north_sdpa_packed(job-96,sg,lane,length,head_stride,prep,layer_keys,layer_values,ax,sdpa_outputs,sdpa_max,sdpa_sum);
    }} else if(phase==4) {{
     if(queue_task<128) {{
      uint slot=queue_task/16,rowjob=queue_task%16;
      for(uint j=tid;j<768;j+=256)hidden_tile[j]=hidden[slot*768+j];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for(uint part=0;part<4;part++)north_complete_down(slot,rowjob*128+part*32+sg*4,lane,hidden_tile,ids,ldown,lds,ldb,routed);
     }} else {{
      uint rowjob=queue_task-128;
      for(uint part=0;part<OPROJ_ROWS/32;part++)north_attention(rowjob*(OPROJ_ROWS/32)+part,sg,lane,ax,law,attention);
     }}
    }} else if(queue_task<16) {{
     uint first=queue_task*128;
     for(uint row=first+tid;row<first+128;row+=256) {{
      float value=0;for(uint slot=0;slot<8;slot++){{volatile float product=float(routed[slot*2048+row])*scores[slot];value+=product;}}
      bfloat moe=bfloat(value);x[row]=bfloat(float(bfloat(float(attention[row])+float(moe)))+float(x[row]));
     }}
    }} else {{
     uint job=queue_task-16;
     device const bfloat* nqw=qw+ulong(layer+1)*8388608ul;
     device const bfloat* nkw=kw+ulong(layer+1)*1048576ul;
     device const bfloat* nvw=vw+ulong(layer+1)*1048576ul;
     device const bfloat* nrw=rw+ulong(layer+1)*262144ul;
     constexpr uint qjobs=4096/PREP_ROWS,kvjobs=512/PREP_ROWS;
     volatile device const bfloat* pw;uint rows;
     if(job<qjobs){{pw=nqw+ulong(job)*PREP_ROWS*2048ul;rows=PREP_ROWS;}}
     else if(job<qjobs+kvjobs){{pw=nkw+ulong(job-qjobs)*PREP_ROWS*2048ul;rows=PREP_ROWS;}}
     else if(job<qjobs+2*kvjobs){{pw=nvw+ulong(job-qjobs-kvjobs)*PREP_ROWS*2048ul;rows=PREP_ROWS;}}
     else {{pw=nrw+ulong(job-qjobs-2*kvjobs)*ROUTER_ROWS*2048ul;rows=ROUTER_ROWS;}}
     constexpr uint cols=(PREFETCH_STAGES==0?1:(PREFETCH_STAGES<32?PREFETCH_STAGES:32)*64);
     float sum=0;for(uint i=tid;i<rows*cols;i+=256)sum+=float(pw[(i/cols)*2048+i%cols]);
     sum=simd_sum(sum);if(lane==0)tile[sg]=sum;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if(tid==0){{float total=0;for(uint j=0;j<8;j++)total+=tile[j];atomic_store_explicit(state+3072+job,as_type<uint>(total),memory_order_relaxed);}}
    }}
   }} else if(DO_HEAD && queue_stage<total_stages) {{
    if(queue_stage==head_start) {{
     if(tid<32)norm_sums[tid]=0;
     threadgroup_barrier(mem_flags::mem_threadgroup);
     float acc0=0,acc1=0;uint p0=tid*4,p1=(tid+256)*4;
     for(uint j=0;j<4;j++){{float value=float(x[p0+j]);acc0+=value*value;}}
     for(uint j=0;j<4;j++){{float value=float(x[p1+j]);acc1+=value*value;}}
     acc0=simd_sum(acc0);acc1=simd_sum(acc1);
     if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}
     threadgroup_barrier(mem_flags::mem_threadgroup);
     if(sg==0){{float total=simd_sum(norm_sums[lane]);if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);}}
     threadgroup_barrier(mem_flags::mem_threadgroup);
     for(uint j=0;j<4;j++)norm[p0+j]=final_norm_w[p0+j]*bfloat(float(x[p0+j])*norm_inv[0]);
     for(uint j=0;j<4;j++)norm[p1+j]=final_norm_w[p1+j]*bfloat(float(x[p1+j])*norm_inv[0]);
    }} else north_lm_head(queue_task,LM_ROWS,sg,lane,norm,lm_w,lm_s,lm_b,logits);
   }}
   threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
   if(tid==0)north_queue_finish(state,queue_stage,queue_tasks);
  }} else if(tid==0) {{
   while(atomic_load_explicit(state,memory_order_relaxed)==queue_stage) {{}}
   atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }}
 return;
}}

// Carry the used cache prefix to the new output buffers.
ulong cache_prefix=ulong(NLAYERS+DO_PREFIX)*4ul*ulong(position)*128ul;
for(ulong logical=global_tid;logical<cache_prefix;logical+=global_stride) {{
 uint dim=logical%128ul;ulong z=logical/128ul;
 uint pos=z%position;z/=position;uint head=z%4ul;uint layer=z/4ul;
 ulong target=((ulong(layer)*4ul+head)*capacity+pos)*128ul+dim;
 key_out[target]=key_cache[target];value_out[target]=value_cache[target];
}}
for(uint i=global_tid;i<2048;i+=global_stride)x[i]=x_in[i];
north_grid_barrier(state,0,tid,WORKERS);

device const bfloat* prefix_norm_w=prefix_dense;
device const bfloat* prefix_qw=prefix_dense+2048;
device const bfloat* prefix_kw=prefix_qw+8388608;
device const bfloat* prefix_vw=prefix_kw+1048576;
device const bfloat* prefix_aw=prefix_vw+1048576;
device const uint* prefix_up=prefix_q;
device const uint* prefix_gate=prefix_q+786432;
device const uint* prefix_down=prefix_q+1572864;
device const bfloat* prefix_us=prefix_meta;
device const bfloat* prefix_ub=prefix_meta+98304;
device const bfloat* prefix_gs=prefix_meta+196608;
device const bfloat* prefix_gb=prefix_meta+294912;
device const bfloat* prefix_ds=prefix_meta+393216;
device const bfloat* prefix_db=prefix_meta+491520;

if(DO_PREFIX) {{
 if(group==0) {{
  if(tid<32)norm_sums[tid]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc0=0,acc1=0;uint p0=tid*4,p1=(tid+256)*4;
  for(uint j=0;j<4;j++){{float value=float(x[p0+j]);acc0+=value*value;}}
  for(uint j=0;j<4;j++){{float value=float(x[p1+j]);acc1+=value*value;}}
  acc0=simd_sum(acc0);acc1=simd_sum(acc1);
  if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){{float total=simd_sum(norm_sums[lane]);if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint j=0;j<4;j++)norm[p0+j]=prefix_norm_w[p0+j]*bfloat(float(x[p0+j])*norm_inv[0]);
  for(uint j=0;j<4;j++)norm[p1+j]=prefix_norm_w[p1+j]*bfloat(float(x[p1+j])*norm_inv[0]);
 }}
 north_grid_barrier(state,1,tid,WORKERS);
 for(uint task=group;task<64;task+=WORKERS) {{
  if(task<40)north_next_prep_tiled(task,128,16,sg,lane,norm,prefix_qw,prefix_kw,prefix_vw,prefix_qw,prep,tile);
  else north_prefix_hidden(task-40,sg,lane,norm,prefix_up,prefix_us,prefix_ub,prefix_gate,prefix_gs,prefix_gb,hidden);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }}
 north_grid_barrier(state,2,tid,WORKERS);
 for(uint i=global_tid;i<3072;i+=global_stride)prefix_hidden_debug[i]=hidden[i];
 for(uint item=global_tid;item<2048;item+=global_stride) {{
  uint pair=item%64,head=item/64,index=head*128+pair*2;
  float x1=float(prep[index]),x2=float(prep[index+1]);
  float inv_freq=metal::exp2(-(float(pair)/64.0f)*15.609640474436812f);
  float theta=float(position)*inv_freq,c=metal::fast::cos(theta),s=metal::fast::sin(theta);
  prep[index]=bfloat(x1*c-x2*s);prep[index+1]=bfloat(x1*s+x2*c);
 }}
 for(uint item=global_tid;item<256;item+=global_stride) {{
  uint pair=item%64,head=item/64,index=head*128+pair*2;
  float x1=float(prep[4096+index]),x2=float(prep[4096+index+1]);
  float inv_freq=metal::exp2(-(float(pair)/64.0f)*15.609640474436812f);
  float theta=float(position)*inv_freq,c=metal::fast::cos(theta),s=metal::fast::sin(theta);
  ulong target=(ulong(head)*capacity+position)*128ul+pair*2;
  key_out[target]=bfloat(x1*c-x2*s);key_out[target+1]=bfloat(x1*s+x2*c);
 }}
 for(uint item=global_tid;item<512;item+=global_stride) {{
  uint head=item/128,dim=item%128;
  value_out[(ulong(head)*capacity+position)*128ul+dim]=prep[4608+item];
 }}
 north_grid_barrier(state,3,tid,WORKERS);
 device const bfloat* prefix_keys=key_out;
 device const bfloat* prefix_values=value_out;
 for(uint task=group;task<64;task+=WORKERS) {{
  if(task%2==0)north_sdpa_packed(task/2,sg,lane,length,head_stride,prep,prefix_keys,prefix_values,ax,sdpa_outputs,sdpa_max,sdpa_sum);
  else north_prefix_down(task/2,sg,lane,hidden,prefix_down,prefix_ds,prefix_db,routed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }}
 north_grid_barrier(state,4,tid,WORKERS);
 for(uint rowjob=group;rowjob<64;rowjob+=WORKERS)north_attention(rowjob,sg,lane,ax,prefix_aw,attention);
 north_grid_barrier(state,5,tid,WORKERS);
 for(uint rowjob=group;rowjob<16;rowjob+=WORKERS)for(uint row=rowjob*128+tid;row<(rowjob+1)*128;row+=256)
  x[row]=bfloat(float(bfloat(float(attention[row])+float(routed[row])))+float(x[row]));
 north_grid_barrier(state,6,tid,WORKERS);
 for(uint i=global_tid;i<2048;i+=global_stride){{prefix_debug[i]=x[i];prefix_attn_debug[i]=attention[i];prefix_mlp_debug[i]=routed[i];}}
}}

for(uint layer=0;layer<NLAYERS;layer++) {{
 uint barrier_base=1+DO_PREFIX*6+layer*(FINE?5:6);
 device const bfloat* lnorm=norm_w+ulong(layer)*2048ul;
 device const bfloat* lqw=qw+ulong(layer)*8388608ul;
 device const bfloat* lkw=kw+ulong(layer)*1048576ul;
 device const bfloat* lvw=vw+ulong(layer)*1048576ul;
 device const bfloat* lrw=rw+ulong(layer)*262144ul;
 device const bfloat* law=aw+ulong(layer)*8388608ul;
 device const uint* lup=up+ulong(layer)*25165824ul;
 device const bfloat* lus=us+ulong(layer)*3145728ul;
 device const bfloat* lub=ub+ulong(layer)*3145728ul;
 device const uint* lgate=gate+ulong(layer)*25165824ul;
 device const bfloat* lgs=gs+ulong(layer)*3145728ul;
 device const bfloat* lgb=gb+ulong(layer)*3145728ul;
 device const uint* ldown=down+ulong(layer)*25165824ul;
 device const bfloat* lds=ds+ulong(layer)*3145728ul;
 device const bfloat* ldb=db+ulong(layer)*3145728ul;

 if(group==0) {{
  if(tid<32)norm_sums[tid]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc0=0,acc1=0;uint p0=tid*4,p1=(tid+256)*4;
  for(uint j=0;j<4;j++){{float value=float(x[p0+j]);acc0+=value*value;}}
  for(uint j=0;j<4;j++){{float value=float(x[p1+j]);acc1+=value*value;}}
  acc0=simd_sum(acc0);acc1=simd_sum(acc1);
  if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){{float total=simd_sum(norm_sums[lane]);if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint j=0;j<4;j++)norm[p0+j]=lnorm[p0+j]*bfloat(float(x[p0+j])*norm_inv[0]);
  for(uint j=0;j<4;j++)norm[p1+j]=lnorm[p1+j]*bfloat(float(x[p1+j])*norm_inv[0]);
 }}
 north_grid_barrier(state,barrier_base,tid,WORKERS);

 bool use_rope=((layer+1)%4)!=0;
 constexpr uint prep_tasks=5120/PREP_ROWS+128/ROUTER_ROWS;
 for(uint job=group;job<prep_tasks;job+=WORKERS) {{
  north_next_prep_tiled(job,PREP_ROWS,ROUTER_ROWS,sg,lane,norm,lqw,lkw,lvw,lrw,prep,tile);
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }}
 north_grid_barrier(state,barrier_base+1,tid,WORKERS);

 if(use_rope) for(uint item=global_tid;item<2048;item+=global_stride) {{
  uint pair=item%64,head=item/64,index=head*128+pair*2;
  float x1=float(prep[index]),x2=float(prep[index+1]);
  float inv_freq=metal::exp2(-(float(pair)/64.0f)*15.609640474436812f);
  float theta=float(position)*inv_freq,c=metal::fast::cos(theta),s=metal::fast::sin(theta);
  prep[index]=bfloat(x1*c-x2*s);prep[index+1]=bfloat(x1*s+x2*c);
 }}
 for(uint item=global_tid;item<256;item+=global_stride) {{
  uint pair=item%64,head=item/64,index=head*128+pair*2;
  float x1=float(prep[4096+index]),x2=float(prep[4096+index+1]);
  ulong cache_layer=ulong(layer)+DO_PREFIX;
  ulong target=((cache_layer*4ul+head)*capacity+position)*128ul+pair*2;
  if(use_rope) {{
   float inv_freq=metal::exp2(-(float(pair)/64.0f)*15.609640474436812f);
   float theta=float(position)*inv_freq,c=metal::fast::cos(theta),s=metal::fast::sin(theta);
   key_out[target]=bfloat(x1*c-x2*s);key_out[target+1]=bfloat(x1*s+x2*c);
  }} else {{key_out[target]=prep[4096+index];key_out[target+1]=prep[4096+index+1];}}
 }}
 for(uint item=global_tid;item<512;item+=global_stride) {{
  uint head=item/128,dim=item%128;
  value_out[(((ulong(layer)+DO_PREFIX)*4ul+head)*capacity+position)*128ul+dim]=prep[4608+item];
 }}
 if(global_tid==0) {{
  float best[8];uint best_ids[8];device const ushort* bits=(device const ushort*)(prep+5120);
  for(uint j=0;j<8;j++){{best[j]=-3.402823466e+38f;best_ids[j]=0xffffffffu;}}
  for(uint expert=0;expert<128;expert++) {{
   float score=sigmoid_table[bits[expert]];uint place=8;
   for(uint j=0;j<8;j++)if(score>best[j]){{place=j;break;}}
   if(place<8) {{for(uint j=7;j>place;j--){{best[j]=best[j-1];best_ids[j]=best_ids[j-1];}}best[place]=score;best_ids[place]=expert;}}
  }}
  for(uint j=0;j<8;j++){{ids[j]=best_ids[j];scores[j]=best[j];}}
 }}
 north_grid_barrier(state,barrier_base+2,tid,WORKERS);

 device const bfloat* layer_keys=key_out+(ulong(layer)+DO_PREFIX)*4ul*capacity*128ul;
 device const bfloat* layer_values=value_out+(ulong(layer)+DO_PREFIX)*4ul*capacity*128ul;
 if(FINE) {{
  uint fine_base=512+layer*32,rotation=group%8;
  while(true) {{
   if(tid==0) {{
    dynamic_task=0xffffffffu;
    for(uint ee=0;ee<8 && dynamic_task==0xffffffffu;ee++) {{
     uint slot=(rotation+ee)%8;
     uint claimed=atomic_load_explicit(state+fine_base+9+slot,memory_order_relaxed);
     if(claimed<16 && atomic_load_explicit(state+fine_base+1+slot,memory_order_relaxed)==12) {{
      uint rowjob=atomic_fetch_add_explicit(state+fine_base+9+slot,1u,memory_order_relaxed);
      if(rowjob<16)dynamic_task={FRONT_TASKS}+slot*16+rowjob;
     }}
    }}
    constexpr uint oproj_tasks=2048/OPROJ_ROWS;
    constexpr uint second_tasks=128+oproj_tasks;
    if(dynamic_task==0xffffffffu && atomic_load_explicit(state+fine_base+17,memory_order_relaxed)==32) {{
     uint rowjob=atomic_fetch_add_explicit(state+fine_base+18,1u,memory_order_relaxed);
     if(rowjob<oproj_tasks)dynamic_task={FRONT_TASKS}+128+rowjob;
    }}
    if(dynamic_task==0xffffffffu) {{
     uint front=atomic_fetch_add_explicit(state+fine_base,1u,memory_order_relaxed);
     if(front<{FRONT_TASKS})dynamic_task=front;
    }}
    if(dynamic_task==0xffffffffu && atomic_load_explicit(state+fine_base+19,memory_order_relaxed)==second_tasks)
     dynamic_task=0xfffffffeu;
   }}
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if(dynamic_task==0xfffffffeu)break;
   if(dynamic_task==0xffffffffu) {{
    threadgroup_barrier(mem_flags::mem_threadgroup);rotation=(rotation+1)%8;continue;
   }}
   if(dynamic_task<{FRONT_TASKS}) {{
    uint job=(dynamic_task<64)?((dynamic_task%2)?96+dynamic_task/2:dynamic_task/2):dynamic_task-32;
    if(job<96) {{
     north_hidden(job,sg,lane,norm,ids,lup,lus,lub,lgate,lgs,lgb,hidden,tile);
     threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
     atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
     if(tid==0)atomic_fetch_add_explicit(state+fine_base+1+job/12,1u,memory_order_relaxed);
    }} else {{
     north_sdpa_packed(job-96,sg,lane,length,head_stride,prep,layer_keys,layer_values,ax,sdpa_outputs,sdpa_max,sdpa_sum);
     threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
     atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
     if(tid==0)atomic_fetch_add_explicit(state+fine_base+17,1u,memory_order_relaxed);
    }}
   }} else if(dynamic_task<{FRONT_TASKS}+128) {{
    uint job=dynamic_task-{FRONT_TASKS},slot=job/16,rowjob=job%16;
    if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    for(uint j=tid;j<768;j+=256)hidden_tile[j]=hidden[slot*768+j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint phase=0;phase<4;phase++)north_complete_down(slot,rowjob*128+phase*32+sg*4,lane,hidden_tile,ids,ldown,lds,ldb,routed);
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    if(tid==0)atomic_fetch_add_explicit(state+fine_base+19,1u,memory_order_relaxed);
   }} else {{
    uint rowjob=dynamic_task-({FRONT_TASKS}+128);
    if(tid==0)atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    for(uint phase=0;phase<OPROJ_ROWS/32;phase++)north_attention(rowjob*(OPROJ_ROWS/32)+phase,sg,lane,ax,law,attention);
    threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
    atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
    if(tid==0)atomic_fetch_add_explicit(state+fine_base+19,1u,memory_order_relaxed);
   }}
   threadgroup_barrier(mem_flags::mem_threadgroup);rotation=(rotation+1)%8;
  }}
  north_grid_barrier(state,barrier_base+3,tid,WORKERS);
 }} else {{
  for(uint task=group;task<{FRONT_TASKS};task+=WORKERS) {{
   uint job=(task<64)?((task%2)?96+task/2:task/2):task-32;
   if(job<96)north_hidden(job,sg,lane,norm,ids,lup,lus,lub,lgate,lgs,lgb,hidden,tile);
   else north_sdpa_packed(job-96,sg,lane,length,head_stride,prep,layer_keys,layer_values,ax,sdpa_outputs,sdpa_max,sdpa_sum);
   threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  north_grid_barrier(state,barrier_base+3,tid,WORKERS);

  constexpr uint second_tasks=128+2048/OPROJ_ROWS;
  for(uint local=group;local<second_tasks;local+=WORKERS) {{
   if(local<128) {{
    uint slot=local/16,rowjob=local%16;
    for(uint j=tid;j<768;j+=256)hidden_tile[j]=hidden[slot*768+j];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for(uint phase=0;phase<4;phase++)north_complete_down(slot,rowjob*128+phase*32+sg*4,lane,hidden_tile,ids,ldown,lds,ldb,routed);
   }} else {{
    uint rowjob=local-128;
    for(uint phase=0;phase<OPROJ_ROWS/32;phase++)north_attention(rowjob*(OPROJ_ROWS/32)+phase,sg,lane,ax,law,attention);
   }}
   threadgroup_barrier(mem_flags::mem_threadgroup);
  }}
  north_grid_barrier(state,barrier_base+4,tid,WORKERS);
 }}

 for(uint rowjob=group;rowjob<{JOIN_TASKS};rowjob+=WORKERS) {{
  for(uint row=rowjob*128+tid;row<(rowjob+1)*128;row+=256) {{
   float value=0;for(uint slot=0;slot<8;slot++){{volatile float product=float(routed[slot*2048+row])*scores[slot];value+=product;}}
   bfloat moe=bfloat(value);x[row]=bfloat(float(bfloat(float(attention[row])+float(moe)))+float(x[row]));
  }}
 }}
 if(PREFETCH_STAGES>0 && layer+1<NLAYERS) {{
  device const bfloat* nqw=qw+ulong(layer+1)*8388608ul;
  device const bfloat* nkw=kw+ulong(layer+1)*1048576ul;
  device const bfloat* nvw=vw+ulong(layer+1)*1048576ul;
  device const bfloat* nrw=rw+ulong(layer+1)*262144ul;
  float prefetch_sum=0;
  constexpr uint qjobs=4096/PREP_ROWS,kvjobs=512/PREP_ROWS;
  for(uint job=group;job<prep_tasks;job+=WORKERS) {{
   volatile device const bfloat* pw;
   uint rows;
   if(job<qjobs){{pw=nqw+ulong(job)*PREP_ROWS*2048ul;rows=PREP_ROWS;}}
   else if(job<qjobs+kvjobs){{pw=nkw+ulong(job-qjobs)*PREP_ROWS*2048ul;rows=PREP_ROWS;}}
   else if(job<qjobs+2*kvjobs){{pw=nvw+ulong(job-qjobs-kvjobs)*PREP_ROWS*2048ul;rows=PREP_ROWS;}}
   else {{pw=nrw+ulong(job-qjobs-2*kvjobs)*ROUTER_ROWS*2048ul;rows=ROUTER_ROWS;}}
   constexpr uint cols=(PREFETCH_STAGES==0?1:(PREFETCH_STAGES<32?PREFETCH_STAGES:32)*64);
   for(uint i=tid;i<rows*cols;i+=256)prefetch_sum+=float(pw[(i/cols)*2048+i%cols]);
  }}
  prefetch_sum=simd_sum(prefetch_sum);
  if(lane==0)tile[sg]=prefetch_sum;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid==0){{float total=0;for(uint j=0;j<8;j++)total+=tile[j];atomic_store_explicit(state+3072+group,as_type<uint>(total),memory_order_relaxed);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }}
 north_grid_barrier(state,barrier_base+(FINE?4:5),tid,WORKERS);
}}

if(DO_HEAD) {{
 if(group==0) {{
  if(tid<32)norm_sums[tid]=0;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float acc0=0,acc1=0;uint p0=tid*4,p1=(tid+256)*4;
  for(uint j=0;j<4;j++){{float value=float(x[p0+j]);acc0+=value*value;}}
  for(uint j=0;j<4;j++){{float value=float(x[p1+j]);acc1+=value*value;}}
  acc0=simd_sum(acc0);acc1=simd_sum(acc1);
  if(lane==0){{norm_sums[sg]=acc0;norm_sums[sg+8]=acc1;}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(sg==0){{float total=simd_sum(norm_sums[lane]);if(lane==0)norm_inv[0]=metal::precise::rsqrt(total/2048.0f+1e-6f);}}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for(uint j=0;j<4;j++)norm[p0+j]=final_norm_w[p0+j]*bfloat(float(x[p0+j])*norm_inv[0]);
  for(uint j=0;j<4;j++)norm[p1+j]=final_norm_w[p1+j]*bfloat(float(x[p1+j])*norm_inv[0]);
 }}
 constexpr uint head_barrier=1+DO_PREFIX*6+NLAYERS*(FINE?5:6);
 north_grid_barrier(state,head_barrier,tid,WORKERS);
 for(uint job=group;job<262144/LM_ROWS;job+=WORKERS)north_lm_head(job,LM_ROWS,sg,lane,norm,lm_w,lm_s,lm_b,logits);
}}
'''

kernel = mx.fast.metal_kernel(
    name="north_whole_pass_packed",
    input_names=["x_in", "key_cache", "value_cache", "params", *WEIGHT_NAMES, *HEAD_NAMES, *PREFIX_NAMES, "sigmoid_table"],
    output_names=["key_out", "value_out", "workspace", "logits"],
    header=header,
    source=source,
)


def run_whole_pass(
    weights,
    x,
    key_cache,
    value_cache,
    position,
    workers=36,
    schedule="queue",
    prefetch_stages=0,
    prep_rows=64,
    oproj_rows=64,
    router_rows=8,
    do_head=False,
    lm_rows=512,
    do_prefix=False,
    return_prefix=False,
):
    cache_layers, _, capacity, width = key_cache.shape
    layers = weights["norm_w"].shape[0]
    assert cache_layers == layers + int(do_prefix)
    assert value_cache.shape == key_cache.shape and width == 128
    assert x.shape == (1, 1, 2048)
    params = mx.array([position, capacity], dtype=mx.uint32)
    assert schedule in ("static", "fine", "queue")
    assert 0 <= prefetch_stages <= 32
    assert prep_rows in (32, 64, 128)
    assert oproj_rows in (32, 64, 128)
    assert router_rows in (4, 8, 16, 32)
    assert lm_rows in (32, 64, 128, 256, 512, 1024)
    fine = schedule == "fine"
    safe_queue = schedule == "queue"
    outputs = kernel(
        inputs=[
            x,
            key_cache,
            value_cache,
            params,
            *[weights[name] for name in WEIGHT_NAMES],
            *[weights[name] for name in HEAD_NAMES],
            *[weights[name] for name in PREFIX_NAMES],
            sigmoid_lut(),
        ],
        template=[
            ("NLAYERS", layers),
            ("WORKERS", workers),
            ("FINE", fine),
            ("SAFE_QUEUE", safe_queue),
            ("PREFETCH_STAGES", prefetch_stages),
            ("PREP_ROWS", prep_rows),
            ("OPROJ_ROWS", oproj_rows),
            ("ROUTER_ROWS", router_rows),
            ("DO_HEAD", do_head),
            ("LM_ROWS", lm_rows),
            ("DO_PREFIX", do_prefix),
        ],
        grid=(workers * 256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[key_cache.shape, value_cache.shape, (SIZE,), (1, 1, 262144)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.uint32, mx.bfloat16],
        init_value=0,
    )
    raw = outputs[2].view(mx.bfloat16)
    out = raw[X * 2 : NORM * 2].reshape(1, 1, 2048)
    state_size = (
        3
        if safe_queue
        else 1 + 6 * int(do_prefix) + layers * (5 if fine else 6) + int(do_head)
    )
    values = out, outputs[0], outputs[1], outputs[2][:state_size]
    extras = ()
    if do_head:
        extras += (outputs[3],)
    if return_prefix:
        prefix = raw[PREFIX_DEBUG * 2 : PREFIX_ATTN_DEBUG * 2].reshape(1, 1, 2048)
        prefix_attn = raw[PREFIX_ATTN_DEBUG * 2 : PREFIX_MLP_DEBUG * 2].reshape(1, 1, 2048)
        prefix_mlp = raw[PREFIX_MLP_DEBUG * 2 : PREFIX_HIDDEN_DEBUG * 2].reshape(1, 1, 2048)
        prefix_hidden = raw[PREFIX_HIDDEN_DEBUG * 2 : SIZE * 2].reshape(1, 1, 3072)
        extras += (prefix, prefix_attn, prefix_mlp, prefix_hidden)
    return (*values, *extras)
