"""Real future-layer register weights retained and consumed inside whole decode.

Early and late use identical future-task reservations. Early loads the immutable
prefix while the prior layer still has work; late loads only when its activation
is ready. Held tasks never prevent the worker from running ready producer work.
"""
from affinity_whole import build as base
from register_tail import build_source as register_source


def build(header,source,size,early=True):
    h,s=base(header,source,size)
    rh,_=register_source();rh=rh[rh.index('inline void north_next_prep_tiled'):]
    h+='\n'+rh+r'''
inline uint north_future_job(uint ticket) {
 uint n=8,q=64,k=8;
 if(ticket<32){uint kind=ticket%4,tile=ticket/4;return (kind==0?0:(kind==1?q:(kind==2?q+k:q+2*k)))+tile;}
 ticket-=32;if(ticket<56)return 8+ticket;return 88+ticket-56;
}
'''
    marker='uint fine_cursor=group%440,idle=0;'
    assert marker in s
    s=s.replace(marker,marker+'''
threadgroup uint held_node,held_layer,held_loaded,held_valid;
bfloat future_weights[96];
if(tid==0){held_node=0;held_layer=0;held_loaded=0;held_valid=0;}
threadgroup_barrier(mem_flags::mem_threadgroup);
''')
    needle='    fine_node=north_claim_node(fine_states+fine_layer*512,fine_cursor,group,WORKERS);'
    assert needle in s
    s=s.replace(needle,'''    if(held_valid && held_layer==fine_layer && north_node_done(fine_states+fine_layer*512,0))fine_node=held_node;
    else fine_node=north_claim_node(fine_states+fine_layer*512,fine_cursor,group,WORKERS);''')
    needle='     north_next_prep_tiled(queue_task,PREP_ROWS,ROUTER_ROWS,sg,lane,norm,lqw,lkw,lvw,lrw,prep,tile);'
    assert s.count(needle)==1
    s=s.replace(needle,''' {
     if(held_valid && held_layer==layer && held_node==fine_node) {
      if(!held_loaded)north_load_prefix(queue_task,64,8,128,tid,sg,lane,lqw,lkw,lvw,lrw,future_weights);
      north_next_prep_tiled(queue_task,64,8,sg,lane,norm,lqw,lkw,lvw,lrw,prep,tile,future_weights,128);
     } else north_next_prep_tiled(queue_task,64,8,sg,lane,norm,lqw,lkw,lvw,lrw,prep,tile);
    }''')
    needle='     fine_cursor=(fine_node+1)%440;'
    assert needle in s
    s=s.replace(needle,needle+'''
     if(held_valid && held_layer==fine_layer && held_node==fine_node){held_valid=0;held_loaded=0;}
''')
    # Add reservation/loading at the common loop tail, outside the tid0 branch.
    needle='  threadgroup_barrier(mem_flags::mem_threadgroup);\n }\n return;'
    assert s.count(needle)==1
    s=s.replace(needle,f'''
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(tid==0 && fine_node!=0xffffffffu && !held_valid && fine_layer+1<NLAYERS && north_node_phase(fine_node)>=3) {{
   device atomic_uint* future=fine_states+(fine_layer+1)*512;
   if(atomic_load_explicit(future+510,memory_order_relaxed)<96) {{
    uint ticket=atomic_fetch_add_explicit(future+510,1u,memory_order_relaxed);
    if(ticket<96) {{
     uint node=1+north_future_job(ticket),expected=0;
     if(atomic_compare_exchange_weak_explicit(future+node,&expected,1u,memory_order_relaxed,memory_order_relaxed)){{held_valid=1;held_layer=fine_layer+1;held_node=node;held_loaded=0;}}
    }}
   }}
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if({str(early).lower()} && held_valid && !held_loaded) {{
   north_load_prefix(held_node-1,64,8,128,tid,sg,lane,
      qw+ulong(held_layer)*8388608ul,kw+ulong(held_layer)*1048576ul,
      vw+ulong(held_layer)*1048576ul,rw+ulong(held_layer)*262144ul,future_weights);
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if(tid==0)held_loaded=1;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
 }}
 return;''')
    return h,s
