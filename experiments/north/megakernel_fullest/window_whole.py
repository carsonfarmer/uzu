"""Two reserved future tiles per worker; actual consumed prefix storage depth sweep.

This increases retained future work. It is not an independent copy-engine or
producer-warp implementation; those mechanisms must not be inferred from depth.
"""
from prefetch_whole import build as base

def build(header,source,size,early=True):
    h,s=base(header,source,size,early)
    s=s.replace('threadgroup uint held_node,held_layer,held_loaded,held_valid;\nbfloat future_weights[96];\nif(tid==0){held_node=0;held_layer=0;held_loaded=0;held_valid=0;}',
        'threadgroup uint held_node[2],held_layer[2],held_loaded[2],held_valid[2],selected_slot;\nbfloat future_weights[192];\nif(tid==0){selected_slot=0xffffffffu;for(uint slot=0;slot<2;slot++){held_node[slot]=0;held_layer[slot]=0;held_loaded[slot]=0;held_valid[slot]=0;}}')
    needle='    if(held_valid && held_layer==fine_layer && north_node_done(fine_states+fine_layer*512,0))fine_node=held_node;'
    assert needle in s
    s=s.replace(needle,'''    selected_slot=0xffffffffu;
    if(north_node_done(fine_states+fine_layer*512,0))for(uint slot=0;slot<2;slot++)if(held_valid[slot] && held_layer[slot]==fine_layer){selected_slot=slot;break;}
    if(selected_slot!=0xffffffffu)fine_node=held_node[selected_slot];''')
    start=s.index('     if(held_valid && held_layer==layer && held_node==fine_node) {')
    end=s.index('     } else north_next_prep_tiled',start)
    block=s[start:end]
    block=block.replace('held_valid && held_layer==layer && held_node==fine_node','selected_slot!=0xffffffffu').replace('!held_loaded','!held_loaded[selected_slot]').replace('future_weights','future_weights+selected_slot*96')
    s=s[:start]+block+s[end:]
    s=s.replace('if(held_valid && held_layer==fine_layer && held_node==fine_node){held_valid=0;held_loaded=0;}',
        'if(selected_slot!=0xffffffffu){held_valid[selected_slot]=0;held_loaded[selected_slot]=0;}')
    start=s.index('  if(tid==0 && fine_node!=0xffffffffu && !held_valid')
    end=s.index('  threadgroup_barrier(mem_flags::mem_threadgroup);',start)
    s=s[:start]+'''  if(tid==0 && fine_node!=0xffffffffu && fine_layer+1<NLAYERS && north_node_phase(fine_node)>=3) {
   for(uint slot=0;slot<2;slot++)if(!held_valid[slot]) {
    device atomic_uint* future=fine_states+(fine_layer+1)*512;
    if(atomic_load_explicit(future+510,memory_order_relaxed)<96) {
     uint ticket=atomic_fetch_add_explicit(future+510,1u,memory_order_relaxed);
     if(ticket<96) {
      uint node=1+north_future_job(ticket),expected=0;
      if(atomic_compare_exchange_weak_explicit(future+node,&expected,1u,memory_order_relaxed,memory_order_relaxed)){held_valid[slot]=1;held_layer[slot]=fine_layer+1;held_node[slot]=node;held_loaded[slot]=0;}
     }
    }
    break;
   }
  }
'''+s[end:]
    start=s.index(f'  if({str(early).lower()} && held_valid && !held_loaded) {{')
    end=s.index('  threadgroup_barrier(mem_flags::mem_threadgroup);\n }\n return;',start)
    block=s[start:end]
    for name in ('held_valid','held_loaded','held_node','held_layer'):
        block=block.replace(name,name+'[slot]')
    block=block.replace('future_weights','future_weights+slot*96')
    s=s[:start]+'  for(uint slot=0;slot<2;slot++) {\n'+block+'  }\n'+s[end:]
    return h,s
