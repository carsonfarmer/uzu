"""Continuous round-robin preferred task lists with ready stealing fallback."""
from fine_whole import build as base

def build(header,source,size):
    h,s=base(header,source,size)
    start='inline uint north_claim_node(device atomic_uint* t,uint cursor) {'
    assert start in h
    h=h.replace(start,'''inline uint north_claim_node(device atomic_uint* t,uint cursor,uint group,uint workers) {
 // Preferred lists carry the RR placement across operation boundaries.
 // No worker waits on its list: ready global work remains stealable.
 for(uint n=group;n<440;n+=workers) {
  if(atomic_load_explicit(t+n,memory_order_relaxed)!=0 || !north_node_ready(t,n))continue;
  uint expected=0;
  if(atomic_compare_exchange_weak_explicit(t+n,&expected,1u,memory_order_relaxed,memory_order_relaxed))return n;
 }
''')
    s=s.replace('north_claim_node(fine_states+fine_layer*512,fine_cursor)',
                'north_claim_node(fine_states+fine_layer*512,fine_cursor,group,WORKERS)')
    return h,s
