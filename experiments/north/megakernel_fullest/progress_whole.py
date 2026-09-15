"""Rescan readiness only after a completion changes the ready set."""
from affinity_whole import build as base

def build(header,source,size):
    h,s=base(header,source,size)
    marker='uint fine_cursor=group%440,idle=0;'
    s=s.replace(marker,marker+'\nuint seen_layer=0xffffffffu,seen_complete=0xffffffffu;bool had_no_ready=false;')
    needle='    fine_node=north_claim_node(fine_states+fine_layer*512,fine_cursor,group,WORKERS);'
    assert needle in s
    s=s.replace(needle,'''    uint completed=atomic_load_explicit(fine_states+fine_layer*512+511,memory_order_relaxed);
    if(had_no_ready && seen_layer==fine_layer && seen_complete==completed)fine_node=0xffffffffu;
    else {
'''+needle+'''
     had_no_ready=fine_node==0xffffffffu;seen_layer=fine_layer;seen_complete=completed;
    }''')
    return h,s
