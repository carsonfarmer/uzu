"""Reduce unsuccessful ready scans and exhausted atomic claims, same exact tasks."""
from ready_tail import U

def build_source(register=False):
    if register:
        from register_tail import build_source as base
    else:
        from ready_tail import build_source as base
    h,s=base()
    front=U['FRONT_TASKS'];claim=U['PREP_CLAIM']
    needle='  task=0xffffffffu;'
    assert s.count(needle)==1
    s=s.replace(needle,needle+f'''
  // Claim unclaimed producers before repeatedly scanning their dependency bits.
  // Once producers are assigned, ready consumers can overlap their tails.
  if(atomic_load_explicit(state,memory_order_relaxed)<{front}) {{
   uint first=atomic_fetch_add_explicit(state,1u,memory_order_relaxed);
   if(first<{front})task=first;
  }}''')
    # Skip joins/norm readiness loads entirely when a front task has been selected.
    start=s.index('  // Join a row tile')
    end=s.index('\n  if(task==0xffffffffu) {',start)
    s=s[:start]+'  if(task==0xffffffffu) {\n'+s[start:end]+'\n  }'+s[end:]
    s=s.replace('  if(task==0xffffffffu) {\n   uint front=',f'  if(task==0xffffffffu && atomic_load_explicit(state,memory_order_relaxed)<{front}) {{\n   uint front=')
    s=s.replace('if(owned==0xffffffffu) {',f'if(owned==0xffffffffu && atomic_load_explicit(state+{claim},memory_order_relaxed)<PREP_TASKS) {{')
    s=s.replace('if(tid==0 && owned==0xffffffffu) {',f'if(tid==0 && owned==0xffffffffu && atomic_load_explicit(state+{claim},memory_order_relaxed)<PREP_TASKS) {{')
    return h,s
