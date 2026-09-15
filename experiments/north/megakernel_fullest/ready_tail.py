"""Ready branch-tail tasks with dynamically owned future QKV/router prefixes.

Extends the retained exact tail arithmetic. A worker reserves at most one future
prep tile, retains its prefix while executing producer tasks, then consumes it.
No grid-group identity is required for completion. No global grid barrier.
"""
import ast
from pathlib import Path
from staged_prep import build_source as prep_source

NORTH=Path(__file__).resolve().parents[1]

def upstream():
    """Evaluate source constants only; CPU-safe and independent of MLX imports."""
    path=NORTH/'full_layer/tail_prep.py'
    env={'Path':Path,'__file__':str(path),'HEADER':(NORTH/'quantized/kernel.h').read_text()}
    ktree=ast.parse((NORTH/'quantized/kernels.py').read_text())
    env['NAMES']=next(ast.literal_eval(n.value) for n in ktree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NAMES' for t in n.targets))
    for node in ast.parse(path.read_text()).body:
        if isinstance(node,(ast.Assign,ast.AugAssign)):
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='kernel' for t in node.targets):break
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env)
    return env

U=upstream()

def build_source():
    s=U['source'];header=U['header']
    def replace(old,new):
        nonlocal s
        assert s.count(old)==1,(old[:100],s.count(old))
        s=s.replace(old,new)
    start=s.index('threadgroup bfloat prefetched_q[8192];')
    end=s.index('while(true)',start)
    s=s[:start]+'''threadgroup bfloat staged[ROWS*PREFIX];
threadgroup uint task,owned,loaded;
uint rotation=group%8,idle=0,ready_mask=0;
bool norm_seen=false;
if(tid==0){owned=0xffffffffu;loaded=0;}
threadgroup_barrier(mem_flags::mem_threadgroup);
'''+s[end:]
    old='  if(task==0xffffffffu && norm_seen && !prefetched_done)task=0xfffffffdu;'
    begin=s.index(old);end=s.index('\n  if(task==0xffffffffu) {',begin)
    s=s[:begin]+f'''  if(task==0xffffffffu && norm_seen) {{
   if(owned==0xffffffffu) {{
    uint job=atomic_fetch_add_explicit(state+{U['PREP_CLAIM']},1u,memory_order_relaxed);
    if(job<PREP_TASKS)owned=job;
   }}
   if(owned!=0xffffffffu)task={U['FRONT_TASKS']+U['DOWN_TASKS']+U['JOIN_TASKS']+1}+owned;
  }}
'''+s[end:]
    begin=s.index('  if(task==0xffffffffu && atomic_load_explicit(state+')
    end=s.index('\n }\n threadgroup_barrier',begin)
    s=s[:begin]+f'''  if(task==0xffffffffu && atomic_load_explicit(state+{U['PREFETCH_DONE']},memory_order_relaxed)==PREP_TASKS)
   task=0xfffffffeu;
'''+s[end:]
    begin=s.index(' if(task==0xfffffffdu) {');end=s.index(' else if(task<',begin)
    s=s[:begin]+' if'+s[end+len(' else if'):]
    begin=s.index('  north_next_gemv(');end=s.index(' rotation=(rotation+1)%8;',begin)
    s=s[:begin]+f'''  if(!loaded)north_load_prefix(owned,ROWS,ROUTER_ROWS,PREFIX,tid,next_qw,next_kw,next_vw,next_rw,staged);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  north_next_prep_tiled(owned,ROWS,ROUTER_ROWS,sg,lane,norm,next_qw,next_kw,next_vw,next_rw,prep,tile,staged,PREFIX);
  threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  atomic_thread_fence(mem_flags::mem_device,memory_order_seq_cst,thread_scope_device);
  if(tid==0){{atomic_fetch_add_explicit(state+{U['PREFETCH_DONE']},1u,memory_order_relaxed);owned=0xffffffffu;loaded=0;}}
 }}
 // Reservation does not block: this worker continues producers until norm ready.
 if(PREFETCH && atomic_load_explicit(state+{U['NORM_READY']},memory_order_relaxed)==0) {{
  if(tid==0 && owned==0xffffffffu) {{
   uint job=atomic_fetch_add_explicit(state+{U['PREP_CLAIM']},1u,memory_order_relaxed);
   if(job<PREP_TASKS)owned=job;
  }}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(owned!=0xffffffffu && !loaded) {{
   north_load_prefix(owned,ROWS,ROUTER_ROWS,PREFIX,tid,next_qw,next_kw,next_vw,next_rw,staged);
   threadgroup_barrier(mem_flags::mem_threadgroup);
   if(tid==0){{loaded=1;if(AUDIT)atomic_fetch_add_explicit(state+{U['EARLY_ROUNDS']},1u,memory_order_relaxed);}}
  }}
 }}
 if(tid==0 && AUDIT)atomic_fetch_add_explicit(state+{U['VISITS']}+task,1u,memory_order_relaxed);
 threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
'''+s[end:]
    # All lanes must take the same staging branch even if another group publishes norm.
    s=s.replace(f' if(PREFETCH && atomic_load_explicit(state+{U["NORM_READY"]},memory_order_relaxed)==0) {{',
        f' if(tid==0)norm_seen=atomic_load_explicit(state+{U["NORM_READY"]},memory_order_relaxed)!=0;\n threadgroup_barrier(mem_flags::mem_threadgroup);\n if(PREFETCH && !norm_seen) {{')
    # norm_seen must be shared, not a tid0-local flag used by other lanes.
    s=s.replace('threadgroup uint task,owned,loaded;','threadgroup uint task,owned,loaded;\nthreadgroup bool norm_seen;').replace('bool norm_seen=false;','').replace('owned=0xffffffffu;loaded=0;}','owned=0xffffffffu;loaded=0;norm_seen=false;}',1)
    ph,_=prep_source()
    ph=ph.replace('threadgroup const bfloat* x,','coherent(device) device const bfloat* x,')
    ph=ph.replace('device bfloat* prep,','coherent(device) device bfloat* prep,').replace('device bfloat* y;','coherent(device) device bfloat* y;')
    header+='\n'+ph+'''
inline void north_load_prefix(uint job,uint rows,uint router_rows,uint prefix,uint tid,
 device const bfloat* qw,device const bfloat* kw,device const bfloat* vw,device const bfloat* rw,
 threadgroup bfloat* staged) {
 uint qjobs=4096/rows,kvjobs=512/rows,first,count;device const bfloat* w;
 if(job<qjobs){w=qw;first=job*rows;count=rows;}
 else if(job<qjobs+kvjobs){w=kw;first=(job-qjobs)*rows;count=rows;}
 else if(job<qjobs+2*kvjobs){w=vw;first=(job-qjobs-kvjobs)*rows;count=rows;}
 else {w=rw;first=(job-qjobs-2*kvjobs)*router_rows;count=router_rows;}
 for(uint i=tid;i<count*prefix;i+=256)staged[i]=w[(first+i/prefix)*2048+i%prefix];
}
'''
    return header,s

_KERNEL=None

def run(data,scores,residual,next_layer,workers=32,rows=32,prefix=128,prefetch=False,audit=False):
    global _KERNEL
    import mlx.core as mx
    assert rows in (32,64) and prefix in (128,256) and rows*prefix<=8192
    assert 1<=workers<=256
    rr=8;prep_tasks=5120//rows+128//rr
    if _KERNEL is None:
        h,s=build_source()
        _KERNEL=mx.fast.metal_kernel(name='north_ready_consumed_future_prefix',input_names=U['NAMES']+['scores','residual']+U['NEXT_NAMES'],output_names=['workspace'],header=h,source=s)
    att=next_layer.self_attn
    ws=_KERNEL(inputs=data+[scores,residual,next_layer.input_layernorm.weight,att.q_proj.weight,att.k_proj.weight,att.v_proj.weight,next_layer.mlp.gate.weight],
        template=[('INTERLEAVE',True),('PREFETCH',prefetch),('AUDIT',audit),('ROWS',rows),('PREFIX',prefix),('ROUTER_ROWS',rr),('PREP_TASKS',prep_tasks)],
        grid=(workers*256,1,1),threadgroup=(256,1,1),output_shapes=[(U['SIZE'],)],output_dtypes=[mx.uint32],init_value=0)[0]
    raw=ws.view(mx.bfloat16)
    out=raw[U['OUT']*2:U['NORM']*2].reshape(1,1,2048)
    norm=raw[U['NORM']*2:U['PREP']*2].reshape(1,1,2048)
    p=raw[U['PREP']*2:]
    values=(out,norm,p[:4096].reshape(1,1,4096),p[4096:4608].reshape(1,1,512),p[4608:5120].reshape(1,1,512),p[5120:].reshape(1,1,128))
    return values+(ws[:U['STATE']],) if audit else values
