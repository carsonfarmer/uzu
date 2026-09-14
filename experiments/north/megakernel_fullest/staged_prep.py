"""Real consumed QKV/router prefix staging; primitive cost screen only.

Early means before this task's RMSNorm, NOT cross-layer overlap. This screen
establishes staging cost before embedding the same arithmetic in a task engine.
Both orders load/store identical bytes and retain the distinct MLX reductions.
"""
import ast
from pathlib import Path

BASE = Path(__file__).resolve().parents[1] / 'additional'

def build_source():
    header = (BASE / 'prep.h').read_text()
    header = header.replace('threadgroup float* scratch)',
                            'threadgroup float* scratch,threadgroup const bfloat* staged,uint prefix)')
    header = header.replace('device const bfloat* x,','threadgroup const bfloat* x,')
    header = header.replace('float(rw[(row+rr)*2048+k+j])',
        'float((k+j<prefix)?staged[(phase*4+rr)*prefix+k+j]:rw[(row+rr)*2048+k+j])')
    header = header.replace('float(w[(row+rr)*2048+k+j])',
        'float((k+j<prefix)?staged[(phase*32+sg*4+rr)*prefix+k+j]:w[(row+rr)*2048+k+j])')
    tree=ast.parse((BASE/'prep.py').read_text())
    norm=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
              and any(isinstance(t,ast.Name) and t.id=='NORM' for t in n.targets))
    norm=norm.replace('uint tid=thread_position_in_threadgroup.x;','')
    declarations='''
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint tid=thread_position_in_threadgroup.x,job=threadgroup_position_in_grid.x;
threadgroup bfloat staged[ROWS*PREFIX];
threadgroup float scratch[64];
uint qjobs=4096/ROWS,kvjobs=512/ROWS;
device const bfloat* w;uint first,count;
if(job<qjobs){w=qw;first=job*ROWS;count=ROWS;}
else if(job<qjobs+kvjobs){w=kw;first=(job-qjobs)*ROWS;count=ROWS;}
else if(job<qjobs+2*kvjobs){w=vw;first=(job-qjobs-kvjobs)*ROWS;count=ROWS;}
else {w=rw;first=(job-qjobs-2*kvjobs)*ROUTER_ROWS;count=ROUTER_ROWS;}
'''
    loads='''
for(uint i=tid;i<count*PREFIX;i+=256)
 staged[i]=w[(first+i/PREFIX)*2048+i%PREFIX];
threadgroup_barrier(mem_flags::mem_threadgroup);
'''
    source=declarations+'if(EARLY){'+loads+'}\n'+norm+'if(!EARLY){'+loads+'}\n'+'''
north_next_prep_tiled(job,ROWS,ROUTER_ROWS,sg,lane,normalized,qw,kw,vw,rw,prep,scratch,staged,PREFIX);
'''
    return header,source

_KERNEL=None

def run(x, weights, norm_weight, rows=32, router_rows=8, prefix=128, early=False):
    global _KERNEL
    import mlx.core as mx
    assert rows in (32,64) and router_rows in (4,8,16) and prefix in (128,256)
    assert rows*prefix*2+2048*2+64*4+32*4+4<=32768
    if _KERNEL is None:
        header,source=build_source()
        _KERNEL=mx.fast.metal_kernel(name='north_consumed_prefix_screen',
            input_names=['x','qw','kw','vw','rw','nw'],output_names=['prep','norm'],
            header=header,source=source)
    return _KERNEL(inputs=[x,*weights,norm_weight],
        template=[('ROWS',rows),('ROUTER_ROWS',router_rows),('PREFIX',prefix),('EARLY',early)],
        grid=((5120//rows+128//router_rows)*256,1,1),threadgroup=(256,1,1),
        output_shapes=[(1,1,5248),(1,1,2048)],output_dtypes=[mx.bfloat16,mx.bfloat16])
