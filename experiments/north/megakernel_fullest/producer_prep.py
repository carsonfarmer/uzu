"""Dedicated SIMD loader plus eight exact consumer SIMD groups and a bounded ring.

Primitive prototype: future weights may load while consumers normalize the input
and then compute previous chunks. This is not yet cross-layer model integration.
"""
from staged_prep import BASE
import ast

HEADER=r'''
#include <metal_stdlib>
using namespace metal;
inline bool consumer_barrier(threadgroup atomic_uint* arrivals,threadgroup atomic_uint* epoch,
 threadgroup atomic_uint* error) {
 uint observed=atomic_load_explicit(epoch,memory_order_relaxed);
 simdgroup_barrier(mem_flags::mem_threadgroup);
 if(simd_is_first()) {
  atomic_thread_fence(mem_flags::mem_threadgroup,memory_order_seq_cst,thread_scope_threadgroup);
  if(atomic_fetch_add_explicit(arrivals,1u,memory_order_relaxed)==7) {
   atomic_store_explicit(arrivals,0u,memory_order_relaxed);
   atomic_fetch_add_explicit(epoch,1u,memory_order_relaxed);
  } else {
   uint spins=0;
   while(atomic_load_explicit(epoch,memory_order_relaxed)==observed) {
    if(++spins>1000000){atomic_store_explicit(error,1u,memory_order_relaxed);break;}
   }
  }
  atomic_thread_fence(mem_flags::mem_threadgroup,memory_order_seq_cst,thread_scope_threadgroup);
 }
 simdgroup_barrier(mem_flags::mem_threadgroup);
 return atomic_load_explicit(error,memory_order_relaxed)==0;
}
'''

def build_source(leader=False):
    tree=ast.parse((BASE/'prep.py').read_text())
    norm=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NORM' for t in n.targets))
    # Consumer-only synchronization; producer must not participate in these barriers.
    norm=norm.replace('uint tid=thread_position_in_threadgroup.x;','')
    norm=norm.replace('threadgroup_barrier(mem_flags::mem_threadgroup);','if(!consumer_barrier(&arrivals,&epoch,&error)){if(tid==0)errors[job]=1;return;}')
    # Norm output is already used; pointer named norm matches the extracted body.
    body=r'''
uint tid=thread_position_in_threadgroup.x,sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint job=threadgroup_position_in_grid.x;
constexpr uint rows=ROUTER?4:32;
threadgroup bfloat ring[DEPTH*rows*128];
threadgroup atomic_uint ready[DEPTH],empty[DEPTH],arrivals,epoch,error;
if(tid==0){atomic_store_explicit(&arrivals,0u,memory_order_relaxed);atomic_store_explicit(&epoch,0u,memory_order_relaxed);atomic_store_explicit(&error,0u,memory_order_relaxed);}
if(tid<DEPTH){atomic_store_explicit(ready+tid,0u,memory_order_relaxed);atomic_store_explicit(empty+tid,0u,memory_order_relaxed);}
threadgroup_barrier(mem_flags::mem_threadgroup);
if(sg==8) {
 for(uint chunk=0;chunk<16;chunk++) {
  uint slot=chunk%DEPTH,generation=chunk/DEPTH;
  uint spins=0;
  while(atomic_load_explicit(empty+slot,memory_order_relaxed)!=generation) {
   if(atomic_load_explicit(&error,memory_order_relaxed)!=0)return;
   if(++spins>1000000){atomic_store_explicit(&error,1u,memory_order_relaxed);return;}
  }
  atomic_thread_fence(mem_flags::mem_threadgroup,memory_order_seq_cst,thread_scope_threadgroup);
  for(uint i=lane;i<rows*128;i+=32)ring[slot*rows*128+i]=w[(job*rows+i/128)*2048+chunk*128+i%128];
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0){atomic_thread_fence(mem_flags::mem_threadgroup,memory_order_seq_cst,thread_scope_threadgroup);atomic_store_explicit(ready+slot,generation+1,memory_order_relaxed);}
 }
 return;
}
'''+norm+r'''
float acc[4]={0};
for(uint chunk=0;chunk<16;chunk++) {
 uint slot=chunk%DEPTH,generation=chunk/DEPTH;
 uint spins=0;
 while(atomic_load_explicit(ready+slot,memory_order_relaxed)!=generation+1) {
  if(atomic_load_explicit(&error,memory_order_relaxed)!=0){if(tid==0)errors[job]=1;return;}
  if(++spins>1000000){atomic_store_explicit(&error,1u,memory_order_relaxed);if(tid==0)errors[job]=1;return;}
 }
 atomic_thread_fence(mem_flags::mem_threadgroup,memory_order_seq_cst,thread_scope_threadgroup);
 if(!ROUTER || chunk%8==sg) {
  uint local_row=ROUTER?0:sg*4;
  for(uint rr=0;rr<4;rr++)for(uint j=0;j<4;j++)
   acc[rr]+=float(ring[slot*rows*128+(local_row+rr)*128+lane*4+j])*float(normalized[chunk*128+lane*4+j]);
 }
 if(!consumer_barrier(&arrivals,&epoch,&error)){if(tid==0)errors[job]=1;return;}
 if(tid==0)atomic_store_explicit(empty+slot,generation+1,memory_order_relaxed);
}
for(uint rr=0;rr<4;rr++) {
 for(ushort offset=16;offset>=1;offset>>=1)acc[rr]+=simd_shuffle_down(acc[rr],offset);
 if(!ROUTER){if(lane==0)y[job*rows+sg*4+rr]=bfloat(acc[rr]);}
}
if(ROUTER) {
 threadgroup float partial[32];
 if(lane==0)for(uint rr=0;rr<4;rr++)partial[sg*4+rr]=acc[rr];
 if(!consumer_barrier(&arrivals,&epoch,&error)){if(tid==0)errors[job]=1;return;}
 if(tid==0)for(uint rr=0;rr<4;rr++){float sum=acc[rr];for(uint part=1;part<8;part++)sum+=partial[part*4+rr];y[job*4+rr]=bfloat(sum);}
}
if(tid==0)errors[job]=atomic_load_explicit(&error,memory_order_relaxed);
'''
    h=HEADER
    if leader:
        h=h.replace('uint observed=atomic_load_explicit(epoch,memory_order_relaxed);','uint observed=0;if(simd_is_first())observed=atomic_load_explicit(epoch,memory_order_relaxed);')
        start=body.index('  uint spins=0;')
        end=body.index('  atomic_thread_fence',start)
        body=body[:start]+'''  if(lane==0) {
   uint spins=0;
   while(atomic_load_explicit(empty+slot,memory_order_relaxed)!=generation) {
    if(atomic_load_explicit(&error,memory_order_relaxed)!=0)break;
    if(++spins>1000000){atomic_store_explicit(&error,1u,memory_order_relaxed);break;}
   }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  if(atomic_load_explicit(&error,memory_order_relaxed)!=0)return;
'''+body[end:]
        start=body.index(' uint spins=0;')
        # Locate the consumer wait explicitly; producer's differently indented declaration is earlier.
        start=body.index(' uint spins=0;',body.index('float acc[4]={0};'))
        end=body.index(' atomic_thread_fence',start)
        body=body[:start]+''' if(lane==0) {
  uint spins=0;
  while(atomic_load_explicit(ready+slot,memory_order_relaxed)!=generation+1) {
   if(atomic_load_explicit(&error,memory_order_relaxed)!=0)break;
   if(++spins>1000000){atomic_store_explicit(&error,1u,memory_order_relaxed);break;}
  }
 }
 simdgroup_barrier(mem_flags::mem_threadgroup);
 if(atomic_load_explicit(&error,memory_order_relaxed)!=0){if(tid==0)errors[job]=1;return;}
'''+body[end:]
    return h,body

_KERNELS={}

def run(x,w,nw,router=False,depth=2,leader=False):
    import mlx.core as mx
    assert depth in (1,2)
    if leader not in _KERNELS:
        h,s=build_source(leader);_KERNELS[leader]=mx.fast.metal_kernel(name='north_producer_ring_prep_'+str(int(leader)),input_names=['x','w','nw'],output_names=['y','norm','errors'],header=h,source=s)
    count=w.shape[0]//(4 if router else 32)
    return _KERNELS[leader](inputs=[x,w,nw],template=[('ROUTER',router),('DEPTH',depth)],grid=(count*288,1,1),threadgroup=(288,1,1),output_shapes=[(1,1,w.shape[0]),(1,1,2048),(count,)],output_dtypes=[mx.bfloat16,mx.bfloat16,mx.uint32],init_value=0)
