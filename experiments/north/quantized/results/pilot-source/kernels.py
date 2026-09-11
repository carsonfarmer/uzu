"""Opt-in native-4bit North branch experiment. No full-model patch is implicit."""
from pathlib import Path
import mlx.core as mx
HEADER=(Path(__file__).with_name('kernel.h')).read_text()
NAMES=['x','ax','ids','up','us','ub','gate','gs','gb','down','ds','db','aw']
SHAPES=[(8,768),(8,24,2048),(1,1,2048)]
DTYPES=[mx.bfloat16,mx.float32,mx.bfloat16]
THREADS="""uint tid=thread_position_in_threadgroup.x;
uint lane=thread_index_in_simdgroup;
uint sg=simdgroup_index_in_threadgroup;
uint group=threadgroup_position_in_grid.x;
threadgroup float tile[32];
"""
BODY="""
if(job<192){
 north_hidden(job,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 north_down(job,tid,ids,down,ds,db,tile,partial);
}else north_attention(job-192,sg,lane,ax,aw,attention);
threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
"""
mixed=mx.fast.metal_kernel(name='north_q_mixed',input_names=NAMES,
 output_names=['hidden','partial','attention'],header=HEADER,
 source=THREADS+"""
 for(uint task=group;task<256;task+=WORKERS) {
 uint job=INTERLEAVE ? ((task%4==3)?192+task/4:(task/4)*3+task%4):task;
 """+BODY+"}")
upkernel=mx.fast.metal_kernel(name='north_q_up',input_names=NAMES,
 output_names=['hidden'],header=HEADER,source=THREADS+"north_hidden(group,sg,lane,x,ids,up,us,ub,gate,gs,gb,hidden,tile);")
downkernel=mx.fast.metal_kernel(name='north_q_down',input_names=NAMES+['hidden'],
 output_names=['partial'],header=HEADER,source=THREADS+"""
 if(tid<32)tile[tid]=float(hidden[(group/24)*768+(group%24)*32+tid]);
 threadgroup_barrier(mem_flags::mem_threadgroup);
 north_down(group,tid,ids,down,ds,db,tile,partial);
 """)
attnkernel=mx.fast.metal_kernel(name='north_q_attn',input_names=['ax','aw'],output_names=['attention'],
 header=HEADER,source=THREADS+"north_attention(group,sg,lane,ax,aw,attention);")
join=mx.fast.metal_kernel(name='north_q_join',input_names=['partial','attention','scores','residual'],
 output_names=['out'],source="""
 uint r=thread_position_in_grid.x;
 float sum=0;
 for(uint e=0;e<8;e++) {
  float value=0;
  for(uint c=0;c<24;c++)value+=partial[(e*24+c)*2048+r];
  sum+=float(bfloat(value))*scores[e];
 }
 bfloat ff=bfloat(sum);
 out[r]=bfloat(float(bfloat(float(attention[r])+float(ff)))+float(residual[r]));
 """)

def inputs(mlp,x,ax,ids,aw):
 assert x.shape==(1,1,2048) and ax.shape==(1,1,4096) and ids.size==8
 weights=[]
 for name in ['up_proj','gate_proj','down_proj']:
  m=getattr(mlp.switch_mlp,name)
  assert (m.bits,m.group_size,m.mode)==(4,64,'affine')
  assert m.scales.dtype==mx.bfloat16 and m.biases.dtype==mx.bfloat16
  weights += [m.weight,m.scales,m.biases]
 assert x.dtype==mx.bfloat16 and ax.dtype==mx.bfloat16 and aw.shape==(2048,4096)
 return [x,ax,ids.astype(mx.uint32),*weights,aw]

def run(data,scores,residual,workers=128,interleave=False,phased=False,return_parts=False):
 if phased:
  h=upkernel(inputs=data,grid=(192*256,1,1),threadgroup=(256,1,1),
    output_shapes=[SHAPES[0]],output_dtypes=[DTYPES[0]])[0]
  p=downkernel(inputs=data+[h],grid=(192*256,1,1),threadgroup=(256,1,1),
    output_shapes=[SHAPES[1]],output_dtypes=[DTYPES[1]])[0]
  a=attnkernel(inputs=[data[1],data[-1]],grid=(64*256,1,1),threadgroup=(256,1,1),
    output_shapes=[SHAPES[2]],output_dtypes=[DTYPES[2]])[0]
 else:
  h,p,a=mixed(inputs=data,template=[('WORKERS',workers),('INTERLEAVE',interleave)],
    grid=(workers*256,1,1),threadgroup=(256,1,1),output_shapes=SHAPES,output_dtypes=DTYPES)
 out=join(inputs=[p,a,scores,residual],grid=(2048,1,1),threadgroup=(256,1,1),
   output_shapes=[(1,1,2048)],output_dtypes=[mx.bfloat16])[0]
 return (out,h,p,a) if return_parts else out
