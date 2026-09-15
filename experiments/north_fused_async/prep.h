// Arithmetic adapted from Apple MLX; see NOTICE.md.
#include <metal_stdlib>
using namespace metal;
inline void north_next_prep_tiled(
 uint job,uint rows,uint router_rows,uint sg,uint lane,device const bfloat* x,
 device const bfloat* qw,device const bfloat* kw,
 device const bfloat* vw,device const bfloat* rw,
 device bfloat* prep,threadgroup float* scratch) {
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
 device const bfloat* w;device bfloat* y;uint block;
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

