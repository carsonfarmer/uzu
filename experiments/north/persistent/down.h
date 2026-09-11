// Same full-K affine accumulation as exact.py and the pinned MLX qmv.
inline void north_complete_down(uint slot,uint first,uint lane,
 coherent(device) device const bfloat* hidden,device const uint* ids,
 device const uint* down,device const bfloat* ds,device const bfloat* db,
 device bfloat* routed) {
 uint expert=ids[slot];float acc[4]={0};
 for(uint k=lane*8;k<768;k+=256) {
  coherent(device) device const bfloat* xp=hidden+slot*768+k;
  float v[8],xsum=0;
  for(uint j=0;j<8;j+=4) {
   xsum+=float(xp[j]+xp[j+1]+xp[j+2]+xp[j+3]);
   v[j]=float(xp[j]);v[j+1]=float(xp[j+1])/16.0f;
   v[j+2]=float(xp[j+2])/256.0f;v[j+3]=float(xp[j+3])/4096.0f;
  }
  for(uint rr=0;rr<4;rr++) {
   uint row=expert*2048+first+rr,si=row*12+k/64;
   device const ushort* w=(device const ushort*)(down+row*96+k/8);
   float sum=0;
   for(uint j=0;j<2;j++)sum+=(v[j*4]*(w[j]&15)+v[j*4+1]*(w[j]&240)+v[j*4+2]*(w[j]&3840)+v[j*4+3]*(w[j]&61440));
   acc[rr]+=float(ds[si])*sum+float(db[si])*xsum;
  }
 }
 for(uint rr=0;rr<4;rr++) {
  float value=simd_sum(acc[rr]);
  if(lane==0)routed[slot*2048+first+rr]=bfloat(value);
 }
}
