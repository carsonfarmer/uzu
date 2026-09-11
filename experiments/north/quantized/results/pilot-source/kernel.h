#include <metal_stdlib>
using namespace metal;
constant uint ND=2048,NH=768,NA=4096,NE=8,NC=24,NT=256;
inline void north_hidden(uint job,uint sg,uint lane,
 device const bfloat* x,device const uint* ids,
 device const uint* up,device const bfloat* us,device const bfloat* ub,
 device const uint* gate,device const bfloat* gs,device const bfloat* gb,
 device bfloat* hidden,threadgroup float* tile) {
 uint slot=job/NC,expert=ids[slot],chunk=job%NC;
 for(uint rr=sg;rr<32;rr+=8) {
  uint row=(expert*NH+chunk*32+rr);
  float u=0,g=0;
  for(uint k=lane*8;k<ND;k+=256) {
   uint uw=up[row*(ND/8)+k/8],gw=gate[row*(ND/8)+k/8];
   uint si=row*(ND/64)+k/64;
   float xu=0,xg=0,xsum=0;
   for(uint j=0;j<8;j++) {
    float v=float(x[k+j]);xsum+=v;
    xu+=v*float((uw>>(j*4))&15);xg+=v*float((gw>>(j*4))&15);
   }
   u+=float(us[si])*xu+float(ub[si])*xsum;
   g+=float(gs[si])*xg+float(gb[si])*xsum;
  }
  u=simd_sum(u);g=simd_sum(g);
  if(lane==0) {
   bfloat u16=bfloat(u),g16=bfloat(g);
   bfloat a16=bfloat(float(g16)/(1.0f+exp(-float(g16))));
   bfloat h=bfloat(float(u16)*float(a16));
   hidden[slot*NH+chunk*32+rr]=h;tile[rr]=float(h);
  }
 }
}
inline void north_down(uint job,uint tid,device const uint* ids,
 device const uint* down,device const bfloat* ds,device const bfloat* db,
 threadgroup const float* tile,device float* partial) {
 uint expert=ids[job/NC],chunk=job%NC;
 float xsum=0;
 for(uint k=0;k<32;k++)xsum+=tile[k];
 for(uint r=tid;r<ND;r+=256) {
  uint row=expert*ND+r,si=row*(NH/64)+(chunk*32)/64;
  float sum=0;
  for(uint word=0;word<4;word++) {
   uint w=down[row*(NH/8)+chunk*4+word];
   for(uint j=0;j<8;j++)sum+=float((w>>(j*4))&15)*tile[word*8+j];
  }
  partial[job*ND+r]=float(ds[si])*sum+float(db[si])*xsum;
 }
}
inline void north_attention(uint block,uint sg,uint lane,
 device const bfloat* ax,device const bfloat* aw,device bfloat* attention) {
 for(uint rr=sg;rr<32;rr+=8) {
  uint r=block*32+rr;float sum=0;
  for(uint k=lane;k<NA;k+=32)sum+=float(ax[k])*float(aw[r*NA+k]);
  sum=simd_sum(sum);
  if(lane==0)attention[r]=bfloat(sum);
 }
}
