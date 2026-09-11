#include <metal_stdlib>
using namespace metal;
constant uint ND=2048,NH=768,NA=4096,NE=8,NC=768/NCHUNK,NT=256;
inline void north_hidden(uint job,uint sg,uint lane,
 device const bfloat* x,device const uint* ids,
 device const uint* up,device const bfloat* us,device const bfloat* ub,
 device const uint* gate,device const bfloat* gs,device const bfloat* gb,
 device bfloat* hidden,threadgroup float* tile) {
 uint slot=job/NC,expert=ids[slot],chunk=job%NC;
 // Match MLX's affine-qmv grouping, including BF16 input-sum rounding.
 // Algorithm adapted from Apple's MLX quantized.h (MIT); see NOTICE.md.
 for(uint rr=sg*4;rr<NCHUNK;rr+=32) {
  float uu[4]={0},gg[4]={0};
  for(uint k=lane*16;k<ND;k+=512) {
   float v[16],xsum=0;
   for(uint j=0;j<16;j+=4) {
    xsum+=float(x[k+j]+x[k+j+1]+x[k+j+2]+x[k+j+3]);
    v[j]=float(x[k+j]);v[j+1]=float(x[k+j+1])/16.0f;
    v[j+2]=float(x[k+j+2])/256.0f;v[j+3]=float(x[k+j+3])/4096.0f;
   }
   for(uint rowoff=0;rowoff<4;rowoff++) {
    uint row=expert*NH+chunk*NCHUNK+rr+rowoff;
    uint si=row*(ND/64)+k/64;
    device const ushort* uw=(device const ushort*)(up+row*(ND/8)+k/8);
    device const ushort* gw=(device const ushort*)(gate+row*(ND/8)+k/8);
    float ud=0,gd=0;
    for(uint j=0;j<4;j++) {
     ud+=(v[j*4]*(uw[j]&15)+v[j*4+1]*(uw[j]&240)+v[j*4+2]*(uw[j]&3840)+v[j*4+3]*(uw[j]&61440));
     gd+=(v[j*4]*(gw[j]&15)+v[j*4+1]*(gw[j]&240)+v[j*4+2]*(gw[j]&3840)+v[j*4+3]*(gw[j]&61440));
    }
    uu[rowoff]+=float(us[si])*ud+float(ub[si])*xsum;
    gg[rowoff]+=float(gs[si])*gd+float(gb[si])*xsum;
   }
  }
  for(uint rowoff=0;rowoff<4;rowoff++) {
   float u=simd_sum(uu[rowoff]),g=simd_sum(gg[rowoff]);
   if(lane==0) {
    bfloat u16=bfloat(u),g16=bfloat(g);
    auto z=1/(1+exp(abs(g16)));
    bfloat sigmoid16=(g16<0)?z:1-z;
    bfloat a16=g16*sigmoid16;
    bfloat h=bfloat(float(u16)*float(a16));
    hidden[slot*NH+chunk*NCHUNK+rr+rowoff]=h;tile[rr+rowoff]=float(h);
   }
  }
 }
}
inline void north_down(uint job,uint tid,device const uint* ids,
 device const uint* down,device const bfloat* ds,device const bfloat* db,
 threadgroup const float* tile,device float* partial) {
 uint expert=ids[job/NC],chunk=job%NC;
 for(uint r=tid;r<ND;r+=256) {
  uint row=expert*ND+r;
  float value=0;
  for(uint block=0;block<NCHUNK;block+=64) {
   float sum=0,xsum=0;
   for(uint word=block/8;word<min(block+64,uint(NCHUNK))/8;word++) {
    uint w=down[row*(NH/8)+chunk*(NCHUNK/8)+word];
    for(uint j=0;j<8;j++) {
     float v=tile[word*8+j];
     sum+=float((w>>(j*4))&15)*v;
    }
   }
   for(uint k=block;k<min(block+64,uint(NCHUNK));k+=4)
    xsum+=float(bfloat(tile[k])+bfloat(tile[k+1])+bfloat(tile[k+2])+bfloat(tile[k+3]));
   uint si=row*(NH/64)+(chunk*NCHUNK+block)/64;
   value+=float(ds[si])*sum+float(db[si])*xsum;
  }
  partial[job*ND+r]=value;
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
