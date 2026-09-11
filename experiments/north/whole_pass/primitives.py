"""Exact in-kernel RoPE, KV insertion values, and North top-8 routing."""

import mlx.core as mx


_sigmoid_lut = None


def sigmoid_lut():
    global _sigmoid_lut
    if _sigmoid_lut is None:
        bits = mx.arange(65536, dtype=mx.uint32).astype(mx.uint16)
        _sigmoid_lut = mx.sigmoid(bits.view(mx.bfloat16).astype(mx.float32))
        mx.eval(_sigmoid_lut)
    return _sigmoid_lut


source = r'''
uint tid=thread_position_in_grid.x;
uint position=params[0];
bool use_rope=params[1]!=0;
constexpr float rope_log2_base=15.609640474436812f;

for(uint item=tid;item<2048;item+=256) {
 uint pair=item%64,head=item/64,index=head*128+pair*2;
 float x1=float(q[index]),x2=float(q[index+1]);
 if(use_rope) {
  float inv_freq=metal::exp2(-(float(pair)/64.0f)*rope_log2_base);
  float theta=float(position)*inv_freq;
  float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
  q_out[index]=bfloat(x1*c-x2*s);q_out[index+1]=bfloat(x1*s+x2*c);
 } else {q_out[index]=q[index];q_out[index+1]=q[index+1];}
}
for(uint item=tid;item<256;item+=256) {
 uint pair=item%64,head=item/64,index=head*128+pair*2;
 float x1=float(k[index]),x2=float(k[index+1]);
 if(use_rope) {
  float inv_freq=metal::exp2(-(float(pair)/64.0f)*rope_log2_base);
  float theta=float(position)*inv_freq;
  float c=metal::fast::cos(theta),s=metal::fast::sin(theta);
  k_out[index]=bfloat(x1*c-x2*s);k_out[index+1]=bfloat(x1*s+x2*c);
 } else {k_out[index]=k[index];k_out[index+1]=k[index+1];}
}
for(uint item=tid;item<512;item+=256)v_out[item]=v[item];

if(tid==0) {
 float best[8];uint best_ids[8];
 for(uint j=0;j<8;j++){best[j]=-3.402823466e+38f;best_ids[j]=0xffffffffu;}
 for(uint expert=0;expert<128;expert++) {
  device const ushort* router_bits=(device const ushort*)router;
  float score=sigmoid_table[router_bits[expert]];
  uint place=8;
  for(uint j=0;j<8;j++)if(score>best[j]){place=j;break;}
  if(place<8) {
   for(uint j=7;j>place;j--){best[j]=best[j-1];best_ids[j]=best_ids[j-1];}
   best[place]=score;best_ids[place]=expert;
  }
 }
 for(uint j=0;j<8;j++){ids[j]=best_ids[j];scores[j]=best[j];}
}
'''


kernel = mx.fast.metal_kernel(
    name="north_rope_kv_route",
    input_names=["q", "k", "v", "router", "params", "sigmoid_table"],
    output_names=["q_out", "k_out", "v_out", "ids", "scores"],
    source=source,
)


def run_primitives(q, k, v, router, position, use_rope):
    params = mx.array([position, int(use_rope)], dtype=mx.uint32)
    return kernel(
        inputs=[q, k, v, router, params, sigmoid_lut()],
        grid=(256, 1, 1),
        threadgroup=(256, 1, 1),
        output_shapes=[q.shape, k.shape, v.shape, (1, 1, 8), (1, 1, 8)],
        output_dtypes=[mx.bfloat16, mx.bfloat16, mx.bfloat16, mx.uint32, mx.float32],
    )
