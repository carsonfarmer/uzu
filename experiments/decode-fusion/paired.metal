// Isolated prototype: batch=1, BF16 activations, unsigned W4/group32,
// output RHT, SiLU(alpha=1), input RHT, BF16 output. No clipping or bias.
// Uses upstream GEMV accumulation/reduction and activation functions verbatim.
#include "generated/gemm.h"
#include "matmul/gemv/common/quant_b_source.h"
#include "matmul/gemv/common/reduce.h"
#include "matmul/gemv/common/epilogue.h"
#include "common/gated_act_mul.h"
using namespace metal;
using namespace uzu::gemm;
using Tile = GemvTile<1, 32, 32, 2, 8, 1>;
using Ops = GemvOperands<bfloat, bfloat, bfloat>;

#define ARGS \
 const device uint* weights [[buffer(0)]], \
 const device bfloat* scales [[buffer(1)]], \
 const device uchar* zeros [[buffer(2)]], \
 const device bfloat* input [[buffer(3)]], \
 device bfloat* output [[buffer(4)]], \
 const device int* output_factors [[buffer(5)]], \
 const device int* input_factors [[buffer(6)]], \
 constant uint& k [[buffer(7)]], \
 constant uint& h [[buffer(8)]], \
 uint block [[threadgroup_position_in_grid]], \
 uint lane [[thread_index_in_simdgroup]], \
 uint sg [[simdgroup_index_in_threadgroup]]

kernel void baseline_projection(ARGS) {
  threadgroup float shared[32];
  const Ops ops = {weights, scales, zeros, nullptr, input, output, nullptr, output_factors, nullptr};
  const GemvParams params = {k, 2*h, 1, 1.f, 0.f, GemmDTransform::RHT, false, false};
  auto tile = OutputTile<Tile, true>::make(block, 0, sg, lane, 2*h);
  float result[1][4] = {{0}};
  QuantBSource<Tile,bfloat,bfloat,bfloat,GemmBPrologueKind::ScaleZeroPointDequant,32,4,true,true>::accumulate(result,ops,params,tile);
  Reduce<Tile,true>::run(result,shared,tile);
  Epilogue<Tile,bfloat,bfloat,bfloat,true>::store(result,ops,params,tile,shared);
}

kernel void baseline_gate(ARGS) {
  uint i = block * 128 + sg * 32 + lane;
  if (i >= h) return; // Host requires complete 128-element blocks.
  bfloat value = input[i];
  bfloat gate = input[h+i];
  bfloat activated = activate_silu_alpha(gate, 1.f);
  bfloat gated = value * activated;
  output[i] = bfloat(simdgroup_input_random_hadamard_transform(ushort(lane),float(gated),input_factors[i]));
}

kernel void paired_projection_gate(ARGS) {
  threadgroup float shared[64];
  const Ops ops = {weights, scales, zeros, nullptr, input, output, nullptr, output_factors, nullptr};
  const GemvParams params = {k, 2*h, 1, 1.f, 0.f, GemmDTransform::RHT, false, false};
  // Each group owns both corresponding 32-row blocks. No device-wide waits.
  for (uint half_index=0; half_index<2; ++half_index) {
    auto tile = OutputTile<Tile,true>::make(block + half_index*(h/32),0,sg,lane,2*h);
    float result[1][4] = {{0}};
    QuantBSource<Tile,bfloat,bfloat,bfloat,GemmBPrologueKind::ScaleZeroPointDequant,32,4,true,true>::accumulate(result,ops,params,tile);
    Reduce<Tile,true>::run(result,shared,tile);
    if (lane == 0) {
      for (uint r=0;r<4;++r) shared[half_index*32+tile.local_row+r] = result[0][r];
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint i = block*32+lane;
    // Match the existing epilogue's BF16 rounding before and after output RHT.
    bfloat value = simdgroup_output_random_hadamard_transform(ushort(lane),bfloat(shared[lane]),output_factors[i]);
    bfloat gate = simdgroup_output_random_hadamard_transform(ushort(lane),bfloat(shared[32+lane]),output_factors[h+i]);
    bfloat activated = activate_silu_alpha(gate,1.f);
    bfloat gated = value * activated;
    output[i] = bfloat(simdgroup_input_random_hadamard_transform(ushort(lane),float(gated),input_factors[i]));
  }
}

// Same ownership, but value and gate each use four of the eight SIMD groups.
// This trades twice the rows per lane for concurrent evaluation of both halves.
kernel void paired_parallel_projection_gate(ARGS) {
  using PairTile = GemvTile<1,32,32,2,4,1>;
  threadgroup float shared[64];
  const Ops ops = {weights,scales,zeros,nullptr,input,output,nullptr,output_factors,nullptr};
  const GemvParams params = {k,2*h,1,1.f,0.f,GemmDTransform::RHT,false,false};
  const uint half_index = sg/4;
  auto tile = OutputTile<PairTile,true>::make(block+half_index*(h/32),0,sg%4,lane,2*h);
  float result[1][8] = {{0}};
  QuantBSource<PairTile,bfloat,bfloat,bfloat,GemmBPrologueKind::ScaleZeroPointDequant,32,4,true,true>::accumulate(result,ops,params,tile);
  Reduce<PairTile,true>::run(result,shared,tile);
  if (lane == 0) {
    for (uint r=0;r<8;++r) shared[half_index*32+tile.local_row+r] = result[0][r];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (sg == 0) {
    uint i=block*32+lane;
    bfloat value=simdgroup_output_random_hadamard_transform(ushort(lane),bfloat(shared[lane]),output_factors[i]);
    bfloat gate=simdgroup_output_random_hadamard_transform(ushort(lane),bfloat(shared[32+lane]),output_factors[h+i]);
    bfloat activated=activate_silu_alpha(gate,1.f);
    bfloat gated=value*activated;
    output[i]=bfloat(simdgroup_input_random_hadamard_transform(ushort(lane),float(gated),input_factors[i]));
  }
}
