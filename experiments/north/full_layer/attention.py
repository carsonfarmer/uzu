"""Exact MLX-style one-pass batch-one GQA attention as a callable Metal kernel."""

import mlx.core as mx


source = r'''
uint head=threadgroup_position_in_grid.x;
uint sg=simdgroup_index_in_threadgroup,lane=thread_index_in_simdgroup;
uint n=params[0],kh_stride=params[1],ks_stride=params[2];
uint vh_stride=params[3],vs_stride=params[4];
uint kv_head=head/8;
device const bfloat* qp=q+head*128+lane*4;
device const bfloat* kp=k+kv_head*kh_stride+sg*ks_stride+lane*4;
device const bfloat* vp=v+kv_head*vh_stride+sg*vs_stride+lane*4;
threadgroup float outputs[1024];
threadgroup float max_scores[32];
threadgroup float sum_exp_scores[32];
float qv[4],kv[4],ov[4]={0};
for(uint j=0;j<4;j++)qv[j]=0.08838834764831845f*float(qp[j]);
float max_score=-3.402823466e+38f,sum_exp=0;
for(uint i=sg;i<n;i+=32) {
 for(uint j=0;j<4;j++)kv[j]=float(kp[j]);
 float score=0;
 for(uint j=0;j<4;j++)score+=qv[j]*kv[j];
 score=simd_sum(score);
 float next_max=max(max_score,score);
 float factor=fast::exp(max_score-next_max);
 float exp_score=fast::exp(score-next_max);
 max_score=next_max;
 sum_exp=sum_exp*factor+exp_score;
 for(uint j=0;j<4;j++)ov[j]=ov[j]*factor+exp_score*float(vp[j]);
 kp+=32*ks_stride;vp+=32*vs_stride;
}
if(lane==0){max_scores[sg]=max_score;sum_exp_scores[sg]=sum_exp;}
threadgroup_barrier(mem_flags::mem_threadgroup);
max_score=max_scores[lane];
float next_max=simd_max(max_score);
float factor=fast::exp(max_score-next_max);
sum_exp=simd_sum(sum_exp_scores[lane]*factor);
for(uint j=0;j<4;j++) {
 outputs[lane*32+sg]=ov[j];
 threadgroup_barrier(mem_flags::mem_threadgroup);
 ov[j]=simd_sum(outputs[sg*32+lane]*factor);
 ov[j]=sum_exp==0?ov[j]:(ov[j]/sum_exp);
 threadgroup_barrier(mem_flags::mem_threadgroup);
}
if(lane==0)for(uint j=0;j<4;j++)out[head*128+sg*4+j]=bfloat(ov[j]);
'''

kernel = mx.fast.metal_kernel(
    name="north_sdpa_vector_bf16_128",
    input_names=["q", "k", "v", "params"],
    output_names=["out"],
    source=source,
)


def run_attention(q, k, v, length=None):
    assert q.shape == (1, 32, 1, 128)
    assert k.shape[0:2] == (1, 4) and k.shape[-1] == 128
    assert v.shape == k.shape
    length = k.shape[2] if length is None else int(length)
    assert 0 < length < 1024 and length <= k.shape[2]
    # The cache backing arrays are contiguous even when their used prefix is
    # shorter than allocated capacity. Read only `length` sequence positions.
    head_stride = k.shape[2] * 128
    params = mx.array(
        [length, head_stride, 128, head_stride, 128],
        dtype=mx.uint32,
    )
    return kernel(
        inputs=[q, k, v, params],
        grid=(32 * 1024, 1, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[(1, 32, 1, 128)],
        output_dtypes=[mx.bfloat16],
    )[0]
