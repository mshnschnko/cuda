#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <cuda_fp16.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>

namespace cg = cooperative_groups;

constexpr int BLOCK_SIZE = 128;
constexpr int HEAD_DIM = 64;
constexpr int WARPS_PER_BLOCK = 4;

__global__ void flash_attention_kernel(
    const float* Q,
    const float* K,
    const float* V,
    float* output,
    int n,
    int d,
    float softmax_scale
) {
    constexpr int TILE_SIZE = BLOCK_SIZE / WARPS_PER_BLOCK;

    cg::thread_block blk = cg::this_thread_block();
    cg::thread_block_tile<WARPS_PER_BLOCK> warp = cg::tiled_partition<WARPS_PER_BLOCK>(blk);

    extern __shared__ uint8_t shared_mem[];
    float* Qi = reinterpret_cast<float*>(shared_mem);
    float* Kj = Qi + TILE_SIZE * HEAD_DIM;
    float* Vj = Kj + TILE_SIZE * HEAD_DIM;

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float max_val = -INFINITY;
    float sum = 0.0f;
    float acc[HEAD_DIM / WARPS_PER_BLOCK] = { 0.0f };

    for (int tile = 0; tile < gridDim.x; ++tile) {
        if (row < n && threadIdx.x < TILE_SIZE && tile * TILE_SIZE + threadIdx.x < d) {
            Qi[(threadIdx.y) * HEAD_DIM + threadIdx.x] = Q[row * d + (tile * TILE_SIZE + threadIdx.x)];
        }

        int load_col = tile * TILE_SIZE + threadIdx.y;
        if (load_col < n && threadIdx.x < TILE_SIZE) {
            Kj[threadIdx.y * HEAD_DIM + threadIdx.x] = K[load_col * d + threadIdx.x];
        }

        blk.sync();

        float sum_local = 0.0f;
        for (int k = 0; k < TILE_SIZE; ++k) {
            sum_local += Qi[threadIdx.y * HEAD_DIM + k] * Kj[threadIdx.x * HEAD_DIM + k];
        }
        sum_local *= softmax_scale;

        float max_local = cg::reduce(warp, sum_local, cg::greater<float>());
        float exp_val = expf(sum_local - max_local);
        float sum_exp = cg::reduce(warp, exp_val, cg::plus<float>());

        if (warp.thread_rank() == 0) {
            float old_max = max_val;
            max_val = fmaxf(max_val, max_local);
            sum = sum * expf(old_max - max_val) + sum_exp * expf(max_local - max_val);
        }

        if (load_col < n && threadIdx.x < TILE_SIZE) {
            Vj[threadIdx.y * HEAD_DIM + threadIdx.x] = V[load_col * d + threadIdx.x];
        }

        blk.sync();

        exp_val *= expf(max_local - max_val);
        for (int k = 0; k < HEAD_DIM / WARPS_PER_BLOCK; ++k) {
            acc[k] += exp_val * Vj[threadIdx.x * HEAD_DIM + k * WARPS_PER_BLOCK + warp.thread_rank()];
        }

        blk.sync();
    }

    if (row < n) {
        float scale = 1.0f / sum;
        for (int k = 0; k < HEAD_DIM / WARPS_PER_BLOCK; ++k) {
            acc[k] *= scale;
            output[row * d + k * WARPS_PER_BLOCK + warp.thread_rank()] = acc[k];
        }
    }
}

extern "C" void flash_attention(
    float* Q,
    float* K,
    float* V,
    float* output,
    int n,
    int d
) {
    float *d_Q, *d_K, *d_V, *d_output;

    cudaMalloc(&d_Q, n * d * sizeof(float));
    cudaMalloc(&d_K, n * d * sizeof(float));
    cudaMalloc(&d_V, n * d * sizeof(float));
    cudaMalloc(&d_output, n * d * sizeof(float));

    cudaMemcpy(d_Q, Q, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, n * d * sizeof(float), cudaMemcpyHostToDevice);

    float softmax_scale = 1.0f / sqrtf(d);

    dim3 grid((d + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    dim3 block(BLOCK_SIZE, WARPS_PER_BLOCK);

    size_t shared_mem_size = (2 * BLOCK_SIZE * HEAD_DIM +
        WARPS_PER_BLOCK * HEAD_DIM) * sizeof(float);

    flash_attention_kernel <<<grid, block, shared_mem_size>>> (
        d_Q, d_K, d_V, d_output, n, d, softmax_scale
    );

    cudaMemcpy(output, d_output, n * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_Q);
    cudaFree(d_K);
    cudaFree(d_V);
    cudaFree(d_output);
}