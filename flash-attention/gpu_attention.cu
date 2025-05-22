#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <math.h>

__global__ void matrixMultiply(
    const float* A,
    const float* B,
    float* C,
    int A_rows,
    int A_cols,
    int B_cols
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols) {
        float sum = 0.0f;
        for (int k = 0; k < A_cols; ++k) {
            sum += A[row * A_cols + k] * B[k * B_cols + col];
        }
        C[row * B_cols + col] = sum;
    }
}

__global__ void scaleMatrix(float* matrix, int n, float scale_factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < n && idy < n) {
        matrix[idy * n + idx] /= scale_factor;
    }
}

__global__ void rowMax(const float* input, float* output, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float max_val = -INFINITY;
    for (int i = 0; i < n; i++) {
        max_val = fmaxf(max_val, input[row * n + i]);
    }
    output[row] = max_val;
}

__global__ void softmaxSumExp(float* input, const float* max_vals, float* sums, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;

    float sum = 0.0f;
    float max_val = max_vals[row];

    for (int i = 0; i < n; i++) {
        input[row * n + i] = expf(input[row * n + i] - max_val);
        sum += input[row * n + i];
    }
    sums[row] = sum;
}

__global__ void softmaxNormalize(float* input, const float* sums, int n) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < n && col < n) {
        input[row * n + col] /= sums[row];
    }
}

extern "C" void gpu_attention(
    float* Q,
    float* K,
    float* V,
    float* output,
    int n,
    int d
) {
    float *d_Q, *d_K, *d_V, *d_output;

    cudaMalloc((void**)&d_Q, n * d * sizeof(float));
    cudaMalloc((void**)&d_K, n * d * sizeof(float));
    cudaMalloc((void**)&d_V, n * d * sizeof(float));
    cudaMalloc((void**)&d_output, n * d * sizeof(float));

    cudaMemcpy(d_Q, Q, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, K, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, V, n * d * sizeof(float), cudaMemcpyHostToDevice);

    const int BLOCK_SIZE = 16;
    float* d_QK, * d_softmax, * d_max, * d_sum;

    cudaMalloc(&d_QK, n * n * sizeof(float));
    cudaMalloc(&d_softmax, n * n * sizeof(float));
    cudaMalloc(&d_max, n * sizeof(float));
    cudaMalloc(&d_sum, n * sizeof(float));

    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((n + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiply <<<grid, threads>>> (d_Q, d_K, d_QK, n, d, n);

    float scale_factor = sqrtf(d);
    scaleMatrix <<<grid, threads>>> (d_QK, n, scale_factor);
    cudaMemcpy(d_softmax, d_QK, n * n * sizeof(float), cudaMemcpyDeviceToDevice);

    rowMax <<<grid, threads>>> (d_softmax, d_max, n);
    softmaxSumExp <<<grid, threads>>> (d_softmax, d_max, d_sum, n);
    softmaxNormalize <<<grid, threads>>> (d_softmax, d_sum, n);

    dim3 grid_AV((d + BLOCK_SIZE - 1) / BLOCK_SIZE, (n + BLOCK_SIZE - 1) / BLOCK_SIZE);
    matrixMultiply <<<grid_AV, threads>>> (d_softmax, d_V, d_output, n, n, d);

    cudaMemcpy(output, d_output, n * d * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_QK);
    cudaFree(d_softmax);
    cudaFree(d_max);
    cudaFree(d_sum);
    cudaFree(d_output);
}
