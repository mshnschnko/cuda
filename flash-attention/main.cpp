#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
#include <chrono>
#include "cpu_attention.h"

extern "C" void gpu_attention(
    float* Q,
    float* K,
    float* V,
    float* output,
    int n,
    int d
);

void print_matrix(float* M, int n, int d) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < d; ++j) {
            std::cout << M[i * d + j] << "  ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

int main()
{
    int n = 2048, d = 1024;
    float* Q = new float[n * d];
    float* K = new float[n * d];
    float* V = new float[n * d];
    float* output_cpu = new float[n * d];
    float* output_gpu = new float[n * d];

    for (int i = 0; i < n * d; ++i) {
        Q[i] = (float)(rand()) / ((float)RAND_MAX / 1000.0f);
        K[i] = (float)(rand()) / ((float)RAND_MAX / 1000.0f);
        V[i] = (float)(rand()) / ((float)RAND_MAX / 1000.0f);
    }

    //for (int i = 0; i < n * d; ++i) {
    //    Q[i] = 2;
    //    K[i] = 20;
    //    V[i] = 36;
    //}

    //print_matrix(Q, n, d);
    //print_matrix(K, n, d);
    //print_matrix(V, n, d);

    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpu_attention(Q, K, V, n, d, output_cpu);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);

    std::cout << "Calculation time (CPU): " << (double)(cpu_duration.count()) / 1000.0 << "sec" << std::endl;

    auto gpu_start = std::chrono::high_resolution_clock::now();
    gpu_attention(Q, K, V, output_gpu, n, d);
    auto gpu_end = std::chrono::high_resolution_clock::now();
    auto gpu_duration = std::chrono::duration_cast<std::chrono::milliseconds>(gpu_end - gpu_start);

    std::cout << "Calculation time (GPU, simple): " << (double)(gpu_duration.count()) / 1000.0 << "sec" << std::endl;

    //print_matrix(output_gpu, n, d);

    bool is_same= true;
    for (int i = 0; i < n * d; ++i) {
        if (fabs(output_cpu[i] - output_gpu[i]) > 0.001) {
            is_same= false;
            break;
        }
    }
    std::cout << "Is correct: " << std::boolalpha << is_same<< std::endl;

    return 0;
}
