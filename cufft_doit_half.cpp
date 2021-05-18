#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cstdlib>
#include <cuda_fp16.h>

half *data_host, *data_device;
cufftHandle plan;
int N, N_batch;

void setup(double *data, int n, int n_batch)
{
    N = n;
    N_batch = n_batch;

    data_host = (half *)malloc(sizeof(half) * N * N_batch * 2);
    for (int i = 0; i < N_batch; ++i)
        for (int j = 0; j < N; ++j)
        {
            data_host[(j + i * N) * 2 + 0] = __float2half((float)data[0 + j * 2 + i * N * 2]);
            data_host[(j + i * N) * 2 + 1] = __float2half((float)data[1 + j * 2 + i * N * 2]);
        }

    cudaMalloc(&data_device, sizeof(half) * N * N_batch * 2);
    cudaMemcpy(data_device, data_host, sizeof(half) * N * N_batch * 2, cudaMemcpyHostToDevice);
    
    long long p_n[1];
    size_t worksize[1];
    p_n[0] = N;
    cufftCreate(&plan);
    cufftXtMakePlanMany(plan, 1, p_n, NULL, 0, 0, CUDA_C_16F, NULL, 0, 0, CUDA_C_16F, N_batch, worksize, CUDA_C_16F);
}

void doit(int iter)
{
    for (int i = 0; i < iter; ++i)
        cufftXtExec(plan, data_device, data_device, CUFFT_FORWARD);
    cudaDeviceSynchronize(); 
}

void finalize(double *result)
{
    cudaMemcpy(data_host, data_device, sizeof(half) * N * N_batch * 2, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N_batch; ++i)
        for (int j = 0; j < N; ++j)
        {
            result[0 + j * 2 + i * N * 2] = __half2float(data_host[(j + i * N) * 2 + 0]);
            result[1 + j * 2 + i * N * 2] = __half2float(data_host[(j + i * N) * 2 + 1]);
        }
}