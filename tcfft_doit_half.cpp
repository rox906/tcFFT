#include "tcfft_half.h"
int *rev, N, N_batch;
half *in_host, *in_device_0;
tcfftHandle plan;

void gen_rev(int N, int rev[], int radices[], int n_radices)
{
    int *tmp_0 = (int *)malloc(sizeof(int) * N);
    int *tmp_1 = (int *)malloc(sizeof(int) * N);
    int now_N = N;
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        tmp_0[i] = i;
    for (int i = n_radices - 1; i >= 0; --i)
    {
#pragma omp parallel for
        for (int j = 0; j < N; j += now_N)
            for (int k = 0; k < radices[i]; ++k)
                for (int l = 0; l < now_N / radices[i]; ++l)
                {
                    tmp_1[j + l + k * (now_N / radices[i])] = tmp_0[j + l * radices[i] + k];
                }
        now_N /= radices[i];
        std::swap(tmp_0, tmp_1);
    }
#pragma omp parallel for
    for (int i = 0; i < N; ++i)
        rev[i] = tmp_0[i];
}

void setup(double *data, int n, int n_batch)
{
    N = n;
    N_batch = n_batch;
    tcfftCreate(&plan, N, N_batch);
    // in_host
    rev = (int *)malloc(sizeof(int) * N);
    gen_rev(N, rev, plan.radices, plan.n_radices);
    in_host = (half *)malloc(sizeof(half) * 2 * N * N_batch);
#pragma omp parallel for
    for (int j = 0; j < N_batch; ++j)
        for (int i = 0; i < N; ++i)
        {
            in_host[2 * N * j + 2 * i + 0] = data[2 * N * j + 2 * rev[i] + 0];
            in_host[2 * N * j + 2 * i + 1] = data[2 * N * j + 2 * rev[i] + 1];
        }
    cudaMalloc(&in_device_0, sizeof(half) * N * 2 * N_batch);
    cudaMemcpy(in_device_0, in_host, sizeof(half) * N * 2 * N_batch, cudaMemcpyHostToDevice);
}

void finalize(double *result)
{
    cudaMemcpy(in_host, in_device_0, sizeof(half) * N * 2 * N_batch, cudaMemcpyDeviceToHost);
#pragma omp paralllel for
    for (int j = 0; j < N_batch; ++j)
        for (int i = 0; i < N; ++i)
        {
            result[0 + i * 2 + 2 * N * j] = in_host[2 * i + 0 + 2 * N * j];
            result[1 + i * 2 + 2 * N * j] = in_host[2 * i + 1 + 2 * N * j];
        }
}

void doit(int iter)
{
    for (int t = 0; t < iter; ++t)
        tcfftExec(plan, in_device_0);
    cudaDeviceSynchronize();
}