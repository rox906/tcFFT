#include "tcfft_half_2d.h"
int *rev_x, *rev_y, Nx, Ny, N_batch;
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

void setup_2d(double *data, int nx, int ny, int n_batch)
{
    Nx = nx;
    Ny = ny;
    N_batch = n_batch;
    tcfftCreate(&plan, Nx, Ny, N_batch);
 // in_host
    rev_x = (int *)malloc(sizeof(int) * Nx);
    rev_y = (int *)malloc(sizeof(int) * Ny);
    gen_rev(Nx, rev_x, plan.radices_x, plan.n_radices_x);
    gen_rev(Ny, rev_y, plan.radices_y, plan.n_radices_y);
    in_host = (half *)malloc(sizeof(half) * 2 * Nx * Ny * N_batch);
#pragma omp parallel for
    for (int i = 0; i < N_batch; ++i)
        for (int j = 0; j < Nx; ++j)
            for (int k = 0; k < Ny; ++k)
            {
                in_host[2 * (i * Nx * Ny + j * Ny + k) + 0] = data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 0];
                in_host[2 * (i * Nx * Ny + j * Ny + k) + 1] = data[2 * (i * Nx * Ny + rev_x[j] * Ny + rev_y[k]) + 1];
            }
    cudaMalloc(&in_device_0, sizeof(half) * Nx * Ny * N_batch * 2);
    cudaMemcpy(in_device_0, in_host, sizeof(half) * Nx * Ny * N_batch * 2, cudaMemcpyHostToDevice);
}

void finalize_2d(double *result)
{
    cudaMemcpy(in_host, in_device_0, sizeof(half) * Nx * Ny * N_batch * 2, cudaMemcpyDeviceToHost);
#pragma omp paralllel for
    for (int i = 0; i < N_batch * Nx * Ny; ++i)
    {
        result[2 * i + 0] = in_host[2 * i + 0];
        result[2 * i + 1] = in_host[2 * i + 1];
    }
}

void doit_2d(int iter)
{
    for (int t = 0; t < iter; ++t)
        tcfftExec(plan, in_device_0);
    cudaDeviceSynchronize();
}