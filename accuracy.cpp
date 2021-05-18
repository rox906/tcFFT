#include <fftw3.h>
#include <cstdio>
#include <unistd.h>
#include <cstring>
#include <cmath>
#include <algorithm>
#include "doit.h"
extern char *optarg;
extern int optopt;

void fftw3_get_result(double *data, double *result, int n, int n_batch)
{
    fftw_complex *in = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * n);
    fftw_plan p = fftw_plan_dft_1d(n, in, in, FFTW_FORWARD, FFTW_ESTIMATE);
    for (int i = 0; i < n_batch; ++i)
    {
        memcpy(in, data + 2 * i * n, sizeof(fftw_complex) * n);
        fftw_execute(p);
        memcpy(result + 2 * i * n, in, sizeof(fftw_complex) * n);
    }
    fftw_destroy_plan(p);
    fftw_free(in);
}

double get_error(double *tested, double *standard, int n, int n_batch)
{
    double error = 0;
    for (int i = 0; i < n_batch; ++i)
#pragma omp parallel for reduction(+ \
                                   : error)
        for (int j = 0; j < n; ++j)
        {
            double tested_e = tested[0 + j * 2 + i * n * 2];
            double standard_e = standard[0 + j * 2 + i * n * 2];
            error += std::min(1.0, std::abs((tested_e - standard_e) / standard_e));
            tested_e = tested[1 + j * 2 + i * n * 2];
            standard_e = standard[1 + j * 2 + i * n * 2];
            error += std::min(1.0, std::abs((tested_e - standard_e) / standard_e));
        }
    return error / n / n_batch;
}

int main(int argc, char *argv[])
{
    int n = 65536, n_batch = 1, seed = 42;
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "n:b:s:")))
    {
        switch (opt_c)
        {
        case 'n':
            n = atoi(optarg);
            break;
        case 'b':
            n_batch = atoi(optarg);
            break;
        case 's':
            seed = atoi(optarg);
            break;
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    double *data = (double *)malloc(sizeof(double) * n * n_batch * 2);
    generate_data(data, n, n_batch, seed);

    double *standard = (double *)malloc(sizeof(double) * n * n_batch * 2);
    fftw3_get_result(data, standard, n, n_batch);

    double *tested = (double *)malloc(sizeof(double) * n * n_batch * 2);
    setup(data, n, n_batch);
    doit(1);
    finalize(tested);

    printf("%e\n", get_error(tested, standard, n, n_batch));

    return 0;
}