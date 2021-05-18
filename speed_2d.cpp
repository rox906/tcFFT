#include <unistd.h>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cstdio>
#include "doit_2d.h"
extern char *optarg;
extern int optopt;

double gettime()
{
    timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_usec * 1.0e-6 + tv.tv_sec;
}

int main(int argc, char *argv[])
{
    int nx = 256, ny = 256, n_batch = 1, max_times = 1 << 30;
    double t_min = 4;
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "x:y:b:m:")))
    {
        switch (opt_c)
        {
        case 'x':
            nx = atoi(optarg);
            break;
        case 'y':
            ny = atoi(optarg);
            break;
        case 'b':
            n_batch = atoi(optarg);
            break;
        case 'm':
            max_times = atoi(optarg);
            break;
        case '?':
            printf("unkown option %c\n", optopt);
            break;
        default:
            break;
        }
    }

    double *data = (double *)malloc(sizeof(double) * nx * ny * n_batch * 2);
    generate_data(data, nx * ny, n_batch);

    double *result = (double *)malloc(sizeof(double) * nx * ny * n_batch * 2);

    double run_time;
    int iter;
    setup_2d(data, nx, ny, n_batch);
    for (int i = 1; i <= max_times; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit_2d(i);
        run_time = gettime() - t1;
        iter = i;
        if (run_time > t_min)
            break;
        // printf("%d\n", iter);
    }
    finalize_2d(result);

    printf("nx: %d, ny: %d, n_batch: %d, iter: %d, time per iter: %lf\n", nx, ny, n_batch, iter, run_time / iter);

    return 0;
}