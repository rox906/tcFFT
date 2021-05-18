#include <unistd.h>
#include <cstring>
#include <cmath>
#include <sys/time.h>
#include <cstdio>
#include "doit.h"
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
    int n = 65536, n_batch = 1, max_times = 1 << 30;
    double t_min = 4;
    char opt_c = 0;
    while (EOF != (opt_c = getopt(argc, argv, "n:b:m:")))
    {
        switch (opt_c)
        {
        case 'n':
            n = atoi(optarg);
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

    double *data = (double *)malloc(sizeof(double) * n * n_batch * 2);
    generate_data(data, n, n_batch);

    double *result = (double *)malloc(sizeof(double) * n * n_batch * 2);

    double run_time;
    int iter;
    setup(data, n, n_batch);
    for (int i = 1; i <= max_times; i <<= 1)
    {
        double t1;
        t1 = gettime();
        doit(i);
        run_time = gettime() - t1;
        iter = i;
        if (run_time > t_min)
            break;
        // printf("%d\n", iter);
    }
    finalize(result);

    printf("n: %d, n_batch: %d, iter: %d, time per iter: %lf\n", n, n_batch, iter, run_time / iter);

    return 0;
}