#include <cstdlib>
void setup(double *data, int n, int n_batch);
void doit(int iter);
void finalize(double *result);
// const int PRIME = 100003;

void generate_data(double *data, int n, int n_batch, int seed = 42)
{
    srand(seed);
    for (int i = 0; i < n_batch; ++i)
        // #pragma omp parallel for
        for (int j = 0; j < n; ++j)
        {
            // data[0 + j * 2 + i * n * 2] = 0.0001f * (j % PRIME) / PRIME;
            // data[1 + j * 2 + i * n * 2] = 0.0001f * (j % PRIME) / PRIME;
            data[0 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
            data[1 + j * 2 + i * n * 2] = 0.0001f * rand() / RAND_MAX;
        }
}