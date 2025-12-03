#ifndef OTHER_OPERATING_PARALLEL_H
#define OTHER_OPERATING_PARALLEL_H

double** transpose_parallel(double** A, int rows, int cols, int num_threads);
double* argsort_parallel(double* A, int n, int num_threads);
double* square_parallel(double* A, int n, int num_threads);

#endif
