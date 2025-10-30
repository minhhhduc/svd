#ifndef NUMC_H
#define NUMC_H
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/n2array.h"

<<<<<<< HEAD
// NumC class for array creation utilities
class NumC {
public:
    static double* arange(double start, double stop, double step = 1.0);
    static double* linspace(double start, double stop, int num = 50);
    static N2Array zeros(int rows, int cols);
    static N2Array ones(int rows, int cols);
    static N2Array diag(const N2Array& A);
    static N2Array diag(const double** darray, int* shape);
    static N2Array dot(const N2Array& a, const N2Array& b);
    static N2Array min(const N2Array& a, int axis=-1);
    static N2Array max(const N2Array& a, int axis=-1);
    static N2Array sum(const N2Array& a, int axis=-1);
    static N2Array mean(const N2Array& a, int axis=-1);
    static N2Array sd(const N2Array& a, int axis=-1);
    // int* shape(const int* darray);
    // int* argsort(const double* darray, int size, int axis=-1);
    // int* argsort(N2Array arr, int axis=-1);
};

// Non-templated convenience functions operating on double arrays

=======
/* Create a contiguous 1-D array of doubles: [start, start+step, ..., <stop]
 * Returns a newly allocated double* or NULL on invalid range. Caller frees.
 */
double* numc_arange(double start, double stop, double step);
/* Create `num` evenly spaced values from start to stop inclusive. Caller frees. */
double* numc_linspace(double start, double stop, int num);

/* Create a new N2Array* filled with zeros. Caller owns and must free. */
N2Array* numc_zeros(int rows, int cols);
N2Array* numc_dot(const N2Array* a, const N2Array* b);
N2Array* numc_min(const N2Array* a, int axis);
N2Array* numc_max(const N2Array* a, int axis);
N2Array* numc_sum(const N2Array* a, int axis);
N2Array* numc_mean(const N2Array* a, int axis);
N2Array* numc_stdev(const N2Array* a, int axis);
>>>>>>> 9fbc3779f2a5518b7713ec39252b6538d200662a
// N2Array shape(const int* darray);
// N2Array arange(int start, int end, int step=1);
// N2Array arange(int end, int step=1);
// N2Array linspace(int start, int end, int num=50);
#endif