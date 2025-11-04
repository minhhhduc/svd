#ifndef NUMC_H
#define NUMC_H
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/n2array.h"

/* Create a contiguous 1-D array of doubles: [start, start+step, ..., <stop]
 * Returns a newly allocated double* or NULL on invalid range. Caller frees.
 */
double* numc_arange(double start, double stop, double step);
/* Create `num` evenly spaced values from start to stop inclusive. Caller frees. */
double* numc_linspace(double start, double stop, int num);

/* Create a new N2Array* filled with zeros. Caller owns and must free. */
N2Array* numc_zeros(int rows, int cols);
/* Create a new N2Array* filled with ones. Caller owns and must free. */
N2Array* numc_ones(int rows, int cols);
N2Array* numc_dot(const N2Array* a, const N2Array* b);
N2Array* numc_min(const N2Array* a, int axis);
N2Array* numc_max(const N2Array* a, int axis);
N2Array* numc_sum(const N2Array* a, int axis);
N2Array* numc_mean(const N2Array* a, int axis);
N2Array* numc_stdev(const N2Array* a, int axis);
// N2Array shape(const int* darray);
// N2Array arange(int start, int end, int step=1);
// N2Array arange(int end, int step=1);
// N2Array linspace(int start, int end, int num=50);
#endif