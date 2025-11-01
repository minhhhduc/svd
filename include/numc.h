#ifndef NUMC_H
#define NUMC_H
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../include/n2array.h"

/* Create a contiguous 1-D array of doubles: [start, start+step, ..., <stop]
 * Returns a newly allocated double* or NULL on invalid range. Caller frees.
 */
double* arange(double start, double stop, double step);
/* Create `num` evenly spaced values from start to stop inclusive. Caller frees. */
double* linspace(double start, double stop, int num);

/* Create a new N2Array* filled with zeros. Caller owns and must free. */
N2Array* zeros(int rows, int cols);
N2Array* ones(int rows, int cols);
N2Array* dot(const N2Array* a, const N2Array* b);
N2Array* min(const N2Array* a, int axis);
N2Array* max(const N2Array* a, int axis);
N2Array* sum(const N2Array* a, int axis);
N2Array* mean(const N2Array* a, int axis);
N2Array* stdev(const N2Array* a, int axis);
N2Array* transpose(const N2Array* a);
N2Array* diag(const N2Array* a); // extract diagonal or create diagonal matrix
pair* eig(const N2Array* a); // reduced norm jacobi method
pair* eigh(const N2Array* a); // jacobi method for symmetric matrices

#endif