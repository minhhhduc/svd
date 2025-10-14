#ifndef NUMC_H
#define NUMC_H
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "n2array.h"

// NumC class for array creation utilities
class NumC {
public:
    double* arange(double start, double stop, double step = 1.0);
    double* linspace(double start, double stop, int num = 50);
    N2Array zeros(int rows, int cols);
    N2Array ones(int rows, int cols);
};

// Non-templated convenience functions operating on double arrays
N2Array dot(const N2Array& a, const N2Array& b);
N2Array min(const N2Array& a, int axis=-1);
N2Array max(const N2Array& a, int axis=-1);
N2Array sum(const N2Array& a, int axis=-1);
N2Array mean(const N2Array& a, int axis=-1);
N2Array stdev(const N2Array& a, int axis=-1);
// N2Array shape(const int* darray);
// N2Array arange(int start, int end, int step=1);
// N2Array arange(int end, int step=1);
// N2Array linspace(int start, int end, int num=50);
#endif