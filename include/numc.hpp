#ifndef NUMC_H
#define NUMC_H
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "n2array.h"

N2Array dot(const N2Array& a, const N2Array& b);
N2Array min(const N2Array& a);
N2Array max(const N2Array& a);
N2Array sum(const N2Array& a);
N2Array mean(const N2Array& a);
N2Array std(const N2Array& a);
// N2Array shape(const int* darray);
// N2Array arange(int start, int end, int step=1);
// N2Array arange(int end, int step=1);
// N2Array linspace(int start, int end, int num=50);
#endif