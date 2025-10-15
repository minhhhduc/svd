#ifndef _DIAG_H
#define _DIAG_H
#include "n2array.h"

N2Array diag(const N2Array& A);
N2Array diag(const double** darray, int* shape);

#endif