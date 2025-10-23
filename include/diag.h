#ifndef _DIAG_H
#define _DIAG_H

#include "n2array.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Extract diagonal into a new N2Array* (1 x n or n x 1 vector depending on shape).
 * Caller owns the returned N2Array* and must call N2Array_free().
 */
N2Array* N2Array_diag(const N2Array* A);

/* Build a diagonal N2Array* from a 2D C array (darray) and shape {rows,cols}.
 * If darray is a square matrix, returns a 1 x n N2Array* containing the diagonal.
 * Caller owns the returned N2Array* and must call N2Array_free().
 */
N2Array* N2Array_diag_from_2d(const double** darray, const int* shape);

#ifdef __cplusplus
}
#endif
#endif