#ifndef N2ARRAY_H
#define N2ARRAY_H

#include <stdlib.h>
#include <stdbool.h>


typedef struct N2Array {
    double** n2array;    /* per-row pointers, or NULL */
    const int* shape;    /* pointer to two ints: rows, cols (owned by struct) */
    double* n1array;     /* contiguous buffer (row-major), or NULL */
} N2Array;

/* Creation / destruction (functions take ownership of passed buffers) */
N2Array* N2Array_from_2d(double** darray, const int* shape);
N2Array* N2Array_from_1d(double* darray, const int* shape);
N2Array* N2Array_copy(const N2Array* other);
void N2Array_free(N2Array* a);

/* Element access */
double N2Array_get(const N2Array* a, int r, int c); /* returns value */
void N2Array_set(N2Array* a, int r, int c, double v); /* sets value */

/* Conversions */
double** N2Array_to_array(N2Array* a); /* returns internal n2array (may be NULL) */
double* N2Array_data(N2Array* a);      /* returns internal contiguous buffer or NULL */

/* Arithmetic (allocate and return new N2Array*, caller frees) */
N2Array* N2Array_add(const N2Array* A, const N2Array* B);
N2Array* N2Array_sub(const N2Array* A, const N2Array* B);
N2Array* N2Array_mul(const N2Array* A, const N2Array* B);
N2Array* N2Array_div(const N2Array* A, const N2Array* B);

N2Array* N2Array_add_scalar(const N2Array* A, double scalar);
N2Array* N2Array_sub_scalar(const N2Array* A, double scalar);
N2Array* N2Array_mul_scalar(const N2Array* A, double scalar);
N2Array* N2Array_div_scalar(const N2Array* A, double scalar);

/* Transpose */
N2Array* N2Array_transpose(const N2Array* A);

/* Comparisons */
bool N2Array_equals(const N2Array* A, const N2Array* B);

/* Index helpers */
N2Array* N2Array_row_copy(const N2Array* A, int row_index); /* returns 1 x cols copy */
double N2Array_element(const N2Array* A, int row_index, int col_index);

/* Debug / string (caller frees returned string with free()) */
char* N2Array_to_string(const N2Array* A);

/* Utility: check shape equality */
bool N2Array_shape_equal(const N2Array* A, const N2Array* B);

#endif /* N2ARRAY_H */