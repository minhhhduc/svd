#include <stdlib.h>
#include <string.h>
#include "../include/n2array.h"
#include <stdio.h>
#include <math.h>
#include <omp.h>

void pair_free(pair* p) {
    if (!p) return;
    if (p->first) N2Array_free(p->first);
    if (p->second) N2Array_free(p->second);
    free(p);
}

N2Array* N2Array_from_2d(double** darray, const int* shape) {
    if (!darray || !shape) return NULL;
    int rows = shape[0];
    int cols = shape[1];
    if (rows <= 0 || cols <= 0) return NULL;

    N2Array* arr = (N2Array*)malloc(sizeof(N2Array));
    if (!arr) return NULL;

    double* data = (double*)malloc(sizeof(double) * (size_t)(rows * cols));
    if (!data) { free(arr); return NULL; }

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        memcpy(data + (size_t)i * cols, darray[i], sizeof(double) * (size_t)cols);
    }

    double** rows_ptr = (double**)malloc(sizeof(double*) * (size_t)rows);
    if (!rows_ptr) { free(data); free(arr); return NULL; }

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        rows_ptr[i] = data + (size_t)i * cols;
    }

    int* shape_copy = (int*)malloc(sizeof(int) * 2);
    if (!shape_copy) { free(rows_ptr); free(data); free(arr); return NULL; }
    shape_copy[0] = rows;
    shape_copy[1] = cols;

    arr->n2array = rows_ptr;
    arr->n1array = data;
    arr->shape = shape_copy;

    return arr;
}

N2Array* N2Array_from_1d(double* darray, const int* shape) {
    if (!darray || !shape) return NULL;
    int rows = shape[0];
    int cols = shape[1];
    if (rows <= 0 || cols <= 0) return NULL;

    N2Array* arr = (N2Array*)malloc(sizeof(N2Array));
    if (!arr) return NULL;

    double** rows_ptr = (double**)malloc(sizeof(double*) * (size_t)rows);
    if (!rows_ptr) { free(arr); return NULL; }

    #pragma omp parallel for
    for (int i = 0; i < rows; ++i) {
        rows_ptr[i] = darray + (size_t)i * cols;
    }

    int* shape_copy = (int*)malloc(sizeof(int) * 2);
    if (!shape_copy) { free(rows_ptr); free(arr); return NULL; }
    shape_copy[0] = rows;
    shape_copy[1] = cols;

    arr->n2array = rows_ptr;
    arr->n1array = darray;
    arr->shape = shape_copy;

    return arr;
}

N2Array* N2Array_copy(const N2Array* other) {
    if (!other || !other->shape) return NULL;
    int rows = other->shape[0];
    int cols = other->shape[1];
    if (rows <= 0 || cols <= 0) return NULL;

    /* allocate contiguous buffer and rows */
    double* data = (double*)malloc(sizeof(double) * (size_t)rows * cols);
    if (!data) return NULL;
    double** rows_ptr = (double**)malloc(sizeof(double*) * (size_t)rows);
    if (!rows_ptr) { free(data); return NULL; }

    /* copy element-wise using available storage */
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            double v = other->n2array ? other->n2array[i][j] : other->n1array[i * cols + j];
            data[(size_t)i * cols + j] = v;
        }
        rows_ptr[i] = data + (size_t)i * cols;
    }

    int* shape_copy = (int*)malloc(sizeof(int) * 2);
    if (!shape_copy) { free(rows_ptr); free(data); return NULL; }
    shape_copy[0] = rows; shape_copy[1] = cols;

    N2Array* arr = (N2Array*)malloc(sizeof(N2Array));
    if (!arr) { free(shape_copy); free(rows_ptr); free(data); return NULL; }
    arr->n2array = rows_ptr;
    arr->n1array = data;
    arr->shape = shape_copy;
    return arr;
}

void N2Array_free(N2Array* a) {
    if (!a) return;
    if (a->n2array) {
        free(a->n2array);
        a->n2array = NULL;
    }
    if (a->n1array) {
        free((void*)a->n1array);
        a->n1array = NULL;
    }
    if (a->shape) {
        free((void*)a->shape);
        a->shape = NULL;
    }
    free(a);
}

double N2Array_get(const N2Array* a, int r, int c) {
    if (!a || !a->shape) return 0.0;
    int rows = a->shape[0];
    int cols = a->shape[1];
    if (r < 0 || r >= rows || c < 0 || c >= cols) return 0.0;
    if (a->n2array) return a->n2array[r][c];
    return a->n1array[(size_t)r * cols + c];
}

void N2Array_set(N2Array* a, int r, int c, double v) {
    if (!a || !a->shape) return;
    int rows = a->shape[0];
    int cols = a->shape[1];
    if (r < 0 || r >= rows || c < 0 || c >= cols) return;
    if (a->n2array) a->n2array[r][c] = v;
    else a->n1array[(size_t)r * cols + c] = v;
}

double** N2Array_to_array(N2Array* a) {
    if (!a) return NULL;
    return a->n2array;
}

double* N2Array_data(N2Array* a) {
    if (!a) return NULL;
    return a->n1array;
}

static N2Array* n2array_alloc_empty(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    double* data = (double*)malloc(sizeof(double) * (size_t)rows * cols);
    if (!data) return NULL;
    double** rows_ptr = (double**)malloc(sizeof(double*) * (size_t)rows);
    if (!rows_ptr) { free(data); return NULL; }
    for (int i = 0; i < rows; ++i) rows_ptr[i] = data + (size_t)i * cols;
    int* shape_copy = (int*)malloc(sizeof(int) * 2);
    if (!shape_copy) { free(rows_ptr); free(data); return NULL; }
    shape_copy[0] = rows; shape_copy[1] = cols;
    N2Array* arr = (N2Array*)malloc(sizeof(N2Array));
    if (!arr) { free(shape_copy); free(rows_ptr); free(data); return NULL; }
    arr->n2array = rows_ptr;
    arr->n1array = data;
    arr->shape = shape_copy;
    return arr;
}

N2Array* N2Array_add(const N2Array* A, const N2Array* B) {
    if (!A || !B || !A->shape || !B->shape) return NULL;
    int a_rows = A->shape[0], a_cols = A->shape[1];
    int b_rows = B->shape[0], b_cols = B->shape[1];
    if (a_rows != b_rows || a_cols != b_cols) return NULL; /* no broadcasting */
    N2Array* out = n2array_alloc_empty(a_rows, a_cols);
    if (!out) return NULL;
    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < a_cols; ++j) {
            double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * a_cols + j];
            double vb = B->n2array ? B->n2array[i][j] : B->n1array[(size_t)i * b_cols + j];
            out->n1array[(size_t)i * a_cols + j] = va + vb;
        }
    }
    return out;
}

N2Array* N2Array_sub(const N2Array* A, const N2Array* B) {
    if (!A || !B || !A->shape || !B->shape) return NULL;
    int a_rows = A->shape[0], a_cols = A->shape[1];
    int b_rows = B->shape[0], b_cols = B->shape[1];
    if (a_rows != b_rows || a_cols != b_cols) return NULL;
    N2Array* out = n2array_alloc_empty(a_rows, a_cols);
    if (!out) return NULL;
    for (int i = 0; i < a_rows; ++i) for (int j = 0; j < a_cols; ++j) {
        double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * a_cols + j];
        double vb = B->n2array ? B->n2array[i][j] : B->n1array[(size_t)i * b_cols + j];
        out->n1array[(size_t)i * a_cols + j] = va - vb;
    }
    return out;
}

N2Array* N2Array_mul(const N2Array* A, const N2Array* B) {
    if (!A || !B || !A->shape || !B->shape) return NULL;
    int a_rows = A->shape[0], a_cols = A->shape[1];
    int b_rows = B->shape[0], b_cols = B->shape[1];
    if (a_rows != b_rows || a_cols != b_cols) return NULL;
    N2Array* out = n2array_alloc_empty(a_rows, a_cols);
    if (!out) return NULL;
    for (int i = 0; i < a_rows; ++i) for (int j = 0; j < a_cols; ++j) {
        double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * a_cols + j];
        double vb = B->n2array ? B->n2array[i][j] : B->n1array[(size_t)i * b_cols + j];
        out->n1array[(size_t)i * a_cols + j] = va * vb;
    }
    return out;
}

N2Array* N2Array_div(const N2Array* A, const N2Array* B) {
    if (!A || !B || !A->shape || !B->shape) return NULL;
    int a_rows = A->shape[0], a_cols = A->shape[1];
    int b_rows = B->shape[0], b_cols = B->shape[1];
    if (a_rows != b_rows || a_cols != b_cols) return NULL;
    N2Array* out = n2array_alloc_empty(a_rows, a_cols);
    if (!out) return NULL;
    for (int i = 0; i < a_rows; ++i) for (int j = 0; j < a_cols; ++j) {
        double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * a_cols + j];
        double vb = B->n2array ? B->n2array[i][j] : B->n1array[(size_t)i * b_cols + j];
        out->n1array[(size_t)i * a_cols + j] = vb == 0.0 ? 0.0 : va / vb;
    }
    return out;
}

N2Array* N2Array_add_scalar(const N2Array* A, double scalar) {
    if (!A || !A->shape) return NULL;
    int rows = A->shape[0], cols = A->shape[1];
    N2Array* out = n2array_alloc_empty(rows, cols);
    if (!out) return NULL;
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * cols + j];
        out->n1array[(size_t)i * cols + j] = va + scalar;
    }
    return out;
}

N2Array* N2Array_sub_scalar(const N2Array* A, double scalar) {
    if (!A || !A->shape) return NULL;
    int rows = A->shape[0], cols = A->shape[1];
    N2Array* out = n2array_alloc_empty(rows, cols);
    if (!out) return NULL;
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * cols + j];
        out->n1array[(size_t)i * cols + j] = va - scalar;
    }
    return out;
}

N2Array* N2Array_mul_scalar(const N2Array* A, double scalar) {
    if (!A || !A->shape) return NULL;
    int rows = A->shape[0], cols = A->shape[1];
    N2Array* out = n2array_alloc_empty(rows, cols);
    if (!out) return NULL;
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * cols + j];
        out->n1array[(size_t)i * cols + j] = va * scalar;
    }
    return out;
}

N2Array* N2Array_div_scalar(const N2Array* A, double scalar) {
    if (!A || !A->shape) return NULL;
    if (scalar == 0.0) return NULL;
    int rows = A->shape[0], cols = A->shape[1];
    N2Array* out = n2array_alloc_empty(rows, cols);
    if (!out) return NULL;
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * cols + j];
        out->n1array[(size_t)i * cols + j] = va / scalar;
    }
    return out;
}

N2Array* N2Array_transpose(const N2Array* A) {
    if (!A || !A->shape) return NULL;
    int rows = A->shape[0], cols = A->shape[1];
    N2Array* out = n2array_alloc_empty(cols, rows);
    if (!out) return NULL;
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        double v = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * cols + j];
        out->n1array[(size_t)j * rows + i] = v;
    }
    return out;
}

bool N2Array_equals(const N2Array* A, const N2Array* B) {
    if (!A || !B || !A->shape || !B->shape) return false;
    int a_rows = A->shape[0], a_cols = A->shape[1];
    int b_rows = B->shape[0], b_cols = B->shape[1];
    if (a_rows != b_rows || a_cols != b_cols) return false;
    for (int i = 0; i < a_rows; ++i) for (int j = 0; j < a_cols; ++j) {
        double va = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * a_cols + j];
        double vb = B->n2array ? B->n2array[i][j] : B->n1array[(size_t)i * b_cols + j];
        if (va != vb) return false;
    }
    return true;
}

N2Array* N2Array_row_copy(const N2Array* A, int row_index) {
    if (!A || !A->shape) return NULL;
    int rows = A->shape[0], cols = A->shape[1];
    if (row_index < 0 || row_index >= rows) return NULL;
    N2Array* out = n2array_alloc_empty(1, cols);
    if (!out) return NULL;
    for (int j = 0; j < cols; ++j) {
        double v = A->n2array ? A->n2array[row_index][j] : A->n1array[(size_t)row_index * cols + j];
        out->n1array[j] = v;
    }
    return out;
}

double N2Array_element(const N2Array* A, int row_index, int col_index) {
    return N2Array_get(A, row_index, col_index);
}

char* N2Array_to_string(const N2Array* A) {
    if (!A || !A->shape) {
        char* s = (char*)malloc(8);
        if (s) strcpy(s, "null");
        return s;
    }
    int rows = A->shape[0], cols = A->shape[1];
    /* estimate buffer: 32 chars per element + extras */
    size_t estimate = (size_t)rows * cols * 32 + rows * 4 + 32;
    char* buf = (char*)malloc(estimate);
    if (!buf) return NULL;
    char* p = buf;
    size_t remain = estimate;
    int wrote = snprintf(p, remain, "[");
    p += wrote; remain -= wrote;
    for (int i = 0; i < rows; ++i) {
        wrote = snprintf(p, remain, "["); p += wrote; remain -= wrote;
        for (int j = 0; j < cols; ++j) {
            double v = A->n2array ? A->n2array[i][j] : A->n1array[(size_t)i * cols + j];
            wrote = snprintf(p, remain, "%g", v); p += wrote; remain -= wrote;
            if (j < cols - 1) { wrote = snprintf(p, remain, ", "); p += wrote; remain -= wrote; }
        }
        wrote = snprintf(p, remain, "]"); p += wrote; remain -= wrote;
        if (i < rows - 1) { wrote = snprintf(p, remain, ",\n "); p += wrote; remain -= wrote; }
    }
    snprintf(p, remain, "]");
    return buf;
}

bool N2Array_shape_equal(const N2Array* A, const N2Array* B) {
    if (!A || !B || !A->shape || !B->shape) return false;
    return A->shape[0] == B->shape[0] && A->shape[1] == B->shape[1];
}

