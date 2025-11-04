/* Pure C implementations of the numc_* helpers using the C N2Array API */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_OPENCL
#include "../include/opencl_helper.h"
#endif
#include "../include/n2array.h"

double* numc_arange(double start, double stop, double step) {
    if (step <= 0.0) return NULL;
    int n = (int)floor((stop - start) / step);
    if (n <= 0) return NULL;
    double* out = (double*)malloc(sizeof(double) * n);
    if (!out) return NULL;
    for (int i = 0; i < n; ++i) out[i] = start + i * step;
    return out;
}

double* numc_linspace(double start, double stop, int num) {
    if (num <= 0) return NULL;
    double* out = (double*)malloc(sizeof(double) * num);
    if (!out) return NULL;
    if (num == 1) {
        out[0] = start;
        return out;
    }
    double step = (stop - start) / (double)(num - 1);
    for (int i = 0; i < num; ++i) out[i] = start + i * step;
    return out;
}

N2Array* numc_zeros(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    /* allocate contiguous buffer and per-row pointers */
    double* data = (double*)calloc((size_t)rows * cols, sizeof(double));
    if (!data) return NULL;
    double** rows_ptr = (double**)malloc(sizeof(double*) * (size_t)rows);
    if (!rows_ptr) { free(data); return NULL; }
    for (int i = 0; i < rows; ++i) rows_ptr[i] = data + (size_t)i * cols;

     int* shape = (int*)malloc(sizeof(int) * 2);
     shape[0] = rows; shape[1] = cols;

     N2Array* arr = (N2Array*)malloc(sizeof(N2Array));
     if (!arr) { free(rows_ptr); free(data); free(shape); return NULL; }
     arr->n2array = rows_ptr;
     arr->n1array = data;
     arr->shape = shape;
     return arr;
}

N2Array* numc_ones(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    double* data = (double*)malloc((size_t)rows * cols * sizeof(double));
    if (!data) return NULL;
    for (int i = 0; i < rows * cols; ++i) data[i] = 1.0;
    double** rows_ptr = (double**)malloc(sizeof(double*) * (size_t)rows);
    if (!rows_ptr) { free(data); return NULL; }
    for (int i = 0; i < rows; ++i) rows_ptr[i] = data + (size_t)i * cols;
    int* shape = (int*)malloc(sizeof(int) * 2);
    shape[0] = rows; shape[1] = cols;
    N2Array* arr = (N2Array*)malloc(sizeof(N2Array));
    if (!arr) { free(rows_ptr); free(data); free(shape); return NULL; }
    arr->n2array = rows_ptr;
    arr->n1array = data;
    arr->shape = shape;
    return arr;
}

/* Matrix multiplication: a (m x k) dot b (k x n) => result (m x n) */
N2Array* numc_dot(const N2Array* a, const N2Array* b) {
    if (!a || !b || !a->shape || !b->shape) return NULL;
    int a_rows = a->shape[0];
    int a_cols = a->shape[1];
    int b_rows = b->shape[0];
    int b_cols = b->shape[1];
    if (a_cols != b_rows) return NULL;

    int m = a_rows, k = a_cols, n = b_cols;
    double* data = (double*)malloc(sizeof(double) * (size_t)m * n);
    if (!data) return NULL;
    double** rows_ptr = (double**)malloc(sizeof(double*) * (size_t)m);
    if (!rows_ptr) { free(data); return NULL; }
    for (int i = 0; i < m; ++i) rows_ptr[i] = data + (size_t)i * n;
    /* Try GPU path first when compiled with OpenCL support. Falls back to CPU path on failure. */
#ifdef USE_OPENCL
    if (opencl_init()) {
        /* We prefer contiguous buffers; our N2Array implementations provide n1array. */
        const double* A_data = a->n1array ? a->n1array : NULL;
        const double* B_data = b->n1array ? b->n1array : NULL;
        if (A_data && B_data) {
            int ok = opencl_matmul(A_data, B_data, data, m, k, n);
            if (ok) {
                /* GPU filled `data` */
                goto gpu_done;
            }
            /* else fall through to CPU implementation */
        }
    }
#endif

    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sumv = 0.0;
            for (int t = 0; t < k; ++t) {
                double va = a->n2array ? a->n2array[i][t] : a->n1array[(size_t)i * k + t];
                double vb = b->n2array ? b->n2array[t][j] : b->n1array[(size_t)t * n + j];
                sumv += va * vb;
            }
            rows_ptr[i][j] = sumv;
        }
    }

#ifdef USE_OPENCL
gpu_done:;
#endif

    int* shape = (int*)malloc(sizeof(int) * 2);
    if (!shape) { free(rows_ptr); free(data); return NULL; }
    shape[0] = m; shape[1] = n;
    N2Array* out = (N2Array*)malloc(sizeof(N2Array));
    if (!out) { free(shape); free(rows_ptr); free(data); return NULL; }
    out->n2array = rows_ptr; out->n1array = data; out->shape = shape;
    return out;
}

/* Helper: allocate result with given rows x cols */
static N2Array* alloc_result(int rows, int cols) {
    if (rows <= 0 || cols <= 0) return NULL;
    double* data = (double*)malloc(sizeof(double) * (size_t)rows * cols);
    if (!data) return NULL;
    double** rows_ptr = (double**)malloc(sizeof(double*) * (size_t)rows);
    if (!rows_ptr) { free(data); return NULL; }
    for (int i = 0; i < rows; ++i) rows_ptr[i] = data + (size_t)i * cols;
    int* shape = (int*)malloc(sizeof(int) * 2);
    if (!shape) { free(rows_ptr); free(data); return NULL; }
    shape[0] = rows; shape[1] = cols;
    N2Array* out = (N2Array*)malloc(sizeof(N2Array));
    if (!out) { free(shape); free(rows_ptr); free(data); return NULL; }
    out->n2array = rows_ptr; out->n1array = data; out->shape = shape;
    return out;
}

N2Array* numc_min(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0];
    int cols = a->shape[1];
    if (axis == -1) {
        double minv = a->n2array ? a->n2array[0][0] : a->n1array[0];
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
            double v = a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
            if (v < minv) minv = v;
        }
        N2Array* out = alloc_result(1,1);
        if (!out) return NULL;
        out->n1array[0] = minv; out->n2array[0][0] = minv;
        return out;
    } else if (axis == 0) {
        N2Array* out = alloc_result(1, cols);
        if (!out) return NULL;
        for (int j = 0; j < cols; ++j) {
            double mv = a->n2array ? a->n2array[0][j] : a->n1array[j];
            for (int i = 1; i < rows; ++i) {
                double v = a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
                if (v < mv) mv = v;
            }
            out->n1array[j] = mv; out->n2array[0][j] = mv;
        }
        return out;
    } else if (axis == 1) {
        N2Array* out = alloc_result(rows, 1);
        if (!out) return NULL;
        for (int i = 0; i < rows; ++i) {
            double mv = a->n2array ? a->n2array[i][0] : a->n1array[(size_t)i * cols + 0];
            for (int j = 1; j < cols; ++j) {
                double v = a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
                if (v < mv) mv = v;
            }
            out->n1array[(size_t)i * 1 + 0] = mv; out->n2array[i][0] = mv;
        }
        return out;
    }
    return NULL;
}

N2Array* numc_max(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0];
    int cols = a->shape[1];
    if (axis == -1) {
        double maxv = a->n2array ? a->n2array[0][0] : a->n1array[0];
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
            double v = a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
            if (v > maxv) maxv = v;
        }
        N2Array* out = alloc_result(1,1);
        if (!out) return NULL;
        out->n1array[0] = maxv; out->n2array[0][0] = maxv;
        return out;
    } else if (axis == 0) {
        N2Array* out = alloc_result(1, cols);
        if (!out) return NULL;
        for (int j = 0; j < cols; ++j) {
            double mv = a->n2array ? a->n2array[0][j] : a->n1array[j];
            for (int i = 1; i < rows; ++i) {
                double v = a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
                if (v > mv) mv = v;
            }
            out->n1array[j] = mv; out->n2array[0][j] = mv;
        }
        return out;
    } else if (axis == 1) {
        N2Array* out = alloc_result(rows, 1);
        if (!out) return NULL;
        for (int i = 0; i < rows; ++i) {
            double mv = a->n2array ? a->n2array[i][0] : a->n1array[(size_t)i * cols + 0];
            for (int j = 1; j < cols; ++j) {
                double v = a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
                if (v > mv) mv = v;
            }
            out->n1array[(size_t)i * 1 + 0] = mv; out->n2array[i][0] = mv;
        }
        return out;
    }
    return NULL;
}

N2Array* numc_sum(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0];
    int cols = a->shape[1];
    if (axis == -1) {
        double s = 0.0;
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) s += a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
        N2Array* out = alloc_result(1,1);
        if (!out) return NULL;
        out->n1array[0] = s; out->n2array[0][0] = s; return out;
    } else if (axis == 0) {
        N2Array* out = alloc_result(1, cols);
        if (!out) return NULL;
        for (int j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int i = 0; i < rows; ++i) s += a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
            out->n1array[j] = s; out->n2array[0][j] = s;
        }
        return out;
    } else if (axis == 1) {
        N2Array* out = alloc_result(rows, 1);
        if (!out) return NULL;
        for (int i = 0; i < rows; ++i) {
            double s = 0.0;
            for (int j = 0; j < cols; ++j) s += a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j];
            out->n1array[(size_t)i * 1 + 0] = s; out->n2array[i][0] = s;
        }
        return out;
    }
    return NULL;
}

N2Array* numc_mean(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0];
    int cols = a->shape[1];
    if (axis == -1) {
        N2Array* s = numc_sum(a, -1);
        if (!s) return NULL;
        double val = s->n2array ? s->n2array[0][0] : s->n1array[0];
        double meanv = val / (double)(rows * cols);
        N2Array* out = alloc_result(1,1);
        if (!out) { N2Array_free(s); return NULL; }
        out->n1array[0] = meanv; out->n2array[0][0] = meanv; N2Array_free(s); return out;
    } else if (axis == 0) {
        N2Array* s = numc_sum(a, 0);
        if (!s) return NULL;
        N2Array* out = alloc_result(1, cols);
        if (!out) { N2Array_free(s); return NULL; }
        for (int j = 0; j < cols; ++j) {
            double val = s->n2array ? s->n2array[0][j] : s->n1array[j];
            out->n1array[j] = val / (double)rows; out->n2array[0][j] = out->n1array[j];
        }
        N2Array_free(s); return out;
    } else if (axis == 1) {
        N2Array* s = numc_sum(a, 1);
        if (!s) return NULL;
        N2Array* out = alloc_result(rows, 1);
        if (!out) { N2Array_free(s); return NULL; }
        for (int i = 0; i < rows; ++i) {
            double val = s->n2array ? s->n2array[i][0] : s->n1array[i];
            out->n1array[(size_t)i * 1 + 0] = val / (double)cols; out->n2array[i][0] = out->n1array[(size_t)i * 1 + 0];
        }
        N2Array_free(s); return out;
    }
    return NULL;
}

N2Array* numc_stdev(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0];
    int cols = a->shape[1];
    if (axis == -1) {
        int N = rows * cols;
        N2Array* m = numc_mean(a, -1);
        if (!m) return NULL;
        double meanv = m->n2array ? m->n2array[0][0] : m->n1array[0];
        double accum = 0.0;
        for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
            double d = (a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j]) - meanv;
            accum += d * d;
        }
        double variance = accum / (double)N;
        double sigma = sqrt(variance);
        N2Array* out = alloc_result(1,1);
        if (!out) { N2Array_free(m); return NULL; }
        out->n1array[0] = sigma; out->n2array[0][0] = sigma; N2Array_free(m); return out;
    } else if (axis == 0) {
        N2Array* m = numc_mean(a, 0);
        if (!m) return NULL;
        N2Array* out = alloc_result(1, cols);
        if (!out) { N2Array_free(m); return NULL; }
        for (int j = 0; j < cols; ++j) {
            double meanv = m->n2array ? m->n2array[0][j] : m->n1array[j];
            double accum = 0.0;
            for (int i = 0; i < rows; ++i) {
                double d = (a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j]) - meanv;
                accum += d * d;
            }
            out->n1array[j] = sqrt(accum / (double)rows); out->n2array[0][j] = out->n1array[j];
        }
        N2Array_free(m); return out;
    } else if (axis == 1) {
        N2Array* m = numc_mean(a, 1);
        if (!m) return NULL;
        N2Array* out = alloc_result(rows, 1);
        if (!out) { N2Array_free(m); return NULL; }
        for (int i = 0; i < rows; ++i) {
            double meanv = m->n2array ? m->n2array[i][0] : m->n1array[i];
            double accum = 0.0;
            for (int j = 0; j < cols; ++j) {
                double d = (a->n2array ? a->n2array[i][j] : a->n1array[(size_t)i * cols + j]) - meanv;
                accum += d * d;
            }
            out->n1array[(size_t)i * 1 + 0] = sqrt(accum / (double)cols); out->n2array[i][0] = out->n1array[(size_t)i * 1 + 0];
        }
        N2Array_free(m); return out;
    }
    return NULL;
}


