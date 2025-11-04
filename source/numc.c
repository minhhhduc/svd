/* Pure C implementations of the * helpers using the C N2Array API */
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE_OPENCL
#include "../include/opencl_helper.h"
#endif
#include "../include/n2array.h"

double* arange(double start, double stop, double step) {
    if (step <= 0.0) return NULL;
    int n = (int)floor((stop - start) / step);
    if (n <= 0) return NULL;
    double* out = (double*)malloc(sizeof(double) * n);
    if (!out) return NULL;
    for (int i = 0; i < n; ++i) out[i] = start + i * step;
    return out;
}

double* linspace(double start, double stop, int num) {
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

N2Array* zeros(int rows, int cols) {
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

N2Array* ones(int rows, int cols) {
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

/* Compute sum of N2Array along axis (-1=all, 0=rows, 1=cols) */
N2Array* sum(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0], cols = a->shape[1];
    
    if (axis == -1) {
        /* sum all elements -> return 1x1 array */
        double s = 0.0;
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                s += N2Array_get(a, i, j);
            }
        }
        N2Array* out = zeros(1, 1);
        if (out) out->n1array[0] = s;
        return out;
    } else if (axis == 0) {
        /* sum along rows -> return 1xN array */
        N2Array* out = zeros(1, cols);
        if (!out) return NULL;
        for (int j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int i = 0; i < rows; ++i) s += N2Array_get(a, i, j);
            out->n1array[j] = s;
        }
        return out;
    } else if (axis == 1) {
        /* sum along cols -> return Nx1 array */
        N2Array* out = zeros(rows, 1);
        if (!out) return NULL;
        for (int i = 0; i < rows; ++i) {
            double s = 0.0;
            for (int j = 0; j < cols; ++j) s += N2Array_get(a, i, j);
            out->n1array[i] = s;
        }
        return out;
    }
    return NULL;
}

/* Compute mean of N2Array along axis */
N2Array* mean(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0], cols = a->shape[1];
    
    N2Array* s = sum(a, axis);
    if (!s) return NULL;
    
    if (axis == -1) {
        s->n1array[0] /= (double)(rows * cols);
    } else if (axis == 0) {
        for (int j = 0; j < cols; ++j) s->n1array[j] /= (double)rows;
    } else if (axis == 1) {
        for (int i = 0; i < rows; ++i) s->n1array[i] /= (double)cols;
    }
    return s;
}

/* Compute min of N2Array along axis */
N2Array* min(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0], cols = a->shape[1];
    
    if (axis == -1) {
        double minv = N2Array_get(a, 0, 0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double v = N2Array_get(a, i, j);
                if (v < minv) minv = v;
            }
        }
        N2Array* out = zeros(1, 1);
        if (out) out->n1array[0] = minv;
        return out;
    } else if (axis == 0) {
        N2Array* out = zeros(1, cols);
        if (!out) return NULL;
        for (int j = 0; j < cols; ++j) {
            double minv = N2Array_get(a, 0, j);
            for (int i = 1; i < rows; ++i) {
                double v = N2Array_get(a, i, j);
                if (v < minv) minv = v;
            }
            out->n1array[j] = minv;
        }
        return out;
    } else if (axis == 1) {
        N2Array* out = zeros(rows, 1);
        if (!out) return NULL;
        for (int i = 0; i < rows; ++i) {
            double minv = N2Array_get(a, i, 0);
            for (int j = 1; j < cols; ++j) {
                double v = N2Array_get(a, i, j);
                if (v < minv) minv = v;
            }
            out->n1array[i] = minv;
        }
        return out;
    }
    return NULL;
}

/* Compute max of N2Array along axis */
N2Array* max(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0], cols = a->shape[1];
    
    if (axis == -1) {
        double maxv = N2Array_get(a, 0, 0);
        for (int i = 0; i < rows; ++i) {
            for (int j = 0; j < cols; ++j) {
                double v = N2Array_get(a, i, j);
                if (v > maxv) maxv = v;
            }
        }
        N2Array* out = zeros(1, 1);
        if (out) out->n1array[0] = maxv;
        return out;
    } else if (axis == 0) {
        N2Array* out = zeros(1, cols);
        if (!out) return NULL;
        for (int j = 0; j < cols; ++j) {
            double maxv = N2Array_get(a, 0, j);
            for (int i = 1; i < rows; ++i) {
                double v = N2Array_get(a, i, j);
                if (v > maxv) maxv = v;
            }
            out->n1array[j] = maxv;
        }
        return out;
    } else if (axis == 1) {
        N2Array* out = zeros(rows, 1);
        if (!out) return NULL;
        for (int i = 0; i < rows; ++i) {
            double maxv = N2Array_get(a, i, 0);
            for (int j = 1; j < cols; ++j) {
                double v = N2Array_get(a, i, j);
                if (v > maxv) maxv = v;
            }
            out->n1array[i] = maxv;
        }
        return out;
    }
    return NULL;
}

/* Compute standard deviation of N2Array along axis */
N2Array* stdev(const N2Array* a, int axis) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0], cols = a->shape[1];
    
    N2Array* m = mean(a, axis);
    if (!m) return NULL;
    
    N2Array* out = NULL;
    if (axis == -1) {
        out = zeros(1, 1);
        if (out) {
            double mean_val = m->n1array[0];
            double accum = 0.0;
            for (int i = 0; i < rows; ++i) {
                for (int j = 0; j < cols; ++j) {
                    double d = N2Array_get(a, i, j) - mean_val;
                    accum += d * d;
                }
            }
            out->n1array[0] = sqrt(accum / (double)(rows * cols));
        }
    } else if (axis == 0) {
        out = zeros(1, cols);
        if (out) {
            for (int j = 0; j < cols; ++j) {
                double mean_val = m->n1array[j];
                double accum = 0.0;
                for (int i = 0; i < rows; ++i) {
                    double d = N2Array_get(a, i, j) - mean_val;
                    accum += d * d;
                }
                out->n1array[j] = sqrt(accum / (double)rows);
            }
        }
    } else if (axis == 1) {
        out = zeros(rows, 1);
        if (out) {
            for (int i = 0; i < rows; ++i) {
                double mean_val = m->n1array[i];
                double accum = 0.0;
                for (int j = 0; j < cols; ++j) {
                    double d = N2Array_get(a, i, j) - mean_val;
                    accum += d * d;
                }
                out->n1array[i] = sqrt(accum / (double)cols);
            }
        }
    }
    N2Array_free(m);
    return out;
}

/* Matrix dot product */
N2Array* dot(const N2Array* a, const N2Array* b) {
    if (!a || !b || !a->shape || !b->shape) return NULL;
    int a_rows = a->shape[0], a_cols = a->shape[1];
    int b_rows = b->shape[0], b_cols = b->shape[1];
    if (a_cols != b_rows) return NULL;
    
    N2Array* out = zeros(a_rows, b_cols);
    if (!out) return NULL;
    
    for (int i = 0; i < a_rows; ++i) {
        for (int j = 0; j < b_cols; ++j) {
            double sum = 0.0;
            for (int k = 0; k < a_cols; ++k) {
                sum += N2Array_get(a, i, k) * N2Array_get(b, k, j);
            }
            out->n1array[(size_t)i * b_cols + j] = sum;
        }
    }
    return out;
}

N2Array* diag(N2Array* a) {
    if (!a || !a->shape) return NULL;
    int rows = a->shape[0], cols = a->shape[1];
    
    if (rows == 1 || cols == 1) {
        /* Create diagonal matrix from vector */
        int n = (rows > cols) ? rows : cols;
        N2Array* out = zeros(n, n);
        if (!out) return NULL;
        for (int i = 0; i < n; ++i) {
            double val = (rows == 1) ? N2Array_get(a, 0, i) : N2Array_get(a, i, 0);
            out->n1array[(size_t)i * n + i] = val;
        }
        return out;
    } else if (rows == cols) {
        /* Extract diagonal from square matrix */
        N2Array* out = zeros(1, rows);
        if (!out) return NULL;
        for (int i = 0; i < rows; ++i) {
            out->n1array[i] = N2Array_get(a, i, i);
        }
        return out;
    }
    return NULL;
}

void jacobi_eigen(double** A, double** V, int N, double* eigvals, int MAX_ITER, double EPS) {
    // Khởi tạo V là ma trận đơn vị
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            V[i][j] = (i == j) ? 1.0 : 0.0;
    }

    for (int iter = 0; iter < MAX_ITER; iter++) {
        // Tìm phần tử ngoài đường chéo lớn nhất
        int p = 0, q = 1;
        double max = fabs(A[p][q]);
        for (int i = 0; i < N; i++) {
            for (int j = i + 1; j < N; j++) {
                if (fabs(A[i][j]) > max) {
                    max = fabs(A[i][j]);
                    p = i;
                    q = j;
                }
            }
        }

        if (max < EPS) break; // hội tụ

        double tau = (A[q][q] - A[p][p]) / (2.0 * A[p][q]);
        double t = ((tau >= 0) ? 1.0 : -1.0) / (fabs(tau) + sqrt(1.0 + tau * tau));
        double c = 1.0 / sqrt(1 + t * t);
        double s = c * t;

        // Lưu lại giá trị cần dùng
        double app = A[p][p];
        double aqq = A[q][q];

        // Cập nhật A
        A[p][p] = c * c * app - 2.0 * s * c * A[p][q] + s * s * aqq;
        A[q][q] = s * s * app + 2.0 * s * c * A[p][q] + c * c * aqq;
        A[p][q] = A[q][p] = 0.0;

        for (int i = 0; i < N; i++) {
            if (i != p && i != q) {
                double aip = A[i][p];
                double aiq = A[i][q];
                A[i][p] = A[p][i] = c * aip - s * aiq;
                A[i][q] = A[q][i] = s * aip + c * aiq;
            }
        }

        // Cập nhật ma trận vector riêng V
        for (int i = 0; i < N; i++) {
            double vip = V[i][p];
            double viq = V[i][q];
            V[i][p] = c * vip - s * viq;
            V[i][q] = s * vip + c * viq;
        }
    }

    // Lấy giá trị riêng
    for (int i = 0; i < N; i++)
        eigvals[i] = A[i][i];
}

pair* eigh(const N2Array* a) {
    if (a->shape[0] != a->shape[1]) return NULL; // Chỉ áp dụng cho ma trận vuông
    int n = a->shape[0];
}
