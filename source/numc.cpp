// Concrete implementations for numerical helpers operating on N2Array (double)
#include "../include/numc.h"
#include "../include/n2array.h"
#include <cstring>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

N2Array dot(const N2Array& a, const N2Array& b) {
    if (!a.shape || !b.shape) throw std::invalid_argument("Null shape");
    int a_rows = a.shape[0];
    int a_cols = a.shape[1];
    int b_rows = b.shape[0];
    int b_cols = b.shape[1];
    if (a_cols != b_rows) throw std::invalid_argument("Inner dimensions must match for dot");

    int m = a_rows;
    int k = a_cols;
    int n = b_cols;

    double** result = new double*[m];
    for (int i = 0; i < m; ++i) result[i] = new double[n];

    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * a_cols + c]; };
    auto getb = [&](int r, int c) -> double { return b.n2array ? b.n2array[r][c] : b.n1array[r * b_cols + c]; };

    #ifdef _OPENMP
    #pragma omp parallel for
    #endif
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            double sumv = 0.0;
            for (int t = 0; t < k; ++t) {
                sumv += geta(i, t) * getb(t, j);
            }
            result[i][j] = sumv;
        }
    }

    int* shape = new int[2]{m, n};
    return N2Array(result, shape);
}

N2Array min(const N2Array& a) {
    if (!a.shape) throw std::invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];
    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * cols + c]; };

    double minv = geta(0,0);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        double v = geta(i,j);
        if (v < minv) minv = v;
    }
    double** out = new double*[1]; out[0] = new double[1]; out[0][0] = minv;
    int* shape = new int[2]{1,1};
    return N2Array(out, shape);
}

N2Array max(const N2Array& a) {
    if (!a.shape) throw std::invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];
    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * cols + c]; };

    double maxv = geta(0,0);
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        double v = geta(i,j);
        if (v > maxv) maxv = v;
    }
    double** out = new double*[1]; out[0] = new double[1]; out[0][0] = maxv;
    int* shape = new int[2]{1,1};
    return N2Array(out, shape);
}

N2Array sum(const N2Array& a) {
    if (!a.shape) throw std::invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];
    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * cols + c]; };

    double s = 0.0;
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) s += geta(i,j);

    double** out = new double*[1]; out[0] = new double[1]; out[0][0] = s;
    int* shape = new int[2]{1,1};
    return N2Array(out, shape);
}

N2Array mean(const N2Array& a) {
    int rows = a.shape[0];
    int cols = a.shape[1];
    N2Array s = sum(a);
    double val = (s.n2array ? s.n2array[0][0] : s.n1array[0]) / static_cast<double>(rows * cols);
    double** out = new double*[1]; out[0] = new double[1]; out[0][0] = val;
    int* shape = new int[2]{1,1};
    return N2Array(out, shape);
}

N2Array stdev(const N2Array& a) {
    if (!a.shape) throw std::invalid_argument("Null shape");
    int rows = a.shape[0];
    int cols = a.shape[1];
    int N = rows * cols;

    N2Array m = mean(a);
    double meanv = m.n2array ? m.n2array[0][0] : m.n1array[0];

    double accum = 0.0;
    auto geta = [&](int r, int c) -> double { return a.n2array ? a.n2array[r][c] : a.n1array[r * cols + c]; };
    for (int i = 0; i < rows; ++i) for (int j = 0; j < cols; ++j) {
        double d = geta(i,j) - meanv;
        accum += d * d;
    }
    double variance = accum / static_cast<double>(N);
    double sigma = std::sqrt(variance);

    double** out = new double*[1]; out[0] = new double[1]; out[0][0] = sigma;
    int* shape = new int[2]{1,1};
    return N2Array(out, shape);
}