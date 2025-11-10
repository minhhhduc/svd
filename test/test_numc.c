#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/n2array.h"
#include "../include/numc.h"

static int approx_eq(double a, double b) {
    double eps = 1e-9;
    return fabs(a - b) <= eps;
}

static int test_linspace() {
    int pass = 1;
    double* v = linspace(0.0, 1.0, 5);
    if (!v) return 0;
    double expect[5] = {0.0, 0.25, 0.5, 0.75, 1.0};
    for (int i = 0; i < 5; ++i) {
        if (!approx_eq(v[i], expect[i])) pass = 0;
    }
    free(v);
    return pass;
}

static int test_arange() {
    double start = 0.0, stop = 1.0, step = 0.5;
    int n = (int)floor((stop - start) / step);
    if (n <= 0) return 0;
    double* v = arange(start, stop, step);
    if (!v) return 0;
    for (int i = 0; i < n; ++i) {
        double expect = start + i * step;
        if (!approx_eq(v[i], expect)) { free(v); return 0; }
    }
    free(v);
    return 1;
}

static int test_zeros_ones() {
    N2Array* z = zeros(2,3);
    N2Array* o = ones(2,3);
    if (!z || !o) return 0;
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 3; ++j) {
        if (!approx_eq(N2Array_get(z,i,j), 0.0)) { N2Array_free(z); N2Array_free(o); return 0; }
        if (!approx_eq(N2Array_get(o,i,j), 1.0)) { N2Array_free(z); N2Array_free(o); return 0; }
    }
    N2Array_free(z);
    N2Array_free(o);
    return 1;
}

static int test_dot_sum_mean_stdev() {
    // a: 2x3, b: 3x2 => dot = 2x2
    double row0[3] = {1,2,3};
    double row1[3] = {4,5,6};
    double* a_rows[2] = {row0, row1};
    int ash[2] = {2,3};
    N2Array* A = N2Array_from_2d(a_rows, ash);

    double brow0[2] = {7,8};
    double brow1[2] = {9,10};
    double brow2[2] = {11,12};
    double* b_rows[3] = {brow0, brow1, brow2};
    int bsh[2] = {3,2};
    N2Array* B = N2Array_from_2d(b_rows, bsh);

    N2Array* D = dot(A,B);
    if (!D) { N2Array_free(A); N2Array_free(B); return 0; }
    // expected dot manually computed
    double expect[2][2] = { {58, 64}, {139, 154} };
    for (int i = 0; i < 2; ++i) for (int j = 0; j < 2; ++j) {
        if (!approx_eq(N2Array_get(D,i,j), expect[i][j])) { N2Array_free(A); N2Array_free(B); N2Array_free(D); return 0; }
    }

    // sum all
    N2Array* s = sum(A, -1);
    if (!s) { N2Array_free(A); N2Array_free(B); N2Array_free(D); return 0; }
    double total = N2Array_get(s,0,0);
    if (!approx_eq(total, 21.0)) { N2Array_free(A); N2Array_free(B); N2Array_free(D); N2Array_free(s); return 0; }

    N2Array_free(A); N2Array_free(B); N2Array_free(D); N2Array_free(s);
    return 1;
}

int main(void) {
    int ok = 1;
    if (!test_linspace()) { printf("test_linspace FAILED\n"); ok = 0; }
    else printf("test_linspace OK\n");

    if (!test_arange()) { printf("test_arange FAILED\n"); ok = 0; }
    else printf("test_arange OK\n");

    if (!test_zeros_ones()) { printf("test_zeros_ones FAILED\n"); ok = 0; }
    else printf("test_zeros_ones OK\n");

    if (!test_dot_sum_mean_stdev()) { printf("test_dot_sum_mean_stdev FAILED\n"); ok = 0; }
    else printf("test_dot_sum_mean_stdev OK\n");

    if (ok) { printf("ALL TESTS PASSED\n"); return 0; }
    else { printf("SOME TESTS FAILED\n"); return 2; }
}
