#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <math.h>
#include "../include/other_operating_parallel.h"

/* Test/Benchmark code for parallel operations - uses header-only functions */

#ifdef STANDALONE
int main(int argc, char** argv) {
    // Example usage of the functions from other_operating_parallel.h
    int n = 5;
    int num_threads = 4;
    double* A = (double*)malloc(n * sizeof(double));
    printf("Original array A:\n");
    for (int i = 0; i < n; ++i) {
        A[i] = (double)(n - i); // 5, 4, 3, 2, 1
        printf("%.2f ", A[i]);
    }
    printf("\n");

    // Test square
    double* squared_A = square_parallel(A, n, num_threads);
    printf("Squared A:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.2f ", squared_A[i]);
    }
    printf("\n");

    // Test argsort
    double* sorted_indices = argsort_parallel(A, n, num_threads);
    printf("Argsorted indices of A:\n");
    for (int i = 0; i < n; ++i) {
        printf("%.0f ", sorted_indices[i]);
    }
    printf("\n");
    
    // Verify sort
    printf("Values at sorted indices:\n");
    for (int i = 0; i < n; ++i) {
        int idx = (int)sorted_indices[i];
        printf("%.2f ", A[idx]);
    }
    printf("\n");

    free(A);
    free(squared_A);
    free(sorted_indices);
    return 0;
}
#endif
