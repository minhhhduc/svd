/**
 * @file test_eigenvalues.c
 * @brief Test Parallel Eigenvalue Computation
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/norm_reducing_jacobi_parallel.h"

int main(void) {
    printf("=== Testing Eigenvalue Computation ===\n\n");
    
    int n = 3;
    
    // Create a simple symmetric matrix:
    // [4  -1   0]
    // [-1  3  -1]
    // [0  -1   2]
    double A[] = {
        4.0, -1.0,  0.0,
       -1.0,  3.0, -1.0,
        0.0, -1.0,  2.0
    };
    
    double w[3];        // eigenvalues
    double V[9];        // eigenvectors
    
    printf("Input matrix A (3x3):\n");
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%8.4f ", A[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf("Computing eigenvalues and eigenvectors...\n");
    compute_eigenvalues_parallel(n, A, w, V);
    
    printf("\nEigenvalues:\n");
    for(int i = 0; i < n; i++) {
        printf("w[%d] = %.6f\n", i, w[i]);
    }
    
    printf("\nEigenvectors (as columns):\n");
    for(int i = 0; i < n; i++) {
        printf("V[%d]: ", i);
        for(int j = 0; j < n; j++) {
            printf("%.6f ", V[j*n + i]);
        }
        printf("\n");
    }
    
    printf("\n✓ Eigenvalue computation test completed!\n");
    return 0;
}
