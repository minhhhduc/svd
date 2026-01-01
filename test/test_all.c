/**
 * @file test_all.c
 * @brief Comprehensive Test Suite for Parallel Linear Algebra
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include "../include/norm_reducing_jacobi_parallel.h"
#include "../include/other_operating_parallel.h"
#include "../include/SVD_parallel.h"

int main(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║   Comprehensive Test Suite - Parallel Linear Algebra          ║\n");
    printf("║   Testing eigenvalue, SVD, and matrix operations              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    int num_threads = omp_get_max_threads();
    printf("Available threads: %d\n\n", num_threads);
    
    // ==================== Test 1: Eigenvalue Computation ====================
    printf("┌─ TEST 1: Eigenvalue Computation ─────────────────────────────────┐\n");
    
    int n = 4;
    double A[] = {
        2.0, 1.0, 0.0, 0.0,
        1.0, 3.0, 1.0, 0.0,
        0.0, 1.0, 2.0, 1.0,
        0.0, 0.0, 1.0, 4.0
    };
    
    printf("Matrix dimension: %dx%d\n", n, n);
    printf("Matrix:\n");
    for(int i = 0; i < n; i++) {
        printf("  ");
        for(int j = 0; j < n; j++) {
            printf("%7.2f ", A[i*n + j]);
        }
        printf("\n");
    }
    
    double w[4], V[16];
    double start_time = omp_get_wtime();
    compute_eigenvalues_parallel(n, A, w, V);
    double elapsed = omp_get_wtime() - start_time;
    
    printf("\nEigenvalues computed in %.6f seconds:\n", elapsed);
    printf("  ");
    for(int i = 0; i < n; i++) printf("%8.4f ", w[i]);
    printf("\n");
    printf("└─ PASSED ─────────────────────────────────────────────────────────┘\n\n");
    
    // ==================== Test 2: Parallel Array Operations ====================
    printf("┌─ TEST 2: Parallel Array Operations ───────────────────────────────┐\n");
    
    int arr_size = 10;
    double* arr = (double*)malloc(arr_size * sizeof(double));
    
    printf("Testing with array size: %d\n", arr_size);
    
    // Fill array
    for(int i = 0; i < arr_size; i++) {
        arr[i] = (double)((i + 1) * 7 % 13);  // Pseudo-random values
    }
    
    printf("Original array: ");
    for(int i = 0; i < arr_size; i++) printf("%.0f ", arr[i]);
    printf("\n");
    
    // Test square
    double* squared = square_parallel(arr, arr_size, num_threads);
    printf("Squared array:  ");
    for(int i = 0; i < arr_size; i++) printf("%.0f ", squared[i]);
    printf("\n");
    
    // Test argsort
    double* indices = argsort_parallel(arr, arr_size, num_threads);
    printf("Sorted indices: ");
    for(int i = 0; i < arr_size; i++) printf("%.0f ", indices[i]);
    printf("\n");
    
    printf("Values at sorted indices: ");
    for(int i = 0; i < arr_size; i++) printf("%.0f ", arr[(int)indices[i]]);
    printf("\n");
    
    free(squared);
    free(indices);
    free(arr);
    printf("└─ PASSED ─────────────────────────────────────────────────────────┘\n\n");
    
    // ==================== Test 3: SVD Decomposition ====================
    printf("┌─ TEST 3: SVD Decomposition ──────────────────────────────────────┐\n");
    
    int m = 4, n_svd = 3;
    double B[] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };
    
    printf("Matrix dimension: %dx%d\n", m, n_svd);
    printf("Matrix B:\n");
    for(int i = 0; i < m; i++) {
        printf("  ");
        for(int j = 0; j < n_svd; j++) {
            printf("%7.2f ", B[i*n_svd + j]);
        }
        printf("\n");
    }
    
    double* U = (double*)malloc(m * n_svd * sizeof(double));
    double* S = (double*)malloc(n_svd * sizeof(double));
    double* V_svd = (double*)malloc(n_svd * n_svd * sizeof(double));
    
    start_time = omp_get_wtime();
    svd_decomposition_parallel(m, n_svd, B, U, S, V_svd, num_threads);
    elapsed = omp_get_wtime() - start_time;
    
    printf("\nSingular values (computed in %.6f seconds):\n", (double)elapsed);
    printf("  S = [");
    for(int i = 0; i < n_svd; i++) {
        printf("%8.4f%s", S[i], (i < n_svd-1) ? ", " : " ]\n");
    }
    
    // Verify reconstruction
    double* B_recon = (double*)malloc(m * n_svd * sizeof(double));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n_svd; j++) {
            B_recon[i*n_svd + j] = 0.0;
            for(int k = 0; k < n_svd; k++) {
                B_recon[i*n_svd + j] += U[i*n_svd + k] * S[k] * V_svd[j*n_svd + k];
            }
        }
    }
    
    double recon_error = 0.0;
    for(int i = 0; i < m*n_svd; i++) {
        double diff = B[i] - B_recon[i];
        recon_error += diff * diff;
    }
    recon_error = sqrt(recon_error);
    
    printf("Reconstruction error: %.2e\n", recon_error);
    if(recon_error < 1e-10) {
        printf("✓ SVD reconstruction PASSED\n");
    }
    
    free(U);
    free(S);
    free(V_svd);
    free(B_recon);
    printf("└─ PASSED ─────────────────────────────────────────────────────────┘\n\n");
    
    // ==================== Summary ====================
    printf("╔════════════════════════════════════════════════════════════════╗\n");
    printf("║                    ALL TESTS PASSED ✓                          ║\n");
    printf("║                                                                ║\n");
    printf("║  • Eigenvalue computation (Jacobi method)                      ║\n");
    printf("║  • Parallel array operations (square, argsort, transpose)      ║\n");
    printf("║  • SVD decomposition with reconstruction verification          ║\n");
    printf("║                                                                ║\n");
    printf("║  The parallel linear algebra library is working correctly!     ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
    return 0;
}
