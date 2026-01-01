/**
 * @file test_svd_simple.c
 * @brief Test SVD Decomposition
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/SVD_parallel.h"
#include "../include/other_operating_parallel.h"
#include "../include/norm_reducing_jacobi_parallel.h"

int main(void) {
    printf("=== Testing SVD Decomposition ===\n\n");
    
    int m = 3;  // rows
    int n = 2;  // cols
    int num_threads = 4;
    
    // Create a simple test matrix
    double A[] = {
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0
    };
    
    printf("Input matrix A (%dx%d):\n", m, n);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%8.4f ", A[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    // Allocate outputs
    double* U = (double*)malloc(m * n * sizeof(double));
    double* S = (double*)malloc(n * sizeof(double));
    double* V = (double*)malloc(n * n * sizeof(double));
    
    printf("Computing SVD: A = U * S * V^T\n");
    printf("Using %d threads...\n\n", num_threads);
    
    svd_decomposition_parallel(m, n, A, U, S, V, num_threads);
    
    printf("Singular Values (S):\n");
    for(int i = 0; i < n; i++) {
        printf("S[%d] = %.6f\n", i, S[i]);
    }
    printf("\n");
    
    printf("Left Singular Vectors (U, %dx%d):\n", m, n);
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%8.4f ", U[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
    
    printf("Right Singular Vectors (V, %dx%d):\n", n, n);
    for(int i = 0; i < n; i++) {
        for(int j = 0; j < n; j++) {
            printf("%8.4f ", V[i*n + j]);
        }
        printf("\n");
    }
    
    // Simple verification: Check ||A - U*S*V^T||
    printf("\n--- Verification ---\n");
    printf("Reconstructing matrix from SVD...\n");
    
    // Reconstruct: temp = U * diag(S)
    double* temp = (double*)malloc(m * n * sizeof(double));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            temp[i*n + j] = U[i*n + j] * S[j];
        }
    }
    
    // A_recon = temp * V^T
    double* A_recon = (double*)malloc(m * n * sizeof(double));
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            A_recon[i*n + j] = 0.0;
            for(int k = 0; k < n; k++) {
                A_recon[i*n + j] += temp[i*n + k] * V[j*n + k];
            }
        }
    }
    
    printf("Reconstructed A:\n");
    for(int i = 0; i < m; i++) {
        for(int j = 0; j < n; j++) {
            printf("%8.4f ", A_recon[i*n + j]);
        }
        printf("\n");
    }
    
    // Calculate reconstruction error
    double error = 0.0;
    for(int i = 0; i < m*n; i++) {
        double diff = A[i] - A_recon[i];
        error += diff * diff;
    }
    error = sqrt(error);
    
    printf("\nReconstruction error ||A - A_recon||: %.2e\n", error);
    
    if(error < 1e-10) {
        printf("✓ SVD reconstruction successful!\n");
    } else {
        printf("⚠ Warning: Reconstruction error is larger than expected\n");
    }
    
    // Cleanup
    free(U);
    free(S);
    free(V);
    free(temp);
    free(A_recon);
    
    printf("\n✓ SVD test completed!\n");
    return 0;
}
