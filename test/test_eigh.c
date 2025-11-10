#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../include/n2array.h"
#include "../include/numc.h"

int main() {
    // Create a simple 3x3 symmetric matrix:
    // [4  1  0]
    // [1  3  1]
    // [0  1  4]
    double data[] = {4.0, 1.0, 0.0,
                     1.0, 3.0, 1.0,
                     0.0, 1.0, 4.0};
    
    int shape[2] = {3, 3};
    N2Array* A = N2Array_from_1d(data, shape);
    
    if (!A) {
        printf("ERROR: Failed to create matrix\n");
        return 1;
    }
    
    printf("Input matrix:\n");
    char* str = N2Array_to_string(A);
    if (str) {
        printf("%s\n", str);
        free(str);
    }
    
    // Compute eigenvalues and eigenvectors
    pair* result = eigh(A);
    
    if (!result) {
        printf("ERROR: Failed to compute eigenvalues\n");
        N2Array_free(A);
        return 1;
    }
    
    printf("\nEigenvalues:\n");
    for (int i = 0; i < 3; ++i) {
        printf("  lambda[%d] = %.6f\n", i, N2Array_get(result->first, i, 0));
    }
    
    printf("\nEigenvectors (column-wise):\n");
    str = N2Array_to_string(result->second);
    if (str) {
        printf("%s\n", str);
        free(str);
    }
    
    printf("\nVerification (A*v_i = lambda_i*v_i):\n");
    for (int i = 0; i < 3; ++i) {
        double lambda = N2Array_get(result->first, i, 0);
        printf("Column %d: lambda = %.6f\n", i, lambda);
        printf("  A*v = [");
        for (int j = 0; j < 3; ++j) {
            double av_sum = 0.0;
            for (int k = 0; k < 3; ++k) {
                av_sum += N2Array_get(A, j, k) * N2Array_get(result->second, k, i);
            }
            printf("%.4f", av_sum);
            if (j < 2) printf(", ");
        }
        printf("]\n");
        
        printf("  lambda*v = [");
        for (int j = 0; j < 3; ++j) {
            printf("%.4f", lambda * N2Array_get(result->second, j, i));
            if (j < 2) printf(", ");
        }
        printf("]\n");
    }
    
    // Cleanup
    N2Array_free(A);
    if (result->first) N2Array_free(result->first);
    if (result->second) N2Array_free(result->second);
    free(result);
    
    printf("Eigenvalue decomposition (eigh) test completed successfully!\n");
    fflush(stdout);
    exit(0);
}

