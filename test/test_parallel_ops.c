/**
 * @file test_parallel_ops.c
 * @brief Test Parallel Operations (transpose, argsort, square)
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "../include/other_operating_parallel.h"

int main(void) {
    printf("=== Testing Parallel Operations ===\n\n");
    
    int n = 5;
    int num_threads = 4;
    
    // Test 1: Square operation
    printf("Test 1: Square Parallel\n");
    printf("------------------------\n");
    double A[] = {1.0, 2.0, 3.0, 4.0, 5.0};
    
    printf("Input array: ");
    for(int i = 0; i < n; i++) printf("%.1f ", A[i]);
    printf("\n");
    
    double* squared = square_parallel(A, n, num_threads);
    
    printf("Squared:     ");
    for(int i = 0; i < n; i++) printf("%.1f ", squared[i]);
    printf("\n\n");
    
    // Test 2: Argsort operation
    printf("Test 2: Argsort Parallel\n");
    printf("------------------------\n");
    double B[] = {5.0, 2.0, 8.0, 1.0, 9.0};
    
    printf("Input array: ");
    for(int i = 0; i < n; i++) printf("%.1f ", B[i]);
    printf("\n");
    
    double* indices = argsort_parallel(B, n, num_threads);
    
    printf("Sorted indices: ");
    for(int i = 0; i < n; i++) printf("%.0f ", indices[i]);
    printf("\n");
    
    printf("Values at sorted indices: ");
    for(int i = 0; i < n; i++) printf("%.1f ", B[(int)indices[i]]);
    printf("\n\n");
    
    // Test 3: Transpose operation (simple 2D matrix)
    printf("Test 3: Transpose Parallel\n");
    printf("---------------------------\n");
    int rows = 3, cols = 4;
    
    double** C_rows = (double**)malloc(rows * sizeof(double*));
    double C[] = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9, 10, 11, 12
    };
    for(int i = 0; i < rows; i++) C_rows[i] = &C[i*cols];
    
    printf("Original matrix (%dx%d):\n", rows, cols);
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++) {
            printf("%4.0f ", C[i*cols + j]);
        }
        printf("\n");
    }
    
    double** C_T = transpose_parallel(C_rows, rows, cols, num_threads);
    
    printf("\nTransposed matrix (%dx%d):\n", cols, rows);
    for(int i = 0; i < cols; i++) {
        for(int j = 0; j < rows; j++) {
            printf("%4.0f ", C_T[i][j]);
        }
        printf("\n");
    }
    
    // Cleanup
    free(squared);
    free(indices);
    for(int i = 0; i < cols; i++) free(C_T[i]);
    free(C_T);
    free(C_rows);
    
    printf("\n✓ Parallel operations test completed!\n");
    return 0;
}
