#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../include/decompose_operation.h"
#include "../include/SVD_serial.h"
#include "../include/SVD_parallel.h"
#include "../include/mulmat.h"

void decompose_project_serial(int m, int n, int k, const double* A, double* Result) {
    // Allocate SVD outputs
    double* U = (double*)malloc(m * n * sizeof(double));
    double* S = (double*)malloc(n * sizeof(double));
    double* V = (double*)malloc(n * n * sizeof(double));

    if (!U || !S || !V) {
        if(U) free(U);
        if(S) free(S);
        if(V) free(V);
        return;
    }

    // 1. Perform SVD: A = U * S * V^T
    // Note: svd_decomposition returns V where columns are right singular vectors
    svd_decomposition(m, n, A, U, S, V);

    // 2. Project: Result = A * V[:, :k]
    // A is (m x n), V is (n x n), Result is (m x k)
    for(int r = 0; r < m; ++r) {
        for(int c = 0; c < k; ++c) {
            double sum = 0.0;
            for(int p = 0; p < n; ++p) {
                sum += A[r * n + p] * V[p * n + c];
            }
            Result[r * k + c] = sum;
        }
    }

    free(U);
    free(S);
    free(V);
}

void decompose_project_parallel(int m, int n, int k, const double* A, double* Result, int num_threads) {
    // Allocate SVD outputs
    double* U = (double*)malloc(m * n * sizeof(double));
    double* S = (double*)malloc(n * sizeof(double));
    double* V = (double*)malloc(n * n * sizeof(double));

    if (!U || !S || !V) {
        if(U) free(U);
        if(S) free(S);
        if(V) free(V);
        return;
    }

    // 1. Perform Parallel SVD
    svd_decomposition_parallel(m, n, A, U, S, V, num_threads);

    // 2. Project: Result = A * V[:, :k] using DNS Matmul
    
    // Prepare A rows (m x n)
    double** A_rows = (double**)malloc(m * sizeof(double*));
    // We need a non-const copy of pointers or cast const away carefully, 
    // but matmul_dns likely takes double**. 
    // Since A is const double*, we should probably cast it or copy it if matmul modifies it (it shouldn't).
    // Assuming matmul_dns reads A.
    for(int i = 0; i < m; i++) A_rows[i] = (double*)&A[i * n];

    // Prepare V_k (n x k) - Extract first k columns of V
    double* V_k = (double*)malloc(n * k * sizeof(double));
    
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for(int r = 0; r < n; ++r) {
        for(int c = 0; c < k; ++c) {
            V_k[r * k + c] = V[r * n + c];
        }
    }

    double** V_k_rows = (double**)malloc(n * sizeof(double*));
    for(int i = 0; i < n; i++) V_k_rows[i] = &V_k[i * k];

    // Prepare Result rows (m x k)
    double** Result_rows = (double**)malloc(m * sizeof(double*));
    for(int i = 0; i < m; i++) Result_rows[i] = &Result[i * k];

    // Perform Multiplication: Result = A * V_k
    // A is (m x n), V_k is (n x k) -> Result is (m x k)
    matmul_dns(A_rows, V_k_rows, Result_rows, m, n, k, num_threads);

    // Cleanup
    free(A_rows);
    free(V_k);
    free(V_k_rows);
    free(Result_rows);
    
    free(U);
    free(S);
    free(V);
}
