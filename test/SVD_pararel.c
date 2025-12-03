#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include "../include/mulmat.h"
#include "../include/stream.h"
#include "../include/other_operating_parallel.h"
#include "../include/norm_reducing_jacobi_v2_parallel.h"

/* ============================================================================
 * CONFIGURATION CONSTANTS
 * ============================================================================ */

#define TOLERANCE           1e-12
#define REL_TOLERANCE       1e-10
#define MAX_SWEEPS          100
#define PARALLEL_THRESHOLD  32
#define MAX_PATH_LEN        256

#define MAT_AT(arr, row, col, n) ((arr)[(row) * (n) + (col)])
#define SIGN(x) ((x) >= 0.0 ? 1.0 : -1.0)

/* ============================================================================
 * SVD IMPLEMENTATION (PARALLEL)
 * ============================================================================ */

/**
 * @brief Computes SVD: A = U * S * V^T
 * @param m Rows of A
 * @param n Cols of A
 * @param A Input matrix (flat array m*n)
 * @param U Output matrix (m*n) - Left Singular Vectors
 * @param S Output array (n) - Singular Values
 * @param V Output matrix (n*n) - Right Singular Vectors
 * @param num_threads Number of threads to use
 */
void svd_decomposition_parallel(int m, int n, const double* A, double* U, double* S, double* V, int num_threads) {
    // Wrap A to double** for parallel functions
    double** A_rows = (double**)malloc(m * sizeof(double*));
    for(int i=0; i<m; i++) A_rows[i] = (double*)&A[i * n];

    // 1. Compute B = A^T using transpose_parallel
    // transpose_parallel returns a newly allocated double** (jagged array)
    double** B_rows = transpose_parallel(A_rows, m, n, num_threads);

    // 2. Compute C = B * A (n x n) using matmul_dns
    // Allocate C flat for later use, and wrap it for matmul_dns
    double* C = (double*)malloc(n * n * sizeof(double));
    double** C_rows = (double**)malloc(n * sizeof(double*));
    for(int i=0; i<n; i++) C_rows[i] = &C[i * n];

    // C = A^T * A
    matmul_dns(B_rows, A_rows, C_rows, n, m, n, num_threads);

    // Free B_rows (jagged array from transpose_parallel)
    for(int i=0; i<n; i++) free(B_rows[i]);
    free(B_rows);
    free(C_rows); // Only free the pointer array

    // 3. Compute Eigenvalues and Eigenvectors of C (Serial Jacobi)
    // C = V_eig * W * V_eig^T
    double* W = (double*)malloc(n * sizeof(double));
    double* V_eig = (double*)malloc(n * n * sizeof(double));
    
    compute_eigenvalues_parallel(n, C, W, V_eig);
    free(C);

    // 4. Calculate Singular Values S = sqrt(W) and Sort
    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; ++i) {
        if (W[i] < 0) W[i] = 0.0;
        S[i] = sqrt(W[i]);
    }

    // Sort S descending using parallel argsort
    double* sorted_indices = argsort_parallel(S, n, num_threads);

    // Reorder S and V based on sorted indices
    double* S_sorted = (double*)malloc(n * sizeof(double));
    double* V_sorted = (double*)malloc(n * n * sizeof(double));

    #pragma omp parallel for num_threads(num_threads)
    for (int i = 0; i < n; ++i) {
        int idx = (int)sorted_indices[i];
        S_sorted[i] = S[idx];
        
        // Copy column idx of V_eig to column i of V_sorted
        for (int r = 0; r < n; ++r) {
            MAT_AT(V_sorted, r, i, n) = MAT_AT(V_eig, r, idx, n);
        }
    }

    // Update S and V outputs
    memcpy(S, S_sorted, n * sizeof(double));
    memcpy(V, V_sorted, n * n * sizeof(double));

    free(W); free(V_eig); free(sorted_indices); free(S_sorted);

    // 5. Compute U = A * V * S^-1
    // First compute Temp = A * V using matmul_dns
    // We can compute directly into U (m x n)
    
    // Wrap V (which is V_sorted now)
    double** V_rows = (double**)malloc(n * sizeof(double*));
    for(int i=0; i<n; i++) V_rows[i] = &V[i * n];

    // Wrap U for result
    double** U_rows = (double**)malloc(m * sizeof(double*));
    for(int i=0; i<m; i++) U_rows[i] = &U[i * n];

    // U = A * V
    matmul_dns(A_rows, V_rows, U_rows, m, n, n, num_threads);

    // Scale U columns by S^-1
    #pragma omp parallel for collapse(2) num_threads(num_threads)
    for(int j=0; j<n; ++j) {
        for(int i=0; i<m; ++i) {
            if (S[j] > TOLERANCE) {
                U[i*n + j] /= S[j];
            } else {
                U[i*n + j] = 0.0;
            }
        }
    }

    free(A_rows);
    free(V_rows);
    free(U_rows);
    free(V_sorted);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

#ifdef STANDALONE
int main(int argc, char** argv) {
    const char* output_file = "data/output/SVD_parallel.csv";
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error opening output file %s\n", output_file);
        return 1;
    }
    fprintf(fp, "matrix_id,rows,cols,time_seconds\n");

    int num_threads = 8; // Default as requested
    int start_idx = 0;
    int end_idx = 7;
    char input_path[MAX_PATH_LEN];

    if (argc > 1) {
        // Run single file
        strncpy(input_path, argv[1], MAX_PATH_LEN);
        start_idx = 0;
        end_idx = 1;
    }

    for (int i = start_idx; i < end_idx; ++i) {
        if (argc <= 1) {
            snprintf(input_path, MAX_PATH_LEN, "data/input/matrix_%d.txt", i);
        }
        
        double* A;
        int m, n;
        
        // Use stream_matrix_reader (adapting double** to double*)
        double** A_2d;
        stream_matrix_reader(input_path, &A_2d, &m, &n);
        
        // Flatten
        A = (double*)malloc(m * n * sizeof(double));
        for(int r=0; r<m; ++r) {
            for(int c=0; c<n; ++c) {
                A[r*n + c] = A_2d[r][c];
            }
        }
        free_matrix(A_2d, m);

        printf("Processing %s (%dx%d) with %d threads...\n", input_path, m, n, num_threads);

        double* U = (double*)malloc(m * n * sizeof(double));
        double* S = (double*)malloc(n * sizeof(double));
        double* V = (double*)malloc(n * n * sizeof(double));

        double start = omp_get_wtime();
        svd_decomposition_parallel(m, n, A, U, S, V, num_threads);
        double end = omp_get_wtime();
        
        fprintf(fp, "%s,%d,%d,%.6f\n", (argc > 1 ? "custom" : input_path), m, n, end - start);
        printf("  Time: %.6fs\n", end - start);

        free(A); free(U); free(S); free(V);
    }

    fclose(fp);
    printf("Results saved to %s\n", output_file);
    return 0;
}
#endif

