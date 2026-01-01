#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include "../include/mulmat.h"
#include "../include/stream.h"
#include "../include/other_operating_serial.h"
#include "../include/norm_reducing_jacobi_serial.h"

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
 * SVD IMPLEMENTATION
 * ============================================================================ */

/**
 * @brief Computes SVD: A = U * S * V^T
 * @param m Rows of A
 * @param n Cols of A
 * @param A Input matrix (flat array m*n)
 * @param U Output matrix (m*n) - Left Singular Vectors
 * @param S Output array (n) - Singular Values
 * @param V Output matrix (n*n) - Right Singular Vectors
 */
void svd_decomposition(int m, int n, const double* A, double* U, double* S, double* V) {
    // 1. Compute B = A^T
    double* B = (double*)malloc(n * m * sizeof(double));
    // Note: transpose_serial expects double**, but we have flat double*. 
    // We need to adapt or use a flat transpose. 
    // Since other_operating_serial.h defines double**, let's implement a local flat transpose or adapt.
    // Actually, let's just implement a simple flat transpose here or use the one from mulmat if available.
    // mulmat.h has transpose_mat(double**).
    // Let's implement a simple flat transpose loop here to avoid double** conversion overhead.
    for(int i=0; i<m; ++i) {
        for(int j=0; j<n; ++j) {
            B[j*m + i] = A[i*n + j];
        }
    }

    // 2. Compute C = A^T * A = B * A (n x n)
    double* C = (double*)malloc(n * n * sizeof(double));
    
    // We need to use matmul_mono or similar. But they take double**.
    // Let's wrap flat arrays to double** for the library calls.
    double** A_rows = (double**)malloc(m * sizeof(double*));
    for(int i=0; i<m; i++) A_rows[i] = (double*)&A[i * n];
    
    double** B_rows = (double**)malloc(n * sizeof(double*));
    for(int i=0; i<n; i++) B_rows[i] = (double*)&B[i * m];
    
    double** C_rows = (double**)malloc(n * sizeof(double*));
    for(int i=0; i<n; i++) C_rows[i] = &C[i * n];

    // Use serial multiplication for SVD_serial
    matmul_mono(B_rows, A_rows, C_rows, n, m, n);

    free(A_rows); free(B_rows); free(C_rows);

    // 3. Compute Eigenvalues and Eigenvectors of C
    // C = V_eig * W * V_eig^T
    double* W = (double*)malloc(n * sizeof(double));
    double* V_eig = (double*)malloc(n * n * sizeof(double));
    
    // Use the serial Jacobi implementation
    compute_eigenvalues(n, C, W, V_eig);

    // 4. Calculate Singular Values S = sqrt(W) and Sort
    for (int i = 0; i < n; ++i) {
        if (W[i] < 0) W[i] = 0.0;
        S[i] = sqrt(W[i]);
    }

    // Sort S descending using argsort_serial
    double* sorted_indices = argsort_serial(S, n);

    // Reorder S and V based on sorted indices
    double* S_sorted = (double*)malloc(n * sizeof(double));
    double* V_sorted = (double*)malloc(n * n * sizeof(double));

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

    // 5. Compute U = A * V * S^-1
    // U is m x n
    // u_j = (1/s_j) * A * v_j
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            double sum = 0.0;
            for (int k = 0; k < n; ++k) {
                sum += MAT_AT(A, i, k, n) * MAT_AT(V, k, j, n);
            }
            
            if (S[j] > TOLERANCE) {
                MAT_AT(U, i, j, n) = sum / S[j];
            } else {
                MAT_AT(U, i, j, n) = 0.0;
            }
        }
    }

    free(B); free(C); free(W); free(V_eig);
    free(sorted_indices); free(S_sorted); free(V_sorted);
}

/* ============================================================================
 * MAIN
 * ============================================================================ */

#ifdef STANDALONE
int main(int argc, char** argv) {
    const char* output_file = "data/output/SVD_serial.csv";
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error opening output file %s\n", output_file);
        return 1;
    }
    fprintf(fp, "matrix_id,rows,cols,time_seconds\n");

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

        printf("Processing %s (%dx%d)...\n", input_path, m, n);

        double* U = (double*)malloc(m * n * sizeof(double));
        double* S = (double*)malloc(n * sizeof(double));
        double* V = (double*)malloc(n * n * sizeof(double));

        double start = omp_get_wtime();
        svd_decomposition(m, n, A, U, S, V);
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
