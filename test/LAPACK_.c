#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <omp.h>
#include "../include/stream.h"

// LAPACK SVD prototype
// dgesvd: Computes the singular value decomposition (SVD) of a real M-by-N matrix A
// A = U * SIGMA * V^T
extern void dgesvd_(char *jobu, char *jobvt, int *m, int *n, double *a, int *lda,
                    double *s, double *u, int *ldu, double *vt, int *ldvt,
                    double *work, int *lwork, int *info,
                    size_t len_jobu, size_t len_jobvt);

// BLAS dgemm for matrix multiplication
extern void dgemm_(char *transa, char *transb, int *m, int *n, int *k,
                   double *alpha, double *a, int *lda,
                   double *b, int *ldb,
                   double *beta, double *c, int *ldc,
                   size_t len_transa, size_t len_transb);

// Function to perform SVD + Projection using LAPACK (similar to decompose.c)
void lapack_decompose_project(int m, int n, int k, const double* A, double* Result) {
    // Convert to column-major for LAPACK
    double* A_col = (double*)malloc(m * n * sizeof(double));
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < m; i++) {
            A_col[j * m + i] = A[i * n + j];
        }
    }

    // Allocate for SVD
    double* S = (double*)malloc(n * sizeof(double));
    double* VT = (double*)malloc(n * n * sizeof(double));  // V^T in column-major
    
    // We need U for projection, allocate full U (m x n)
    double* U = (double*)malloc(m * n * sizeof(double));

    char jobu = 'S';   // Compute first min(m,n) columns of U
    char jobvt = 'S';  // Compute first min(m,n) rows of V^T
    int lda = m;
    int ldu = m;
    int ldvt = n;
    int info;
    int lwork = -1;
    double wkopt;

    // Query workspace
    dgesvd_(&jobu, &jobvt, &m, &n, A_col, &lda, S, U, &ldu, VT, &ldvt,
            &wkopt, &lwork, &info, 1, 1);

    lwork = (int)wkopt;
    double* work = (double*)malloc(lwork * sizeof(double));

    // Compute SVD: A = U * S * V^T
    dgesvd_(&jobu, &jobvt, &m, &n, A_col, &lda, S, U, &ldu, VT, &ldvt,
            work, &lwork, &info, 1, 1);

    if (info != 0) {
        printf("LAPACK dgesvd error: %d\n", info);
        free(A_col); free(S); free(VT); free(U); free(work);
        return;
    }

    // Project: Result = A * V[:, :k]
    // V^T is stored in VT (column-major), so V = VT^T
    // We need first k columns of V, which are first k rows of VT
    
    // Extract V_k (n x k) from VT - VT is (n x n) in column-major
    // V[i][j] = VT[j][i] = VT[j * n + i] (column-major)
    // V_k[i][j] for j < k means V[i][j] = VT[j * n + i]
    double* V_k = (double*)malloc(n * k * sizeof(double));
    for(int j = 0; j < k; j++) {
        for(int i = 0; i < n; i++) {
            // V_k in column-major: V_k[j * n + i] = V[i][j] = VT[j][i] = VT[i * n + j]
            V_k[j * n + i] = VT[i * n + j];
        }
    }

    // Reload A_col since dgesvd destroyed it
    for(int j = 0; j < n; j++) {
        for(int i = 0; i < m; i++) {
            A_col[j * m + i] = A[i * n + j];
        }
    }

    // Result = A * V_k using dgemm
    // A is (m x n), V_k is (n x k), Result is (m x k)
    // All in column-major
    double* Result_col = (double*)malloc(m * k * sizeof(double));
    
    char transa = 'N';
    char transb = 'N';
    int m_gemm = m;
    int n_gemm = k;
    int k_gemm = n;
    double alpha = 1.0;
    double beta = 0.0;

    dgemm_(&transa, &transb, &m_gemm, &n_gemm, &k_gemm, &alpha, 
           A_col, &m, V_k, &n, &beta, Result_col, &m, 1, 1);

    // Convert Result back to row-major
    for(int j = 0; j < k; j++) {
        for(int i = 0; i < m; i++) {
            Result[i * k + j] = Result_col[j * m + i];
        }
    }

    free(A_col);
    free(S);
    free(VT);
    free(U);
    free(V_k);
    free(Result_col);
    free(work);
}

int main() {
    int start_idx = 1;
    int end_idx = 7;  // Process matrix_1 to matrix_6 (same as decompose.c)
    char filename[256];
    const char* output_filename = "data/output/lapack_svd_output.txt";
    
    FILE* fout = fopen(output_filename, "w");
    if (!fout) {
        fout = fopen("../data/output/lapack_svd_output.txt", "w");
    }
    
    if (fout) {
        fprintf(fout, "col,row,lapack_time\n");
    } else {
        printf("Warning: Could not open output file for writing.\n");
    }

    printf("=== LAPACK SVD + Projection Test ===\n");
    printf("Same computation as decompose.c but using LAPACK\n\n");

    for (int f_idx = start_idx; f_idx < end_idx; f_idx++) {
        sprintf(filename, "data/input/matrix_%d.txt", f_idx);
        
        double** A_rows = NULL;
        int m, n;
        
        FILE* f = fopen(filename, "r");
        if (!f) {
            char temp[256];
            sprintf(temp, "../%s", filename);
            f = fopen(temp, "r");
            if (!f) {
                printf("Skipping %s (not found)\n", filename);
                continue;
            }
            strcpy(filename, temp);
        }
        fclose(f);

        printf("\nProcessing %s...\n", filename);
        stream_matrix_reader(filename, &A_rows, &m, &n);
        printf("Dimensions: %d x %d\n", m, n);

        // Flatten matrix (row-major)
        double* A = (double*)malloc(m * n * sizeof(double));
        for(int r = 0; r < m; r++) {
            for(int c = 0; c < n; c++) {
                A[r * n + c] = A_rows[r][c];
            }
        }
        free_matrix(A_rows, m);

        // Set k same as decompose.c
        int k = n / 2;
        if (k < 1) k = 1;

        printf("Projecting to k=%d components...\n", k);

        double* Result = (double*)malloc(m * k * sizeof(double));

        // Time LAPACK SVD + Projection
        double start = omp_get_wtime();
        lapack_decompose_project(m, n, k, A, Result);
        double end = omp_get_wtime();
        double lapack_time = end - start;

        printf("LAPACK Time: %.6f s\n", lapack_time);

        // Write to CSV
        if (fout) {
            fprintf(fout, "%d,%d,%.6f\n", n, m, lapack_time);
            fflush(fout);
        }

        free(A);
        free(Result);
    }

    if (fout) fclose(fout);
    printf("\nResults saved to %s\n", output_filename);
    return 0;
}
