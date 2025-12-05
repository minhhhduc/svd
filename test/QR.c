#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <omp.h>
#include "../include/stream.h"
#include "../include/mulmat.h"

#define NUM_THREADS 8

// Helper to allocate matrix
double** allocate_matrix(int rows, int cols) {
    double** data = (double**)malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        data[i] = (double*)calloc(cols, sizeof(double));
    }
    return data;
}

// Helper to copy matrix
void copy_matrix(double** src, double** dest, int rows, int cols) {
    #pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        memcpy(dest[i], src[i], cols * sizeof(double));
    }
}

// Helper for identity
void set_identity(double** mat, int n) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            mat[i][j] = (i == j) ? 1.0 : 0.0;
        }
    }
}

// QR Decomposition using Householder reflections (Optimized Parallel - Implicit Update & Row-wise Access)
void qr_decomposition_parallel(double** A, double** Q, double** R, int m, int n) {
    // Initialize Q = I, R = A
    set_identity(Q, m);
    copy_matrix(A, R, m, n);

    double* v = (double*)malloc(m * sizeof(double));
    double* w = (double*)malloc(n * sizeof(double)); // Workspace for v^T * R
    double* y = (double*)malloc(m * sizeof(double)); // Workspace for Q * v
    
    // Buffer for parallel reduction of w
    int max_threads = omp_get_max_threads();
    double* w_buffers = (double*)malloc(max_threads * n * sizeof(double));

    for (int k = 0; k < n && k < m - 1; k++) {
        // 1. Compute Householder vector v
        double norm_x = 0.0;
        #pragma omp parallel for reduction(+:norm_x)
        for (int i = k; i < m; i++) {
            norm_x += R[i][k] * R[i][k];
        }
        norm_x = sqrt(norm_x);

        double alpha = -((R[k][k] >= 0) ? 1.0 : -1.0) * norm_x;
        
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            if (i < k) v[i] = 0.0;
            else v[i] = R[i][k];
        }
        v[k] -= alpha;

        double norm_v = 0.0;
        #pragma omp parallel for reduction(+:norm_v)
        for (int i = k; i < m; i++) norm_v += v[i] * v[i];
        norm_v = sqrt(norm_v);

        if (norm_v < 1e-10) continue;

        double inv_norm_v = 1.0 / norm_v;
        #pragma omp parallel for
        for (int i = k; i < m; i++) v[i] *= inv_norm_v;

        // 2. Apply H to R: R = R - 2 * v * (v^T * R)
        // Compute w = v^T * R using row-wise access to R (cache friendly)
        // We accumulate partial results in w_buffers
        
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            double* w_local = &w_buffers[tid * n];
            
            // Reset local buffer for relevant columns
            for (int j = k; j < n; j++) w_local[j] = 0.0;

            // Each thread processes a chunk of rows of R
            #pragma omp for nowait
            for (int i = k; i < m; i++) {
                double vi = v[i];
                double* Ri = R[i];
                for (int j = k; j < n; j++) {
                    w_local[j] += vi * Ri[j];
                }
            }
        }

        // Reduce w_buffers to w
        #pragma omp parallel for
        for (int j = k; j < n; j++) {
            double sum = 0.0;
            // Note: We must sum over all potential threads. 
            // Since we don't know exactly how many threads ran in the previous region (could be dynamic),
            // we assume up to max_threads but check if we can optimize.
            // For safety with default OMP settings, iterating up to max_threads is safe if we allocated enough.
            // However, if fewer threads were used, their buffers might be uninitialized?
            // No, we only initialized buffers for threads that ran.
            // Wait, this is risky if omp_get_num_threads varies.
            // Let's force num_threads or assume fixed.
            // Better: Initialize ALL buffers to 0 once? No, too slow.
            // Correct approach: Use reduction clause on array? OpenMP 4.5+ supports it.
            // Fallback: Just use the previous inefficient column-wise loop if this is too complex?
            // No, let's use the fact that we set NUM_THREADS=8 globally or use omp_get_num_threads() from a parallel region.
            
            // Let's use a simpler approach: Parallel region for both accumulation and reduction.
        }
        
        // Retry Step 2 with safer logic
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nthreads = omp_get_num_threads();
            double* w_local = &w_buffers[tid * n];
            
            // Reset
            for (int j = k; j < n; j++) w_local[j] = 0.0;
            
            #pragma omp for
            for (int i = k; i < m; i++) {
                double vi = v[i];
                double* Ri = R[i];
                for (int j = k; j < n; j++) {
                    w_local[j] += vi * Ri[j];
                }
            }
            
            // Barrier implicit at end of for
            
            // Reduce
            #pragma omp for
            for (int j = k; j < n; j++) {
                double sum = 0.0;
                for (int t = 0; t < nthreads; t++) {
                    sum += w_buffers[t * n + j];
                }
                w[j] = sum;
            }
        }

        // Update R
        #pragma omp parallel for collapse(2)
        for (int i = k; i < m; i++) {
            for (int j = k; j < n; j++) {
                R[i][j] -= 2.0 * v[i] * w[j];
            }
        }

        // 3. Apply H to Q: Q = Q * H = Q - 2 * (Q * v) * v^T
        // Compute y = Q * v (m x 1 column vector)
        // Q is accessed row-wise, which is good.
        
        #pragma omp parallel for
        for (int i = 0; i < m; i++) {
            double sum = 0.0;
            double* Qi = Q[i];
            for (int j = k; j < m; j++) {
                sum += Qi[j] * v[j];
            }
            y[i] = sum;
        }

        // Update Q
        #pragma omp parallel for collapse(2)
        for (int i = 0; i < m; i++) {
            for (int j = k; j < m; j++) {
                Q[i][j] -= 2.0 * y[i] * v[j];
            }
        }
    }

    free(v);
    free(w);
    free(y);
    free(w_buffers);
}

// QR Algorithm for Eigenvalues (Parallelized using DNS)
void qr_algorithm_eigen_parallel(double** A, double* eigenvalues, double** eigenvectors, int n, int max_iter) {
    double** Ak = allocate_matrix(n, n);
    double** Q = allocate_matrix(n, n);
    double** R = allocate_matrix(n, n);
    double** V = allocate_matrix(n, n);
    double** temp = allocate_matrix(n, n);

    copy_matrix(A, Ak, n, n);
    set_identity(V, n);

    for (int iter = 0; iter < max_iter; iter++) {
        qr_decomposition_parallel(Ak, Q, R, n, n);

        // Ak = R * Q
        matmul_dns(R, Q, Ak, n, n, n, NUM_THREADS);

        // V = V * Q
        matmul_dns(V, Q, temp, n, n, n, NUM_THREADS);
        copy_matrix(temp, V, n, n);
        
        // Check convergence (off-diagonal elements)
        double off_diag_norm = 0.0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < i; j++) {
                off_diag_norm += Ak[i][j] * Ak[i][j];
            }
        }
        if (sqrt(off_diag_norm) < 1e-9) {
            break;
        }
    }

    // Extract eigenvalues
    for (int i = 0; i < n; i++) {
        eigenvalues[i] = Ak[i][i];
    }

    // Copy eigenvectors
    copy_matrix(V, eigenvectors, n, n);

    free_matrix(Ak, n);
    free_matrix(Q, n);
    free_matrix(R, n);
    free_matrix(V, n);
    free_matrix(temp, n);
}

int main() {
    int num_files = 21;
    char filename[256];
    
    // Open output file
    const char* output_filename = "data/output/qr_output.txt";
    FILE* fout = fopen(output_filename, "w");
    if (!fout) {
        // Try with ../ prefix
        fout = fopen("../data/output/qr_output.txt", "w");
        if (!fout) {
            // Try absolute path or just skip file writing
            printf("Warning: Could not open output file.\n");
        }
    }
    
    if (fout) {
        fprintf(fout, "rows,cols,qr_time,reconstruction_error,eigen_time\n");
    }

    printf("=== Parallel QR Algorithm Test (DNS, %d threads) ===\n", NUM_THREADS);

    for (int i = 0; i < num_files; i++) {
        sprintf(filename, "data/input/matrix_%d.txt", i);
        
        double** A = NULL;
        int rows, cols;
        
        // Check if file exists
        FILE* f = fopen(filename, "r");
        if (!f) {
            // Try with ../ prefix
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
        stream_matrix_reader(filename, &A, &rows, &cols);
        printf("Dimensions: %d x %d\n", rows, cols);

        // Perform QR Decomposition
        double** Q = allocate_matrix(rows, rows);
        double** R = allocate_matrix(rows, cols);
        
        double start_time = omp_get_wtime();
        qr_decomposition_parallel(A, Q, R, rows, cols);
        double end_time = omp_get_wtime();
        double qr_time = end_time - start_time;
        printf("QR Decomposition Time: %.4f seconds\n", qr_time);

        // Verify QR = A
        double** QR = allocate_matrix(rows, cols);
        matmul_dns(Q, R, QR, rows, rows, cols, NUM_THREADS);
        
        double error = 0.0;
        for(int r=0; r<rows; r++) {
            for(int c=0; c<cols; c++) {
                double diff = A[r][c] - QR[r][c];
                error += diff * diff;
            }
        }
        printf("Reconstruction Error (||A - QR||^2): %.6e\n", error);

        free_matrix(Q, rows);
        free_matrix(R, rows);
        free_matrix(QR, rows);

        double eigen_time = -1.0;

        // Compute S = A^T * A for eigenvalue calculation (SVD related)
        // A is rows x cols, A^T is cols x rows
        // S is cols x cols (symmetric)
        int n = cols;
        double** S = allocate_matrix(n, n);
        
        // We need A^T first
        double** AT = allocate_matrix(cols, rows);
        #pragma omp parallel for collapse(2)
        for(int r=0; r<rows; r++) {
            for(int c=0; c<cols; c++) {
                AT[c][r] = A[r][c];
            }
        }
        
        // S = AT * A
        // Use matmul_dns or parallel
        matmul_dns(AT, A, S, cols, rows, cols, NUM_THREADS);
        
        double* eigenvalues = (double*)malloc(n * sizeof(double));
        double** eigenvectors = allocate_matrix(n, n);

        start_time = omp_get_wtime();
        qr_algorithm_eigen_parallel(S, eigenvalues, eigenvectors, n, 100); // 100 iterations max
        end_time = omp_get_wtime();
        eigen_time = end_time - start_time;

        printf("Eigenvalue Solver Time (on A^T*A): %.4f seconds\n", eigen_time);
        printf("Eigenvalues of A^T*A (first 5): ");
        for (int j = 0; j < (n < 5 ? n : 5); j++) {
            printf("%.4f ", eigenvalues[j]);
        }
        printf("\n");

        free(eigenvalues);
        free_matrix(eigenvectors, n);
        free_matrix(S, n);
        free_matrix(AT, cols);
        
        // Log to screen and file
        printf("{rows: %d, cols: %d, qr_time: %f, error: %e, eigen_time: %f}\n", rows, cols, qr_time, error, eigen_time);
        if (fout) {
            fprintf(fout, "%d,%d,%f,%e,%f\n", rows, cols, qr_time, error, eigen_time);
            fflush(fout);
        }

        free_matrix(A, rows);
    }
    
    if (fout) fclose(fout);

    return 0;
}
