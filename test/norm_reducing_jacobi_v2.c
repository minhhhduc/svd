/**
 * @file norm_reducing_jacobi_v2.c
 * @brief Parallel Symmetric Jacobi Algorithm for Eigenvalue Computation
 * @details Implementation based on standard Symmetric Jacobi method for Real Symmetric Matrices.
 *          Formulas provided by user.
 * 
 * @author Refactored with clean architecture and OpenMP parallelization
 * @date 2025
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <stdbool.h>
#include <string.h>
#include "mulmat.c"

/* ============================================================================
 * CONFIGURATION CONSTANTS
 * ============================================================================ */

#define TOLERANCE           1e-12
#define REL_TOLERANCE       1e-10
#define MAX_SWEEPS          100
#define PARALLEL_THRESHOLD  32      // Minimum n for parallelization
#define NUM_TEST_FILES      21
#define MAX_PATH_LEN        256

/* ============================================================================
 * MACROS
 * ============================================================================ */

#define MAT_AT(arr, row, col, n) ((arr)[(row) * (n) + (col)])
#define SIGN(x) ((x) >= 0.0 ? 1.0 : -1.0)

/* ============================================================================
 * DATA STRUCTURES
 * ============================================================================ */

typedef struct {
    double c; // cos(theta)
    double s; // sin(theta)
} RotationParams;

/* ============================================================================
 * FUNCTION PROTOTYPES
 * ============================================================================ */

static bool compute_rotation_parameters(double a_pp, double a_qq, double a_pq, RotationParams* params);
static void update_matrix_A(double* A, int n, int p, int q, RotationParams params);
static void update_matrix_V(double* V, int n, int p, int q, RotationParams params);
static void process_pair(double* A, double* V, int n, int p, int q);
static void perform_jacobi_sweep(double* A, double* V, int n);
static double compute_off_diagonal_norm(const double* A, int n);
static void stream_matrix_reader_double(const char* filename, double** data, int* rows, int* cols);
static void matmul_double(const double* A, const double* B, double* C, int n, int m, int p);
static void transpose_mat_double(const double* A, double* T, int rows, int cols);

void compute_eigenvalues(int n, const double* A_in, double* w, double* V_out);

/* ============================================================================
 * CORE ALGORITHM FUNCTIONS
 * ============================================================================ */

/**
 * @brief Compute rotation parameters (c, s) to annihilate A_pq
 * @details Based on formulas:
 *          tau = (A_qq - A_pp) / (2 * A_pq)
 *          t = sgn(tau) / (|tau| + sqrt(1 + tau^2))
 *          c = 1 / sqrt(1 + t^2)
 *          s = t * c
 */
static bool compute_rotation_parameters(
    double a_pp,
    double a_qq,
    double a_pq,
    RotationParams* params
) {
    if (fabs(a_pq) < TOLERANCE) {
        params->c = 1.0;
        params->s = 0.0;
        return false; // No rotation needed
    }

    double tau = (a_qq - a_pp) / (2.0 * a_pq);
    double t;
    
    if (tau >= 0) {
        t = 1.0 / (tau + sqrt(1.0 + tau * tau));
    } else {
        t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
    }

    params->c = 1.0 / sqrt(1.0 + t * t);
    params->s = t * params->c;

    return true;
}

/* ============================================================================
 * PARALLEL SWEEP LOGIC
 * ============================================================================ */

/**
 * @brief Perform one full Jacobi sweep
 * @details Uses the "Music Festival" (Round-Robin) schedule to enable full parallelization.
 *          To avoid race conditions on the shared matrix A, the updates are split into:
 *          1. Compute all rotations for disjoint pairs.
 *          2. Apply row rotations (A' = J^T A) in parallel.
 *          3. Apply column rotations (A'' = A' J) in parallel.
 *          4. Update eigenvectors V in parallel.
 */
static void perform_jacobi_sweep(
    double* A,
    double* V,
    int n
) {
    int* idx = (int*)malloc(n * sizeof(int));
    if (!idx) return;
    for (int i = 0; i < n; ++i) idx[i] = i;

    int n_pairs = n / 2;
    int* p_indices = (int*)malloc(n_pairs * sizeof(int));
    int* q_indices = (int*)malloc(n_pairs * sizeof(int));
    RotationParams* rot_params = (RotationParams*)malloc(n_pairs * sizeof(RotationParams));

    if (!p_indices || !q_indices || !rot_params) {
        free(idx);
        if (p_indices) free(p_indices);
        if (q_indices) free(q_indices);
        if (rot_params) free(rot_params);
        return;
    }

    // Number of steps in a sweep. 
    // For even n: n-1 steps to cover all pairs.
    // For odd n: n steps.
    int steps = (n % 2 == 0) ? (n - 1) : n;

    for (int step = 0; step < steps; ++step) {
        
        // --- 1. Form Pairs (Serial) ---
        if (n % 2 == 0) {
            // Even n: Fix idx[0], pair others.
            // Pairs: (idx[0], idx[n-1]), (idx[1], idx[n-2]), ...
            for (int k = 0; k < n_pairs; ++k) {
                int p = idx[k];
                int q = idx[n - 1 - k];
                if (p > q) { int t = p; p = q; q = t; }
                p_indices[k] = p;
                q_indices[k] = q;
            }
        } else {
            // Odd n: Cycle all.
            // Pairs: (idx[0], idx[n-1]), (idx[1], idx[n-2]), ...
            // idx[n/2] sits out.
            for (int k = 0; k < n_pairs; ++k) {
                int p = idx[k];
                int q = idx[n - 1 - k];
                if (p > q) { int t = p; p = q; q = t; }
                p_indices[k] = p;
                q_indices[k] = q;
            }
        }

        // --- 2. Compute Rotations (Parallel) ---
        #pragma omp parallel for if(n > PARALLEL_THRESHOLD)
        for (int k = 0; k < n_pairs; ++k) {
            int p = p_indices[k];
            int q = q_indices[k];
            double app = MAT_AT(A, p, p, n);
            double aqq = MAT_AT(A, q, q, n);
            double apq = MAT_AT(A, p, q, n);
            compute_rotation_parameters(app, aqq, apq, &rot_params[k]);
        }

        // --- 3. Apply Row Rotations (Parallel) ---
        // Update rows p and q for all columns j
        #pragma omp parallel for if(n > PARALLEL_THRESHOLD)
        for (int k = 0; k < n_pairs; ++k) {
            int p = p_indices[k];
            int q = q_indices[k];
            RotationParams params = rot_params[k];
            double c = params.c;
            double s = params.s;

            // If rotation is trivial, we could skip, but checking might be costlier
            if (fabs(s) > 0.0) {
                for (int j = 0; j < n; ++j) {
                    double apj = MAT_AT(A, p, j, n);
                    double aqj = MAT_AT(A, q, j, n);
                    MAT_AT(A, p, j, n) = c * apj - s * aqj;
                    MAT_AT(A, q, j, n) = s * apj + c * aqj;
                }
            }
        }

        // --- 4. Apply Column Rotations (Parallel) ---
        // Update cols p and q for all rows i
        #pragma omp parallel for if(n > PARALLEL_THRESHOLD)
        for (int k = 0; k < n_pairs; ++k) {
            int p = p_indices[k];
            int q = q_indices[k];
            RotationParams params = rot_params[k];
            double c = params.c;
            double s = params.s;

            if (fabs(s) > 0.0) {
                for (int i = 0; i < n; ++i) {
                    double aip = MAT_AT(A, i, p, n);
                    double aiq = MAT_AT(A, i, q, n);
                    MAT_AT(A, i, p, n) = c * aip - s * aiq;
                    MAT_AT(A, i, q, n) = s * aip + c * aiq;
                }
                // Explicitly zero out the off-diagonal elements to prevent drift
                MAT_AT(A, p, q, n) = 0.0;
                MAT_AT(A, q, p, n) = 0.0;
            }
        }

        // --- 5. Update Eigenvectors V (Parallel) ---
        #pragma omp parallel for if(n > PARALLEL_THRESHOLD)
        for (int k = 0; k < n_pairs; ++k) {
            int p = p_indices[k];
            int q = q_indices[k];
            RotationParams params = rot_params[k];
            double c = params.c;
            double s = params.s;

            if (fabs(s) > 0.0) {
                for (int i = 0; i < n; ++i) {
                    double vip = MAT_AT(V, i, p, n);
                    double viq = MAT_AT(V, i, q, n);
                    MAT_AT(V, i, p, n) = c * vip - s * viq;
                    MAT_AT(V, i, q, n) = s * vip + c * viq;
                }
            }
        }

        // --- 6. Rotate Indices ---
        if (n % 2 == 0) {
            // Fix idx[0], rotate idx[1]..idx[n-1]
            int temp = idx[1];
            for (int i = 1; i < n - 1; ++i) {
                idx[i] = idx[i + 1];
            }
            idx[n - 1] = temp;
        } else {
            // Rotate all
            int temp = idx[0];
            for (int i = 0; i < n - 1; ++i) {
                idx[i] = idx[i + 1];
            }
            idx[n - 1] = temp;
        }
    }

    free(idx);
    free(p_indices);
    free(q_indices);
    free(rot_params);
}

/* ============================================================================
 * CONVERGENCE CHECK
 * ============================================================================ */

static double compute_off_diagonal_norm(const double* A, int n) {
    double sum_sq = 0.0;
    #pragma omp parallel for reduction(+:sum_sq) if(n > PARALLEL_THRESHOLD)
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            if (i != j) {
                double val = MAT_AT(A, i, j, n);
                sum_sq += val * val;
            }
        }
    }
    return sqrt(sum_sq);
}

/* ============================================================================
 * MAIN SOLVER
 * ============================================================================ */

void compute_eigenvalues(
    int n,
    const double* A_in,
    double* w,
    double* V_out
) {
    // Allocate and initialize A and V
    double* A = (double*)malloc(n * n * sizeof(double));
    double* V = (double*)malloc(n * n * sizeof(double));
    
    if (!A || !V) {
        if (A) free(A);
        if (V) free(V);
        return;
    }

    // Initialize A = A_in
    #pragma omp parallel for
    for (int i = 0; i < n * n; ++i) {
        A[i] = A_in[i];
    }

    // Initialize V = Identity
    #pragma omp parallel for
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            MAT_AT(V, i, j, n) = (i == j) ? 1.0 : 0.0;
        }
    }

    // Main Loop
    double prev_off_norm = -1.0;
    for (int sweep = 0; sweep < MAX_SWEEPS; ++sweep) {
        double off_norm = compute_off_diagonal_norm(A, n);
        
        if (sweep % 1 == 0) {
            printf("Sweep %d: ||Off-Diagonal||_F = %.6e\n", sweep, off_norm);
        }

        if (off_norm < TOLERANCE) {
            printf("Converged at sweep %d.\n", sweep);
            break;
        }

        if (prev_off_norm > 0.0 && fabs(prev_off_norm - off_norm) < 1e-15) {
            printf("Converged (stagnation) at sweep %d. Norm stopped changing.\n", sweep);
            break;
        }
        prev_off_norm = off_norm;

        perform_jacobi_sweep(A, V, n);
    }

    // Extract results
    for (int i = 0; i < n; ++i) {
        w[i] = MAT_AT(A, i, i, n);
    }
    for (int i = 0; i < n * n; ++i) {
        V_out[i] = V[i];
    }

    free(A);
    free(V);
}

/* ============================================================================
 * I/O AND HELPERS
 * ============================================================================ */

static void stream_matrix_reader_double(const char* filename, double** data, int* rows, int* cols) {
    FILE* file = fopen(filename, "r");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        exit(EXIT_FAILURE);
    }

    if (fscanf(file, "%d %d", rows, cols) != 2) {
        fprintf(stderr, "Error: invalid matrix size format.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    *data = (double*)malloc((*rows) * (*cols) * sizeof(double));
    if (*data == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < *rows; ++i) {
        for (int j = 0; j < *cols; ++j) {
            if (fscanf(file, "%lf", &(*data)[i * (*cols) + j]) != 1) {
                fprintf(stderr, "Error reading element at %d,%d\n", i, j);
                fclose(file);
                exit(EXIT_FAILURE);
            }
        }
    }
    fclose(file);
}

static void matmul_double(const double* A, const double* B, double* C, int n, int m, int p_dim) {
    // Adapter for matmul_dns from mulmat.c
    
    // 1. Create row pointers for A (n x m)
    double** A_rows = (double**)malloc(n * sizeof(double*));
    if (!A_rows) return;
    for(int i=0; i<n; i++) A_rows[i] = (double*)&A[i * m];

    // 2. Create row pointers for B (m x p_dim)
    double** B_rows = (double**)malloc(m * sizeof(double*));
    if (!B_rows) { free(A_rows); return; }
    for(int i=0; i<m; i++) B_rows[i] = (double*)&B[i * p_dim];

    // 3. Create row pointers for C (n x p_dim)
    double** C_rows = (double**)malloc(n * sizeof(double*));
    if (!C_rows) { free(A_rows); free(B_rows); return; }
    for(int i=0; i<n; i++) C_rows[i] = &C[i * p_dim];

    // 4. Determine threads
    int max_threads = omp_get_max_threads();
    
    // 5. Call DNS
    matmul_dns(A_rows, B_rows, C_rows, n, m, p_dim, max_threads);

    // 6. Cleanup pointers
    free(A_rows);
    free(B_rows);
    free(C_rows);
}

static void transpose_mat_double(const double* A, double* T, int rows, int cols) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            T[j * rows + i] = A[i * cols + j];
        }
    }
}

/* ============================================================================
 * BENCHMARKING
 * ============================================================================ */

void calculate_time_inparallel(
    const char** input_filename,
    const char* output_filename,
    int quantity_files
) {
    FILE* fout = fopen(output_filename, "w");
    if (fout == NULL) {
        fprintf(stderr, "Cannot open output file %s\n", output_filename);
        return;
    }
    
    fprintf(fout, "matrix_size,computation_time_seconds\n");
    
    for (int file_idx = 1; file_idx < quantity_files; file_idx++) {
        double* A_matrix_flat;
        int rows, cols;
        stream_matrix_reader_double(input_filename[file_idx], &A_matrix_flat, &rows, &cols);
        
        // Compute C = A^T * A
        double* B_flat = (double*)malloc(cols * rows * sizeof(double));
        transpose_mat_double(A_matrix_flat, B_flat, rows, cols);
        
        double* C_flat = (double*)malloc(cols * cols * sizeof(double));
        
        printf("Processing matrix %d (%dx%d)...\n", file_idx, rows, cols);
        printf("Computing C = A^T * A...\n");
        matmul_double(B_flat, A_matrix_flat, C_flat, cols, rows, cols);
        
        int n = cols;
        double* w = (double*)malloc(n * sizeof(double));
        double* V = (double*)malloc(n * n * sizeof(double));
        
        printf("Computing eigenvalues for %dx%d matrix...\n", n, n);
        double start_time = omp_get_wtime();
        compute_eigenvalues(n, C_flat, w, V);
        double end_time = omp_get_wtime();
        
        double computation_time = end_time - start_time;
        printf("Total computation time: %.6f seconds\n", computation_time);
        fprintf(fout, "%d,%.6f\n", n, computation_time);
        
        printf("Eigenvalues:\n");
        for (int i = 0; i < n; ++i) {
            printf("w[%d] = %.10f\n", i, w[i]);
        }
        
        free(A_matrix_flat);
        free(B_flat);
        free(C_flat);
        free(w);
        free(V);
    }
    
    fclose(fout);
}

int main(void) {
    const char* path_input = "./data/input";
    const char output_filename[] = "./data/output/norm_reducing_jacobi_output.txt";
    
    // Allocate array of strings for filenames
    char** input_filename = (char**)malloc(NUM_TEST_FILES * sizeof(char*));
    if (!input_filename) {
        fprintf(stderr, "Memory allocation failed for filenames.\n");
        return EXIT_FAILURE;
    }

    for (int i = 0; i < NUM_TEST_FILES; i++) {
        input_filename[i] = (char*)malloc(MAX_PATH_LEN * sizeof(char));
        if (!input_filename[i]) {
            fprintf(stderr, "Memory allocation failed for filename %d.\n", i);
            // Cleanup previously allocated
            for (int j = 0; j < i; j++) free(input_filename[j]);
            free(input_filename);
            return EXIT_FAILURE;
        }
        snprintf(input_filename[i], MAX_PATH_LEN, "%s/matrix_%d.txt", path_input, i);
    }
    
    calculate_time_inparallel((const char**)input_filename, output_filename, NUM_TEST_FILES);
    
    // Cleanup
    for (int i = 0; i < NUM_TEST_FILES; i++) {
        free(input_filename[i]);
    }
    free(input_filename);
    
    return EXIT_SUCCESS;
}
