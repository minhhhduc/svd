#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>
#include "../include/SVD_parallel.h"
#include "../include/mulmat.h"
#include "../include/stream.h"
#include "../include/other_operating_parallel.h"
#include "../include/norm_reducing_jacobi_parallel.h"

/* ============================================================================
 * CONFIGURATION CONSTANTS
 * ============================================================================ */

#define MAX_PATH_LEN        256

#define MAT_AT(arr, row, col, n) ((arr)[(row) * (n) + (col)])

/* ============================================================================
 * MAIN TEST FUNCTION
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
