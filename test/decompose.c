#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>
#include "../include/stream.h"
#include "../include/decompose_operation.h"

#define MAX_PATH_LEN 256
#define NUM_THREADS 12

int main(int argc, char** argv) {
    const char* output_file = "data/output/decompose.csv";
    FILE* fp = fopen(output_file, "w");
    if (!fp) {
        fprintf(stderr, "Error opening output file %s\n", output_file);
        return 1;
    }
    // Header as requested: col, row, serial, parallel
    fprintf(fp, "col,row,serial,parallel\n");

    int start_idx = 1;
    int end_idx = 7; // Process 1 to 6
    int num_threads = NUM_THREADS; // Default for parallel

    printf("Benchmarking Decompose (SVD + Projection) Serial vs Parallel (Matrices %d to %d)...\n", start_idx, end_idx - 1);

    for (int i = start_idx; i < end_idx; ++i) {
        char input_path[MAX_PATH_LEN];
        snprintf(input_path, MAX_PATH_LEN, "data/input/matrix_%d.txt", i);
        
        int m, n;
        double** A_2d;
        
        // Read matrix
        stream_matrix_reader(input_path, &A_2d, &m, &n);
        
        // Flatten for SVD functions
        double* A = (double*)malloc(m * n * sizeof(double));
        for(int r=0; r<m; ++r) {
            for(int c=0; c<n; ++c) {
                A[r*n + c] = A_2d[r][c];
            }
        }
        // Free 2D array as we have the flat copy
        free_matrix(A_2d, m);

        // Set K for projection (e.g., keep 50% of components)
        int k = n / 2;
        if (k < 1) k = 1;

        printf("Processing %s (%dx%d), projecting to k=%d...\n", input_path, m, n, k);

        // Allocate outputs
        double* Result = (double*)malloc(m * k * sizeof(double));

        // --- Serial Run ---
        double start_serial = omp_get_wtime();
        
        decompose_project_serial(m, n, k, A, Result);
        
        double end_serial = omp_get_wtime();
        double time_serial = end_serial - start_serial;
        printf("  Serial: %.6fs\n", time_serial);

        // Reset outputs
        memset(Result, 0, m * k * sizeof(double));

        // --- Parallel Run ---
        double start_parallel = omp_get_wtime();
        
        decompose_project_parallel(m, n, k, A, Result, num_threads);

        double end_parallel = omp_get_wtime();
        double time_parallel = end_parallel - start_parallel;
        printf("  Parallel: %.6fs\n", time_parallel);

        // Write to CSV: col, row, serial, parallel
        fprintf(fp, "%d,%d,%.6f,%.6f\n", n, m, time_serial, time_parallel);

        // Cleanup
        free(A);
        free(Result);
    }

    fclose(fp);
    printf("Results saved to %s\n", output_file);
    return 0;
}

